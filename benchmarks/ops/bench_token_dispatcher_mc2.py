#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark TokenDispatcherWithMC2 dispatch/combine throughput.

Run with torchrun, for example:

  torchrun --nproc_per_node=8 benchmarks/ops/bench_token_dispatcher_mc2.py \
      --ep-size 8 --num-tokens 8192 --hidden-size 7168 --top-k 8 --num-experts 128 \
      --dyn-topk-values 2,3,4,5,6,7,8

The dynamic top-k mask path is the key research target. This benchmark supports
both single-point and multi-point sweeps to profile performance curves.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Tuple

import torch
import torch.distributed as dist
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel,
                                             init_distributed_environment,
                                             initialize_model_parallel)

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.distributed.parallel_state import (destroy_ascend_model_parallel,
                                                    init_ascend_model_parallel)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (MoEQuantParams,
                                                       MoERoutingParams,
                                                       MoETokenDispatchInput)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend import utils as ascend_utils


@dataclass
class BenchResult:
    dispatch_ms: float
    combine_ms: float
    total_ms: float


@dataclass
class CurvePoint:
    target_dyn_topk: float | None
    actual_dyn_topk: float
    dispatch_ms: float
    combine_ms: float
    total_ms: float
    dispatch_tps: float
    combine_tps: float
    end2end_tps: float


def _parse_dtype(dtype: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_map[dtype]


def _safe_ep_size(args_ep_size: int | None, world_size: int) -> int:
    ep_size = world_size if args_ep_size is None else args_ep_size
    if ep_size != world_size:
        raise ValueError(
            f"This benchmark requires world_size == ep_size, got {world_size=} and {ep_size=}."
        )
    return ep_size


def _build_random_topk(num_tokens: int, top_k: int, num_experts: int,
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    # Use topk over random scores to guarantee unique experts per token.
    random_scores = torch.rand((num_tokens, num_experts), device=device)
    topk_weights, topk_ids = torch.topk(random_scores,
                                        k=top_k,
                                        dim=-1,
                                        largest=True,
                                        sorted=True)
    topk_weights = topk_weights / torch.clamp(
        topk_weights.sum(dim=-1, keepdim=True), min=1e-8)
    return topk_weights, topk_ids.to(torch.int32)


def _generate_exact_mean_sequence(target_mean: float, length: int, low: int,
                                  high: int,
                                  seed: int | None = None) -> torch.Tensor:
    """Generate integer sequence with mean close to target_mean.

    Sequence values are in [low, high]. The mean error is bounded by 1/length.
    """
    if length <= 0:
        return torch.empty((0, ), dtype=torch.int32)
    if low > high:
        raise ValueError(f"Invalid range: low={low} > high={high}.")

    target = min(max(target_mean, float(low)), float(high))
    low_i = int(math.floor(target))
    high_i = int(math.ceil(target))

    if low_i == high_i:
        return torch.full((length, ), low_i, dtype=torch.int32)

    high_frac = target - low_i
    num_high = int(round(high_frac * length))
    num_high = max(0, min(length, num_high))

    values = torch.full((length, ), low_i, dtype=torch.int32)
    if num_high > 0:
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        perm = torch.randperm(length, generator=g)
        values[perm[:num_high]] = high_i
    return values


def _apply_dyn_topk_mask(topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                         dyn_top_k: float | None,
                         seed: int | None = None) -> float:
    """Mask trailing experts per token so effective top-k matches dyn_top_k.

    Returns:
        The actual mean active top-k after masking.
    """
    num_tokens, top_k = topk_ids.shape
    if top_k <= 0:
        return 0.0
    if dyn_top_k is None:
        return float(top_k)

    if not (1.0 <= dyn_top_k <= float(top_k)):
        raise ValueError(
            f"dyn_top_k must be in [1, {top_k}], got {dyn_top_k}.")

    # j is active expert count per token in [1, top_k].
    j = _generate_exact_mean_sequence(target_mean=dyn_top_k,
                                      length=num_tokens,
                                      low=1,
                                      high=top_k,
                                      seed=seed).to(topk_ids.device)

    col = torch.arange(top_k, device=topk_ids.device).view(1, top_k)
    keep_until = j.view(num_tokens, 1)
    tail_mask = col >= keep_until

    topk_ids[tail_mask] = -1
    topk_weights[tail_mask] = 0.0

    actual_mean = float(j.to(torch.float32).mean().item())
    return actual_mean


def _reduce_scalar(value: float, op: dist.ReduceOp) -> float:
    # HCCL may not support kDouble for all_reduce on some environments.
    # Use all_gather with float32 and perform reduction locally.
    t = torch.tensor([value], dtype=torch.float32, device=torch.device("npu"))
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    vals = [x.item() for x in gathered]

    if op == dist.ReduceOp.MAX:
        return float(max(vals))
    if op == dist.ReduceOp.MIN:
        return float(min(vals))
    if op == dist.ReduceOp.SUM:
        return float(sum(vals))

    raise ValueError(f"Unsupported reduction op for scalar gather-reduce: {op}")


def _run_one_iter(dispatcher: TokenDispatcherWithMC2,
                  hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  expert_map: torch.Tensor) -> BenchResult:
    # Synchronize before each measured section to reduce timer noise.
    dist.barrier()
    torch.npu.synchronize()

    d_start = torch.npu.Event(enable_timing=True)
    d_end = torch.npu.Event(enable_timing=True)
    c_start = torch.npu.Event(enable_timing=True)
    c_end = torch.npu.Event(enable_timing=True)

    token_dispatch_input = MoETokenDispatchInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        routing=MoERoutingParams(
            expert_map=expert_map,
            global_redundant_expert_num=0,
            mc2_mask=None,
            apply_router_weight_on_input=False,
        ),
        quant=MoEQuantParams(
            quant_type=QuantType.NONE,
            comm_quant_mode=None,
            mxfp=None,
        ),
    )

    d_start.record()
    dispatch_out = dispatcher.token_dispatch(token_dispatch_input)
    d_end.record()
    torch.npu.synchronize()
    dispatch_ms = d_start.elapsed_time(d_end)

    c_start.record()
    _ = dispatcher.token_combine(hidden_states=dispatch_out.hidden_states,
                                 combine_metadata=dispatch_out.combine_metadata)
    c_end.record()
    torch.npu.synchronize()
    combine_ms = c_start.elapsed_time(c_end)

    return BenchResult(dispatch_ms=dispatch_ms,
                       combine_ms=combine_ms,
                       total_ms=dispatch_ms + combine_ms)


def _summarize(ms_values: list[float]) -> Tuple[float, float, float]:
    return statistics.mean(ms_values), statistics.median(ms_values), max(ms_values)


def _print_rank0(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def _parse_dyn_topk_values(dyn_topk: float | None,
                           dyn_topk_values: str | None,
                           top_k: int) -> list[float | None]:
    if dyn_topk is not None and dyn_topk_values is not None:
        raise ValueError("Use only one of --dyn-topk or --dyn-topk-values.")

    if dyn_topk_values is not None:
        values: list[float] = []
        for x in dyn_topk_values.split(","):
            v = float(x.strip())
            if not (1.0 <= v <= float(top_k)):
                raise ValueError(
                    f"Each dyn-topk value must be in [1, {top_k}], got {v}.")
            values.append(v)
        if not values:
            raise ValueError("--dyn-topk-values must not be empty.")
        return values

    if dyn_topk is not None:
        if not (1.0 <= dyn_topk <= float(top_k)):
            raise ValueError(
                f"dyn_top_k must be in [1, {top_k}], got {dyn_topk}.")
        return [dyn_topk]

    # Baseline: static top-k (no dynamic masking).
    return [None]


def _write_curve_csv(path: str, rows: Iterable[CurvePoint]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "target_dyn_topk",
            "actual_dyn_topk",
            "dispatch_ms",
            "combine_ms",
            "total_ms",
            "dispatch_tps",
            "combine_tps",
            "end2end_tps",
        ])
        for row in rows:
            writer.writerow([
                "none" if row.target_dyn_topk is None else f"{row.target_dyn_topk:.4f}",
                f"{row.actual_dyn_topk:.6f}",
                f"{row.dispatch_ms:.6f}",
                f"{row.combine_ms:.6f}",
                f"{row.total_ms:.6f}",
                f"{row.dispatch_tps:.6f}",
                f"{row.combine_tps:.6f}",
                f"{row.end2end_tps:.6f}",
            ])


def _build_minimal_runtime_vllm_config(max_capture_tokens: int,
                                       parallel_config: ParallelConfig
                                       ) -> SimpleNamespace:
    """Build minimal runtime config required by MC2 dispatcher.

    This avoids constructing full VllmConfig(), which triggers platform-wide
    checks that require model_config and are unnecessary for this micro-benchmark.
    """
    scheduler_config = SimpleNamespace(max_num_seqs=max_capture_tokens,
                                       decode_max_num_seqs=max_capture_tokens)
    pass_config = SimpleNamespace(enable_sp=False)
    compilation_config = SimpleNamespace(
        cudagraph_capture_sizes=[max_capture_tokens],
        max_cudagraph_capture_size=max_capture_tokens,
        pass_config=pass_config,
        compile_ranges_split_points=[],
    )

    return SimpleNamespace(
        additional_config={"refresh": True},
        scheduler_config=scheduler_config,
        compilation_config=compilation_config,
        speculative_config=None,
        parallel_config=parallel_config,
        kv_transfer_config=None,
        model_config=None,
        cache_config=SimpleNamespace(block_size=128),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TokenDispatcherWithMC2 dispatch/combine throughput."
    )
    parser.add_argument("--ep-size",
                        type=int,
                        default=None,
                        help="Expected EP size; must equal torchrun world size.")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-tokens",
                        type=int,
                        default=4096,
                        help="Tokens per rank.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--dtype",
                        type=str,
                        default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)

    parser.add_argument("--dyn-topk",
                        type=float,
                        default=None,
                        help="Single target mean for dynamic top-k masking.")
    parser.add_argument(
        "--dyn-topk-values",
        type=str,
        default=None,
        help=
        "Comma-separated target means for dynamic top-k sweep, e.g. '2,3,4,5,6,7,8'."
    )
    parser.add_argument("--curve-csv",
                        type=str,
                        default=None,
                        help="Optional output CSV path for performance curve.")

    return parser.parse_args()

from vllm.benchmarks.lib.utils import default_vllm_config

@default_vllm_config()
def main() -> None:
    args = parse_args()

    # Mock MoE predicate for this micro-benchmark to bypass model_config access.
    ascend_utils.is_moe_model = lambda *_args, **_kwargs: True
    ascend_utils.should_skip_allreduce_across_dp_group = lambda *_args, **_kwargs: True

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Please launch this script with torchrun.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    _ = _safe_ep_size(args.ep_size, world_size)

    if args.top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {args.top_k}.")
    if args.top_k > args.num_experts:
        raise ValueError(
            f"top_k must be <= num_experts, got {args.top_k=} and {args.num_experts=}."
        )
    if args.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts must be divisible by world_size for this benchmark, got {args.num_experts=} and {world_size=}."
        )

    dyn_topk_targets = _parse_dyn_topk_values(args.dyn_topk, args.dyn_topk_values,
                                              args.top_k)

    torch.npu.set_device(local_rank)
    init_distributed_environment(world_size=world_size,
                                 rank=rank,
                                 distributed_init_method="env://",
                                 local_rank=local_rank,
                                 backend="hccl")
    initialize_model_parallel(tensor_model_parallel_size=world_size,
                              pipeline_model_parallel_size=1,
                              decode_context_model_parallel_size=1,
                              backend="hccl")

    parallel_config = ParallelConfig(tensor_parallel_size=world_size,
                                     pipeline_parallel_size=1,
                                     data_parallel_size=1,
                                     prefill_context_parallel_size=1)

    # Keep this large enough so MC2 global_bs can cover benchmark token count.
    max_capture_tokens = max(1, args.num_tokens * world_size)
    runtime_vllm_config = _build_minimal_runtime_vllm_config(
        max_capture_tokens=max_capture_tokens,
        parallel_config=parallel_config,
    )

    try:
        dtype = _parse_dtype(args.dtype)
        device = torch.device("npu")
        torch.manual_seed(args.seed + rank)

        with set_current_vllm_config(runtime_vllm_config):
            init_ascend_config(runtime_vllm_config)
            init_ascend_model_parallel(parallel_config)

            hidden_states = torch.randn((args.num_tokens, args.hidden_size),
                                        dtype=dtype,
                                        device=device)
            base_topk_weights, base_topk_ids = _build_random_topk(
                args.num_tokens, args.top_k, args.num_experts, device)
            expert_map = torch.arange(args.num_experts,
                                      dtype=torch.int32,
                                      device=device)

            dispatcher = TokenDispatcherWithMC2(top_k=args.top_k,
                                                num_experts=args.num_experts)

            curve_rows: list[CurvePoint] = []

            _print_rank0("=" * 72)
            _print_rank0("TokenDispatcherWithMC2 Benchmark")
            _print_rank0("=" * 72)
            _print_rank0(
                f"world_size(ep=mc2): {world_size}, rank0 device: {torch.npu.current_device()}"
            )
            _print_rank0(
                f"shape: tokens/rank={args.num_tokens}, hidden_size={args.hidden_size}, top_k={args.top_k}, num_experts={args.num_experts}"
            )
            _print_rank0(
                f"dtype={args.dtype}, warmup={args.warmup_iters}, iters={args.iters}, dyn_topk_targets={dyn_topk_targets}"
            )
            _print_rank0("=" * 72)

            for idx, dyn_topk in enumerate(dyn_topk_targets):
                topk_weights = base_topk_weights.clone()
                topk_ids = base_topk_ids.clone()
                actual_dyn_topk = _apply_dyn_topk_mask(
                    topk_ids,
                    topk_weights,
                    dyn_topk,
                    seed=args.seed * 1009 + rank * 31 + idx,
                )
                actual_dyn_topk_avg = _reduce_scalar(actual_dyn_topk,
                                                     dist.ReduceOp.SUM) / world_size

                # Warmup
                for _ in range(args.warmup_iters):
                    _ = _run_one_iter(dispatcher, hidden_states, topk_weights,
                                      topk_ids, expert_map)

                dist.barrier()
                torch.npu.synchronize()

                dispatch_ms_list: list[float] = []
                combine_ms_list: list[float] = []
                total_ms_list: list[float] = []

                t0 = time.perf_counter()
                for _ in range(args.iters):
                    result = _run_one_iter(dispatcher, hidden_states,
                                           topk_weights, topk_ids, expert_map)
                    dispatch_ms_list.append(result.dispatch_ms)
                    combine_ms_list.append(result.combine_ms)
                    total_ms_list.append(result.total_ms)
                torch.npu.synchronize()
                dist.barrier()
                wall_s = time.perf_counter() - t0

                dispatch_mean, dispatch_p50, dispatch_pmax = _summarize(
                    dispatch_ms_list)
                combine_mean, combine_p50, combine_pmax = _summarize(
                    combine_ms_list)
                total_mean, total_p50, total_pmax = _summarize(total_ms_list)

                # Report synchronized max latency across ranks (robust for comm benchmarks).
                dispatch_mean_max = _reduce_scalar(dispatch_mean,
                                                   dist.ReduceOp.MAX)
                combine_mean_max = _reduce_scalar(combine_mean,
                                                  dist.ReduceOp.MAX)
                total_mean_max = _reduce_scalar(total_mean, dist.ReduceOp.MAX)

                global_tokens = args.num_tokens * world_size
                dispatch_tps = global_tokens / (dispatch_mean_max / 1e3)
                combine_tps = global_tokens / (combine_mean_max / 1e3)
                end2end_tps = global_tokens / (total_mean_max / 1e3)

                _print_rank0("-" * 72)
                _print_rank0(
                    f"dyn_topk target={dyn_topk}, actual_mean={actual_dyn_topk_avg:.4f}"
                )
                _print_rank0(
                    f"dispatch latency (ms): mean={dispatch_mean_max:.3f}, p50(local)={dispatch_p50:.3f}, max(local)={dispatch_pmax:.3f}"
                )
                _print_rank0(
                    f"combine  latency (ms): mean={combine_mean_max:.3f}, p50(local)={combine_p50:.3f}, max(local)={combine_pmax:.3f}"
                )
                _print_rank0(
                    f"total    latency (ms): mean={total_mean_max:.3f}, p50(local)={total_p50:.3f}, max(local)={total_pmax:.3f}"
                )
                _print_rank0(f"dispatch throughput: {dispatch_tps:,.2f} tokens/s")
                _print_rank0(f"combine  throughput: {combine_tps:,.2f} tokens/s")
                _print_rank0(f"end2end throughput: {end2end_tps:,.2f} tokens/s")
                _print_rank0(f"wall time (main loop): {wall_s:.3f} s")

                curve_rows.append(
                    CurvePoint(target_dyn_topk=dyn_topk,
                               actual_dyn_topk=actual_dyn_topk_avg,
                               dispatch_ms=dispatch_mean_max,
                               combine_ms=combine_mean_max,
                               total_ms=total_mean_max,
                               dispatch_tps=dispatch_tps,
                               combine_tps=combine_tps,
                               end2end_tps=end2end_tps))

            _print_rank0("=" * 72)
            _print_rank0("MC2 Dynamic Top-k Performance Curve (rank0)")
            _print_rank0("target_dyn_topk,actual_dyn_topk,total_ms,end2end_tps")
            for row in curve_rows:
                target = "none" if row.target_dyn_topk is None else f"{row.target_dyn_topk:.4f}"
                _print_rank0(
                    f"{target},{row.actual_dyn_topk:.4f},{row.total_ms:.4f},{row.end2end_tps:.2f}"
                )
            _print_rank0("=" * 72)

            if args.curve_csv is not None and (not dist.is_initialized()
                                               or dist.get_rank() == 0):
                _write_curve_csv(args.curve_csv, curve_rows)
                _print_rank0(f"Curve CSV written to: {args.curve_csv}")

    finally:
        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
