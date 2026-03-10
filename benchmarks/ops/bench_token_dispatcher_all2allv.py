#!/usr/bin/env python3
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark TokenDispatcherWithAll2AllV dispatch/combine throughput.

Run with torchrun, for example:

  torchrun --nproc_per_node=8 benchmarks/ops/bench_token_dispatcher_all2allv.py \
      --ep-size 8 --num-tokens 8192 --hidden-size 7168 --top-k 8 --num-experts 128
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel,
                                             init_distributed_environment,
                                             initialize_model_parallel)
from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel
from vllm_ascend.ops.moe.token_dispatcher import TokenDispatcherWithAll2AllV


@dataclass
class BenchResult:
    dispatch_ms: float
    combine_ms: float
    total_ms: float


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
    return topk_weights, topk_ids.to(torch.int64)


def _generate_exact_mean_sequence(target_mean=5.66, length=4096):
    """
    通过解方程确定每个数字的数量
    """
    import numpy as np
  
    # 假设我们想要这样的分布：
    # 设各数字的数量为：4:a, 5:b, 6:c, 7:d, 8:e
    # 约束：a+b+c+d+e = 4096
    #       4a+5b+6c+7d+8e = 4096*5.66
    
    # 简化：先确定大概比例，然后用随机抽样
    total = length * target_mean  # 需要的总和
    
    # 初始设定比例（可根据需要调整）
    proportions = [0.10, 0.25, 0.30, 0.25, 0.10]  # 4,5,6,7,8的比例
    
    while True:
        # 计算数量（取整）
        counts = [int(length * p) for p in proportions]
        # 调整使总和为length
        diff = length - sum(counts)
        if diff > 0:
            counts[np.argmax(proportions)] += diff
        elif diff < 0:
            counts[np.argmin(proportions)] += diff
        
        # 创建序列
        sequence = []
        for val, count in zip([4,5,6,7,8], counts):
            sequence.extend([val] * count)
        
        # 转为numpy数组并打乱顺序
        sequence = np.array(sequence)
        np.random.shuffle(sequence)
        
        actual_mean = np.mean(sequence)
        
        # 如果足够接近，返回
        if abs(actual_mean - target_mean) <= 0.01:
            return sequence, actual_mean
        
        # 否则调整比例
        if actual_mean < target_mean:
            # 增加大数的比例
            proportions[-1] += 0.01
            proportions[0] -= 0.01
        else:
            # 增加小数的比例
            proportions[0] += 0.01
            proportions[-1] -= 0.01
        
        # 保持比例在合理范围内
        proportions = [max(0.05, min(0.35, p)) for p in proportions]
        # 归一化
        proportions = [p/sum(proportions) for p in proportions]


def _apply_dyn_topk_mask(topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                         dyn_top_k: float | None) -> None:
    """Mask trailing j experts per token with j sampled independently."""
    num_tokens, top_k = topk_ids.shape
    if top_k <= 0:
        return

    if dyn_top_k is None:
        return

    # j in [low, high], token-wise and independent.
    j_np, actual_mean = _generate_exact_mean_sequence(target_mean=dyn_top_k, length=num_tokens)
    print(f"Generate actual mean top k: {actual_mean}")
    j = torch.from_numpy(j_np).to(topk_ids.device)
    col = torch.arange(top_k, device=topk_ids.device).view(1, top_k)
    keep_until = j.view(num_tokens, 1)
    tail_mask = col >= keep_until

    topk_ids[tail_mask] = -1
    topk_weights[tail_mask] = 0.0


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


def _run_one_iter(dispatcher: TokenDispatcherWithAll2AllV,
                  hidden_states: torch.Tensor, topk_weights: torch.Tensor,
                  topk_ids: torch.Tensor,
                  expert_map: torch.Tensor) -> Tuple[BenchResult, Dict[str, float]]:
    # Synchronize before each measured section to reduce timer noise.
    dist.barrier()
    torch.npu.synchronize()

    d_start = torch.npu.Event(enable_timing=True)
    d_end = torch.npu.Event(enable_timing=True)
    c_start = torch.npu.Event(enable_timing=True)
    c_end = torch.npu.Event(enable_timing=True)

    d_start.record()
    out = dispatcher.token_dispatch(hidden_states=hidden_states,
                                    topk_weights=topk_weights,
                                    topk_ids=topk_ids,
                                    expert_map=expert_map)
    d_end.record()
    torch.npu.synchronize()
    dispatch_ms = d_start.elapsed_time(d_end)

    # Approximate one-way comm bytes from split sizes in this rank.
    send_tokens_dispatch = int(dispatcher.input_splits.sum())
    recv_tokens_dispatch = int(dispatcher.output_splits.sum())
    elem_size = out["hidden_states"].element_size()
    hidden_size = out["hidden_states"].shape[-1]
    dispatch_send_bytes = send_tokens_dispatch * hidden_size * elem_size
    dispatch_recv_bytes = recv_tokens_dispatch * hidden_size * elem_size

    c_start.record()
    _ = dispatcher.token_combine(out["hidden_states"])
    c_end.record()
    torch.npu.synchronize()
    combine_ms = c_start.elapsed_time(c_end)

    # token_combine resets split metadata, so use mirrored stats for combine phase.
    combine_send_bytes = dispatch_recv_bytes
    combine_recv_bytes = dispatch_send_bytes

    total_ms = dispatch_ms + combine_ms
    return BenchResult(dispatch_ms=dispatch_ms,
                       combine_ms=combine_ms,
                       total_ms=total_ms), {
                           "dispatch_send_bytes": float(dispatch_send_bytes),
                           "dispatch_recv_bytes": float(dispatch_recv_bytes),
                           "combine_send_bytes": float(combine_send_bytes),
                           "combine_recv_bytes": float(combine_recv_bytes),
                       }


def _summarize(ms_values: list[float]) -> Tuple[float, float, float]:
    return statistics.mean(ms_values), statistics.median(ms_values), max(ms_values)


def _print_rank0(msg: str) -> None:
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TokenDispatcherWithAll2AllV dispatch/combine throughput."
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
                        help="Target mean for dynamic top-k masking.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    num_local_experts = args.num_experts // world_size

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

    try:
        dtype = _parse_dtype(args.dtype)
        device = torch.device("npu")
        torch.manual_seed(args.seed + rank)

        hidden_states = torch.randn((args.num_tokens, args.hidden_size),
                                    dtype=dtype,
                                    device=device)
        topk_weights, topk_ids = _build_random_topk(args.num_tokens, args.top_k,
                                                     args.num_experts, device)
        if args.dyn_topk is not None:
            _apply_dyn_topk_mask(topk_ids, topk_weights, args.dyn_topk)

        expert_map = torch.arange(args.num_experts,
                                  dtype=torch.int64,
                                  device=device)

        dispatcher = TokenDispatcherWithAll2AllV(top_k=args.top_k,
                                                 num_experts=args.num_experts,
                                                 num_local_experts=num_local_experts,
                                                 with_quant=False)

        # Warmup
        for _ in range(args.warmup_iters):
            _ = _run_one_iter(dispatcher, hidden_states, topk_weights, topk_ids,
                              expert_map)

        dist.barrier()
        torch.npu.synchronize()

        dispatch_ms_list: list[float] = []
        combine_ms_list: list[float] = []
        total_ms_list: list[float] = []
        dispatch_send_bytes_list: list[float] = []
        dispatch_recv_bytes_list: list[float] = []
        combine_send_bytes_list: list[float] = []
        combine_recv_bytes_list: list[float] = []

        t0 = time.perf_counter()
        for _ in range(args.iters):
            result, comm_stats = _run_one_iter(dispatcher, hidden_states,
                                               topk_weights, topk_ids,
                                               expert_map)
            dispatch_ms_list.append(result.dispatch_ms)
            combine_ms_list.append(result.combine_ms)
            total_ms_list.append(result.total_ms)
            dispatch_send_bytes_list.append(comm_stats["dispatch_send_bytes"])
            dispatch_recv_bytes_list.append(comm_stats["dispatch_recv_bytes"])
            combine_send_bytes_list.append(comm_stats["combine_send_bytes"])
            combine_recv_bytes_list.append(comm_stats["combine_recv_bytes"])
        torch.npu.synchronize()
        dist.barrier()
        wall_s = time.perf_counter() - t0

        dispatch_mean, dispatch_p50, dispatch_pmax = _summarize(dispatch_ms_list)
        combine_mean, combine_p50, combine_pmax = _summarize(combine_ms_list)
        total_mean, total_p50, total_pmax = _summarize(total_ms_list)

        # Report synchronized max latency across ranks (robust for comm benchmarks).
        dispatch_mean_max = _reduce_scalar(dispatch_mean, dist.ReduceOp.MAX)
        combine_mean_max = _reduce_scalar(combine_mean, dist.ReduceOp.MAX)
        total_mean_max = _reduce_scalar(total_mean, dist.ReduceOp.MAX)

        global_tokens = args.num_tokens * world_size
        dispatch_tps = global_tokens / (dispatch_mean_max / 1e3)
        combine_tps = global_tokens / (combine_mean_max / 1e3)
        end2end_tps = global_tokens / (total_mean_max / 1e3)

        # Aggregate communication bandwidth estimates.
        dispatch_send_b = _reduce_scalar(statistics.mean(dispatch_send_bytes_list),
                                         dist.ReduceOp.SUM)
        dispatch_recv_b = _reduce_scalar(statistics.mean(dispatch_recv_bytes_list),
                                         dist.ReduceOp.SUM)
        combine_send_b = _reduce_scalar(statistics.mean(combine_send_bytes_list),
                                        dist.ReduceOp.SUM)
        combine_recv_b = _reduce_scalar(statistics.mean(combine_recv_bytes_list),
                                        dist.ReduceOp.SUM)

        dispatch_gbps = (dispatch_send_b + dispatch_recv_b) / (dispatch_mean_max / 1e3) / 1e9
        combine_gbps = (combine_send_b + combine_recv_b) / (combine_mean_max / 1e3) / 1e9

        _print_rank0("=" * 72)
        _print_rank0("TokenDispatcherWithAll2AllV Benchmark")
        _print_rank0("=" * 72)
        _print_rank0(
            f"world_size(ep=dp): {world_size}, rank0 device: {torch.npu.current_device()}")
        _print_rank0(
            f"shape: tokens/rank={args.num_tokens}, hidden_size={args.hidden_size}, top_k={args.top_k}, num_experts={args.num_experts}, num_local_experts={num_local_experts}")
        _print_rank0(
            f"dtype={args.dtype}, dyn_topk={args.dyn_topk}, warmup={args.warmup_iters}, iters={args.iters}")
        _print_rank0("-" * 72)
        _print_rank0(
            f"dispatch latency (ms): mean={dispatch_mean_max:.3f}, p50(local)={dispatch_p50:.3f}, max(local)={dispatch_pmax:.3f}")
        _print_rank0(
            f"combine  latency (ms): mean={combine_mean_max:.3f}, p50(local)={combine_p50:.3f}, max(local)={combine_pmax:.3f}")
        _print_rank0(
            f"total    latency (ms): mean={total_mean_max:.3f}, p50(local)={total_p50:.3f}, max(local)={total_pmax:.3f}")
        _print_rank0("-" * 72)
        _print_rank0(f"dispatch throughput: {dispatch_tps:,.2f} tokens/s")
        _print_rank0(f"combine  throughput: {combine_tps:,.2f} tokens/s")
        _print_rank0(f"end2end throughput: {end2end_tps:,.2f} tokens/s")
        _print_rank0(f"dispatch estimated BW: {dispatch_gbps:.3f} GB/s")
        _print_rank0(f"combine  estimated BW: {combine_gbps:.3f} GB/s")
        _print_rank0(f"wall time (main loop): {wall_s:.3f} s")
        _print_rank0("=" * 72)

    finally:
        destroy_ascend_model_parallel()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
