#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch_npu
from vllm_ascend.ops.triton.activation.swiglu_quant import swiglu_quant
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEFAULT_DTYPE = "bfloat16"
DEFAULT_DEVICE_ID = 0
DEFAULT_WARMUP_ITERS = 10
DEFAULT_ITERS = 50
DEFAULT_KERNELS_PER_SAMPLE = 10


@dataclass(frozen=True)
class BenchResult:
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    std_us: float
    variance_us2: float


def parse_dtype(s: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if s not in table:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {s}")
    return table[s]


def summarize(samples_us: Iterable[float]) -> BenchResult:
    xs = list(samples_us)
    return BenchResult(
        mean_us=statistics.mean(xs),
        median_us=statistics.median(xs),
        min_us=min(xs),
        max_us=max(xs),
        std_us=statistics.stdev(xs) if len(xs) > 1 else 0.0,
        variance_us2=statistics.variance(xs) if len(xs) > 1 else 0.0,
    )


def measure_once(fn: Callable[[], torch.Tensor], kernels_per_sample: int) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(kernels_per_sample):
        y = fn()
    end.record()
    torch.npu.synchronize()

    assert y is not None
    return start.elapsed_time(end) / kernels_per_sample * 1000.0


def bench(
    name: str,
    fn: Callable[[], torch.Tensor],
    warmup_iters: int,
    iters: int,
    kernels_per_sample: int,
) -> BenchResult:
    for _ in range(warmup_iters):
        fn()
    torch.npu.synchronize()

    samples = [measure_once(fn, kernels_per_sample) for _ in range(iters)]
    result = summarize(samples)

    print(
        f"{name:28s} "
        f"avg={result.mean_us:.3f} us, "
        f"median={result.median_us:.3f} us, "
        f"min={result.min_us:.3f} us, "
        f"max={result.max_us:.3f} us, "
        f"std={result.std_us:.3f} us, "
        f"var={result.variance_us2:.3f} us^2",
        flush=True,
    )
    return result


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def max_rel_diff(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    aa = a.float()
    bb = b.float()
    return ((aa - bb).abs() / bb.abs().clamp_min(eps)).max().item()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark torch_npu.npu_swiglu vs npu_clipped_swiglu(group_index)."
    )
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--num-padded-tokens", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--kernels-per-sample", type=int, default=DEFAULT_KERNELS_PER_SAMPLE)
    parser.add_argument("--limit", type=float, default=math.inf)
    args = parser.parse_args()

    if args.num_tokens <= 0:
        raise ValueError("--num-tokens must be positive")
    if args.num_padded_tokens < args.num_tokens:
        raise ValueError("--num-padded-tokens must be >= --num-tokens")
    if args.intermediate_size <= 0:
        raise ValueError("--intermediate-size must be positive")

    dtype = parse_dtype(args.dtype)
    torch.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    torch.manual_seed(args.seed)

    init_device_properties_triton()

    I = args.intermediate_size
    x = torch.randn(
        (args.num_padded_tokens, 2 * I),
        dtype=dtype,
        device=device,
    ).contiguous()

    if args.num_padded_tokens > args.num_tokens:
        x[args.num_tokens:, :] = torch.randn_like(x[args.num_tokens:, :]) * 3.0

    x_active = x[: args.num_tokens, :].contiguous()
    group_index = torch.tensor([args.num_tokens], dtype=torch.int64, device=device)

    def run_swiglu_full() -> torch.Tensor:
        return torch_npu.npu_swiglu(x, dim=-1)

    def run_clipped_active() -> torch.Tensor:
        return torch_npu.npu_clipped_swiglu(
            x,
            group_index=group_index,
            dim=-1,
            alpha=1.0,
            limit=args.limit,
            bias=0.0,
            interleaved=False,
        )

    def run_swiglu_active() -> torch.Tensor:
        return torch_npu.npu_swiglu(x_active, dim=-1)

    def run_swiglu_torch_full() -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return x1 * torch.sigmoid(x1) * x2

    def run_swiglu_torch_active() -> torch.Tensor:
        x1, x2 = torch.chunk(x_active, 2, dim=-1)
        return x1 * torch.sigmoid(x1) * x2

    def run_swiglu_quant() -> torch.Tensor:
        # group_list_type=0 (cusum). Use a single-entry cumsum-style list
        # where the value equals the number of active tokens (no padding).
        group_list = torch.tensor([args.num_tokens], dtype=torch.int64, device=device)
        out, _ = swiglu_quant(x, group_list, 0, need_quant=False)
        # swiglu_quant returns [s, h//2] output; we want full swiglu output shape
        return out

    torch.npu.synchronize()

    # y_full = run_swiglu_full()
    # y_clip = run_clipped_active()
    # y_clip = y_clip[: args.num_tokens, :]
    # y_ref = run_swiglu_active()
    # y_quant = run_swiglu_quant()
    # y_torch = run_swiglu_torch_active()
    # torch.npu.synchronize()

    # y_full_active = y_full[: args.num_tokens, :]

    print("SwiGLU microbenchmark", flush=True)
    print(
        f"device_id={args.device_id}, dtype={dtype}, "
        f"num_tokens={args.num_tokens}, "
        f"num_padded_tokens={args.num_padded_tokens}, "
        f"intermediate_size={I}, input_shape={tuple(x.shape)}",
        flush=True,
    )
    print(
        f"clipped params: group_index={group_index.tolist()}, "
        f"dim=-1, alpha=1.0, limit={args.limit}, bias=0.0, interleaved=False",
        flush=True,
    )
    print(
        f"measurement: kernels_per_sample={args.kernels_per_sample}, "
        f"repeats={args.iters}, warmup_calls={args.warmup_iters}",
        flush=True,
    )

    # print("\nCorrectness against npu_swiglu(x_active)", flush=True)
    # print(f"ref shape          = {tuple(y_ref.shape)}", flush=True)
    # print(f"clipped shape      = {tuple(y_clip.shape)}", flush=True)
    # print(f"full active shape  = {tuple(y_full_active.shape)}", flush=True)
    # print(f"quant shape        = {tuple(y_quant.shape)}", flush=True)
    # print(f"clip max_abs_diff  = {max_abs_diff(y_clip, y_ref):.8e}", flush=True)
    # print(f"clip max_rel_diff  = {max_rel_diff(y_clip, y_ref):.8e}", flush=True)
    # print(f"full max_abs_diff  = {max_abs_diff(y_full_active, y_ref):.8e}", flush=True)
    # print(f"full max_rel_diff  = {max_rel_diff(y_full_active, y_ref):.8e}", flush=True)
    # print(f"quant max_abs_diff = {max_abs_diff(y_quant[: args.num_tokens, :], y_ref):.8e}", flush=True)
    # print(f"quant max_rel_diff = {max_rel_diff(y_quant[: args.num_tokens, :], y_ref):.8e}", flush=True)
    # print(f"torch max_abs_diff = {max_abs_diff(y_torch, y_ref):.8e}", flush=True)
    # print(f"torch max_rel_diff = {max_rel_diff(y_torch, y_ref):.8e}", flush=True)

    print("\nTiming", flush=True)
    r_full = bench(
        "npu_swiglu(full padded)",
        run_swiglu_full,
        args.warmup_iters,
        args.iters,
        args.kernels_per_sample,
    )
    r_clip = bench(
        "npu_clipped_swiglu(active)",
        run_clipped_active,
        args.warmup_iters,
        args.iters,
        args.kernels_per_sample,
    )
    r_active = bench(
        "npu_swiglu(active)",
        run_swiglu_active,
        args.warmup_iters,
        args.iters,
        args.kernels_per_sample,
    )
    # r_torch_full = bench(
    #     "torch_swiglu(full padded)",
    #     run_swiglu_torch_full,
    #     args.warmup_iters,
    #     args.iters,
    #     args.kernels_per_sample,
    # )
    # r_torch_active = bench(
    #     "torch_swiglu(active)",
    #     run_swiglu_torch_active,
    #     args.warmup_iters,
    #     args.iters,
    #     args.kernels_per_sample,
    # )
    # r_quant = bench(
    #     "swiglu_quant(cumsum,no_scale)",
    #     run_swiglu_quant,
    #     args.warmup_iters,
    #     args.iters,
    #     args.kernels_per_sample,
    # )

    print("\nRatios", flush=True)
    print(f"full_padded / clipped_active = {r_full.mean_us / r_clip.mean_us:.3f}x")
    print(f"active / clipped_active      = {r_active.mean_us / r_clip.mean_us:.3f}x")
    # print(f"full_padded / torch_full      = {r_full.mean_us / r_torch_full.mean_us:.3f}x")
    # print(f"active / torch_active         = {r_active.mean_us / r_torch_active.mean_us:.3f}x")
    # print(f"full_padded / quant           = {r_full.mean_us / r_quant.mean_us:.3f}x")
    print(f"padding_factor               = {args.num_padded_tokens / args.num_tokens:.3f}x")


if __name__ == "__main__":
    main()
