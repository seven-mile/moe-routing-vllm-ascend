#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark torch_npu.npu_grouped_matmul for the unquantized MoE MLP path.

This benchmark mirrors the kernel call shape used by the MC2 token dispatcher
and the unquantized MoE MLP implementation:

- one input tensor: [sum(M_i), K]
- one weight tensor: [num_groups, K, N] in storage, transposed before the call
- group_list passed in cumsum form with group_list_type=0
- split_item=2, group_type=0, no bias, no quantization

The CLI intentionally stays narrow:

- use --m-sizes 128,256,512,... to define the grouped GEMM problem sizes
- or use --uniform-m X --num-groups G to build a uniform list
- K and N are fixed for the run

Timing uses the standard warmup + torch.npu.Event pattern.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
from dataclasses import dataclass
from typing import Iterable

import torch
import torch_npu


DEFAULT_DTYPE = "bfloat16"
DEFAULT_DEVICE_ID = 0
DEFAULT_WARMUP_ITERS = 10
DEFAULT_ITERS = 50
KERNELS_PER_SAMPLE = 200


@dataclass(frozen=True)
class ProblemSpec:
    m_sizes: list[int]
    k: int
    n: int
    dtype: torch.dtype
    device_id: int
    warmup_iters: int
    iters: int
    seed: int


@dataclass(frozen=True)
class BenchResult:
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    std_us: float
    variance_us2: float
    effective_gflops: float


def _parse_dtype(value: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if value not in dtype_map:
        raise argparse.ArgumentTypeError(
            f"Unsupported dtype {value!r}; choose from float16, bfloat16, float32."
        )
    return dtype_map[value]


def _parse_csv_ints(value: str) -> list[int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")

    result: list[int] = []
    for part in parts:
        try:
            parsed = int(part)
        except ValueError as exc:  # pragma: no cover - defensive parsing.
            raise argparse.ArgumentTypeError(f"Invalid integer value {part!r}.") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"Problem sizes must be positive, got {parsed}.")
        result.append(parsed)
    return result


def _build_problem_sizes(args: argparse.Namespace) -> list[int]:
    if args.m_sizes is not None and args.uniform_m is not None:
        raise ValueError("Use only one of --m-sizes or --uniform-m.")
    if args.m_sizes is None and args.uniform_m is None:
        raise ValueError("One of --m-sizes or --uniform-m is required.")

    if args.m_sizes is not None:
        return _parse_csv_ints(args.m_sizes)

    if args.num_groups is None:
        raise ValueError("--num-groups is required when --uniform-m is used.")
    if args.num_groups <= 0:
        raise ValueError(f"--num-groups must be positive, got {args.num_groups}.")
    if args.uniform_m <= 0:
        raise ValueError(f"--uniform-m must be positive, got {args.uniform_m}.")
    return [int(args.uniform_m)] * int(args.num_groups)


def _to_cumsum_group_list(m_sizes: list[int], device: torch.device) -> torch.Tensor:
    cumulative = list(itertools.accumulate(m_sizes))
    return torch.tensor(cumulative, dtype=torch.int64, device=device)


def _build_inputs(spec: ProblemSpec) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    device = torch.device(f"npu:{spec.device_id}")
    torch.manual_seed(spec.seed)

    total_m = sum(spec.m_sizes)
    num_groups = len(spec.m_sizes)

    hidden_states = torch.randn((total_m, spec.k), dtype=spec.dtype, device=device)
    weight_storage = torch.randn((num_groups, spec.k, spec.n), dtype=spec.dtype, device=device)
    weight = weight_storage
    group_list = _to_cumsum_group_list(spec.m_sizes, device=device)

    return [hidden_states], [weight], group_list


def _run_kernel(x: list[torch.Tensor], weight: list[torch.Tensor], group_list: torch.Tensor) -> None:
    torch_npu.npu_grouped_matmul(
        x=x,
        weight=weight,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )


def _measure_once(
    x: list[torch.Tensor],
    weight: list[torch.Tensor],
    group_list: torch.Tensor,
    kernels_per_sample: int,
) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(kernels_per_sample):
        _run_kernel(x, weight, group_list)
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / kernels_per_sample * 1000.0


def _summarize(samples_us: Iterable[float], flops: float) -> BenchResult:
    values = list(samples_us)
    if not values:
        raise ValueError("At least one timing sample is required.")

    mean_us = statistics.mean(values)
    median_us = statistics.median(values)
    min_us = min(values)
    max_us = max(values)
    std_us = statistics.stdev(values) if len(values) > 1 else 0.0
    variance_us2 = statistics.variance(values) if len(values) > 1 else 0.0
    effective_gflops = flops / (mean_us * 1e3)
    return BenchResult(
        mean_us=mean_us,
        median_us=median_us,
        min_us=min_us,
        max_us=max_us,
        std_us=std_us,
        variance_us2=variance_us2,
        effective_gflops=effective_gflops,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch_npu.npu_grouped_matmul for grouped GEMM problem sizes."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--m-sizes",
        type=str,
        default=None,
        help="Comma-separated grouped GEMM M sizes, e.g. 128,256,512.",
    )
    group.add_argument(
        "--uniform-m",
        type=int,
        default=None,
        help="Repeat a uniform M size for all groups.",
    )

    parser.add_argument(
        "--num-groups",
        type=int,
        default=None,
        help="Number of groups to create when using --uniform-m.",
    )
    parser.add_argument("--k", type=int, required=True, help="Shared K dimension.")
    parser.add_argument("--n", type=int, required=True, help="Shared N dimension.")
    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=["float16", "bfloat16", "float32"],
        help="Input and weight dtype.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=DEFAULT_DEVICE_ID,
        help="NPU device id to run on.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=DEFAULT_WARMUP_ITERS,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="Number of repeated timing samples.",
    )
    return parser.parse_args()


def _validate_spec(spec: ProblemSpec) -> None:
    if not spec.m_sizes:
        raise ValueError("At least one M size is required.")
    if any(m <= 0 for m in spec.m_sizes):
        raise ValueError(f"All M sizes must be positive, got {spec.m_sizes}.")
    if spec.k <= 0:
        raise ValueError(f"K must be positive, got {spec.k}.")
    if spec.n <= 0:
        raise ValueError(f"N must be positive, got {spec.n}.")
    if spec.warmup_iters < 0:
        raise ValueError(f"warmup-iters must be >= 0, got {spec.warmup_iters}.")
    if spec.iters <= 0:
        raise ValueError(f"iters must be > 0, got {spec.iters}.")


def main() -> None:
    args = parse_args()
    m_sizes = _build_problem_sizes(args)
    dtype = _parse_dtype(args.dtype)
    spec = ProblemSpec(
        m_sizes=m_sizes,
        k=args.k,
        n=args.n,
        dtype=dtype,
        device_id=args.device_id,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        seed=args.seed,
    )
    _validate_spec(spec)

    torch.npu.set_device(spec.device_id)

    x, weight, group_list = _build_inputs(spec)
    total_m = sum(spec.m_sizes)
    flops = 2.0 * float(total_m) * float(spec.k) * float(spec.n)

    for _ in range(spec.warmup_iters):
        _run_kernel(x, weight, group_list)
    torch.npu.synchronize()

    samples_us: list[float] = []
    for _ in range(spec.iters):
        samples_us.append(_measure_once(x, weight, group_list, KERNELS_PER_SAMPLE))

    result = _summarize(samples_us, flops)

    print("Grouped GEMM benchmark", flush=True)
    print(f"device_id={spec.device_id}, dtype={spec.dtype}, m_sizes={spec.m_sizes}, k={spec.k}, n={spec.n}")
    print(
        f"measurement: kernels_per_sample={KERNELS_PER_SAMPLE}, repeats={spec.iters}, "
        f"warmup_kernels={spec.warmup_iters}"
    )
    print(f"group_list(cumsum)={group_list.tolist()}")
    print(
        f"avg={result.mean_us:.3f} us, median={result.median_us:.3f} us, "
        f"min={result.min_us:.3f} us, max={result.max_us:.3f} us, "
        f"std={result.std_us:.3f} us, var={result.variance_us2:.3f} us^2, "
        f"effective_gflops={result.effective_gflops:.3f}"
    )


if __name__ == "__main__":
    main()
