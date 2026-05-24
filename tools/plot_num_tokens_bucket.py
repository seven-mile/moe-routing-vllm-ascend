#!/usr/bin/env python3

import argparse
import re
from collections import Counter

import matplotlib.pyplot as plt


NUM_TOKENS_RE = re.compile(r"BatchDescriptor\(num_tokens=(\d+)")


def parse_num_tokens(log_path: str) -> list[int]:
    values: list[int] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = NUM_TOKENS_RE.search(line)
            if m:
                values.append(int(m.group(1)))
    return values


def bucket_upper_bound(value: int, bucket_size: int) -> int:
    # Bucket definition is (prev, k*bucket_size], i.e. upper bound is inclusive.
    return ((value + bucket_size - 1) // bucket_size) * bucket_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot bucketed count of num_tokens from vLLM log.")
    parser.add_argument("log_file", help="Path to log file")
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=256,
        help="Bucket size (default: 256). Buckets are (prev, k*bucket_size].",
    )
    parser.add_argument(
        "--output",
        default="num_tokens_bucket_count.png",
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        default="num_tokens Bucket Count",
        help="Plot title",
    )
    args = parser.parse_args()

    if args.bucket_size <= 0:
        raise ValueError("--bucket-size must be > 0")

    values = parse_num_tokens(args.log_file)
    if not values:
        raise RuntimeError("No num_tokens found. Check log format or input path.")

    counter: Counter[int] = Counter(
        bucket_upper_bound(v, args.bucket_size) for v in values)

    max_upper = max(counter)
    uppers = list(range(args.bucket_size, max_upper + args.bucket_size,
                        args.bucket_size))
    counts = [counter.get(upper, 0) for upper in uppers]

    plt.figure(figsize=(12, 5))
    plt.bar(uppers, counts, width=args.bucket_size * 0.8, align="center")
    plt.xlabel(f"Bucket Upper Bound (<= k*{args.bucket_size})")
    plt.ylabel("Count")
    plt.title(args.title)
    plt.xticks(uppers, [f"<= {x}" for x in uppers], rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)

    print(f"Parsed {len(values)} points.")
    print(f"Saved bucket count plot to: {args.output}")


if __name__ == "__main__":
    main()
