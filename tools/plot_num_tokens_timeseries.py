#!/usr/bin/env python3

import argparse
import datetime as dt
import math
import re
from collections import defaultdict
from typing import Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


TS_RE = re.compile(r"\b(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\b")
NUM_TOKENS_RE = re.compile(r"BatchDescriptor\(num_tokens=(\d+)")


def parse_points(log_path: str, year: int) -> list[Tuple[dt.datetime, int]]:
    points: list[Tuple[dt.datetime, int]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            ts_m = TS_RE.search(line)
            tok_m = NUM_TOKENS_RE.search(line)
            if not ts_m or not tok_m:
                continue
            ts = dt.datetime.strptime(f"{year}-{ts_m.group(1)}",
                                      "%Y-%m-%d %H:%M:%S")
            tok = int(tok_m.group(1))
            points.append((ts, tok))
    return points


def aggregate_per_second(
    points: list[Tuple[dt.datetime, int]]) -> list[Tuple[dt.datetime, float, int, int]]:
    grouped: dict[dt.datetime, list[int]] = defaultdict(list)
    for ts, tok in points:
        grouped[ts].append(tok)

    out: list[Tuple[dt.datetime, float, int, int]] = []
    for ts in sorted(grouped):
        vals = grouped[ts]
        avg = sum(vals) / len(vals)
        out.append((ts, avg, min(vals), max(vals)))
    return out


def downsample_idx(n: int, max_points: int) -> list[int]:
    if max_points <= 0 or n <= max_points:
        return list(range(n))
    step = int(math.ceil(n / max_points))
    return list(range(0, n, step))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot num_tokens over time from vLLM log.")
    parser.add_argument("log_file", help="Path to log file")
    parser.add_argument(
        "--mode",
        choices=["per-second", "raw"],
        default="per-second",
        help=(
            "per-second: aggregate same-second entries with avg/min/max (default); "
            "raw: plot each line directly"
        ),
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=3000,
        help="Downsample when points exceed this value (default: 3000)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=dt.datetime.now().year,
        help="Year used to parse MM-DD HH:MM:SS timestamps",
    )
    parser.add_argument(
        "--output",
        default="num_tokens_over_time.png",
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        default="num_tokens Over Time",
        help="Plot title",
    )
    args = parser.parse_args()

    points = parse_points(args.log_file, args.year)
    if not points:
        raise RuntimeError("No valid timestamp + num_tokens points found.")

    plt.figure(figsize=(13, 5))

    if args.mode == "per-second":
        sec_points = aggregate_per_second(points)
        idx = downsample_idx(len(sec_points), args.max_points)
        xs = [sec_points[i][0] for i in idx]
        ys_avg = [sec_points[i][1] for i in idx]
        ys_min = [sec_points[i][2] for i in idx]
        ys_max = [sec_points[i][3] for i in idx]

        plt.plot(xs, ys_avg, linewidth=1.2, label="avg num_tokens")
        plt.fill_between(xs,
                         ys_min,
                         ys_max,
                         alpha=0.2,
                         label="min-max in same second")
    else:
        idx = downsample_idx(len(points), args.max_points)
        xs = [points[i][0] for i in idx]
        ys = [points[i][1] for i in idx]
        plt.plot(xs, ys, linewidth=0.9, alpha=0.9, label="raw num_tokens")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("Time")
    plt.ylabel("num_tokens")
    plt.title(args.title)
    plt.grid(linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)

    print(f"Parsed {len(points)} raw points.")
    print(f"Saved time-series plot to: {args.output}")


if __name__ == "__main__":
    main()
