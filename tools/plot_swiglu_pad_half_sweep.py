import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATH = Path(
    "/root/mazhi/repos/vllm-ascend/results/swiglu/20260511_100249/all.log"
)
OUT_PATH = Path(
    "/root/mazhi/repos/vllm-ascend/results/swiglu/20260511_100249/"
    "swiglu_pad_half_sweep_timing.png"
)

SECTION_RE = re.compile(
    r"^===== .* m=(\d+) cap=(\d+) I=(\d+) dtype=([a-z0-9]+) =====$"
)
TIMING_RE = re.compile(
    r"^(npu_[^(]+\([^)]+\))\s+avg=([0-9.]+) us, .* std=([0-9.]+) us"
)

SERIES = {
    "npu_swiglu(full padded)": {
        "key": "full_padded",
        "label": "full padded",
        "color": "#1f77b4",
        "marker": "o",
    },
    "npu_clipped_swiglu(active)": {
        "key": "clipped_active",
        "label": "clipped active",
        "color": "#d62728",
        "marker": "s",
    },
    "npu_swiglu(active)": {
        "key": "active",
        "label": "active",
        "color": "#2ca02c",
        "marker": "^",
    },
}


def parse_log(text: str) -> tuple[dict[str, dict[str, list[float]]], dict[str, str]]:
    data: dict[str, dict[str, list[float]]] = {
        series["key"]: {"m": [], "avg": [], "std": []} for series in SERIES.values()
    }
    meta: dict[str, str] = {}
    pairs: list[tuple[int, int]] = []

    current_m: int | None = None
    for line in text.splitlines():
        line = line.strip()
        section = SECTION_RE.match(line)
        if section:
            current_m = int(section.group(1))
            pairs.append((current_m, int(section.group(2))))
            meta.setdefault("I", section.group(3))
            meta.setdefault("dtype", section.group(4))
            continue

        timing = TIMING_RE.match(line)
        if timing and current_m is not None:
            name = timing.group(1)
            if name in SERIES:
                series_key = SERIES[name]["key"]
                data[series_key]["m"].append(current_m)
                data[series_key]["avg"].append(float(timing.group(2)))
                data[series_key]["std"].append(float(timing.group(3)))

    if pairs:
        is_cap_double = all(cap == 2 * m for m, cap in pairs)
        meta["cap_relation"] = "cap=2m (2x padding)" if is_cap_double else "cap varies"

    return data, meta


def main() -> None:
    text = LOG_PATH.read_text(encoding="utf-8")
    data, meta = parse_log(text)

    if not any(values["m"] for values in data.values()):
        raise SystemExit("No timing entries found in log.")

    fig, ax = plt.subplots(figsize=(9.4, 5.4))

    all_m = set()
    for series_name, series in SERIES.items():
        key = series["key"]
        m_values = data[key]["m"]
        avg_values = data[key]["avg"]
        std_values = data[key]["std"]
        if not m_values:
            continue
        all_m.update(m_values)
        ax.errorbar(
            m_values,
            avg_values,
            yerr=std_values,
            fmt=f"{series['marker']}-",
            color=series["color"],
            capsize=3,
            linewidth=1.6,
            markersize=5,
            label=series["label"],
        )

    all_m_sorted = sorted(all_m)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_m_sorted)
    ax.set_xticklabels([str(m) for m in all_m_sorted])

    for m in all_m_sorted:
        max_y = None
        for series in SERIES.values():
            key = series["key"]
            if m in data[key]["m"]:
                idx = data[key]["m"].index(m)
                y_val = data[key]["avg"][idx]
                max_y = y_val if max_y is None else max(max_y, y_val)
        if max_y is not None:
            ax.annotate(
                str(m),
                (m, max_y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

    i_value = meta.get("I", "?")
    dtype = meta.get("dtype", "?")
    cap_relation = meta.get("cap_relation", "cap varies")
    title = (
        "SwiGLU pad_half_sweep timing (m on x-axis, "
        f"{cap_relation}, I={i_value}, dtype={dtype})"
    )

    ax.set_xlabel("m (log2 scale)")
    ax.set_ylabel("avg latency (us)")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)


if __name__ == "__main__":
    main()
