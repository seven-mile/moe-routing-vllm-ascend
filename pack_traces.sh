#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Usage: $0 /tmp/vllm_profile/optimum_c256}"
ROOT="${ROOT%/}"

OUT_ZIP="${2:-${ROOT##*/}_traces.zip}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

shopt -s nullglob

tars=()

while IFS= read -r -d '' trace; do
    rank_dir="$(basename "$(dirname "$(dirname "$trace")")")"
    tar_path="$TMPDIR/${rank_dir}.tar.gz"

    echo "Packing $rank_dir"

    tar -C "$ROOT" \
        -czf "$tar_path" \
        "${rank_dir}/ASCEND_PROFILER_OUTPUT/trace_view.json"

    tars+=("$tar_path")
done < <(
    find "$ROOT" \
        -path "*/ASCEND_PROFILER_OUTPUT/trace_view.json" \
        -type f \
        -print0
)

if ((${#tars[@]} == 0)); then
    echo "No trace_view.json found under $ROOT" >&2
    exit 1
fi

rm -f "$OUT_ZIP"

# -0 表示 zip archive/store 模式，不再压缩 tar.gz
zip -0 -j "$OUT_ZIP" "${tars[@]}"

echo "Wrote: $OUT_ZIP"
