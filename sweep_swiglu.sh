#!/usr/bin/env bash
set -euo pipefail

SCRIPT=${SCRIPT:-benchmarks/ops/bench_swiglu.py}
LOG_DIR=${LOG_DIR:-results/swiglu/$(date +%Y%m%d_%H%M%S)}
DTYPE=${DTYPE:-bfloat16}
I=${I:-1536}
WARMUP=${WARMUP:-20}
ITERS=${ITERS:-100}

export VLLM_VERSION=0.16.0

mkdir -p "$LOG_DIR"

run_one() {
    local m="$1"
    local cap="$2"
    local tag="$3"
    local log="$LOG_DIR/${tag}_m${m}_cap${cap}_i${I}_${DTYPE}.log"

    echo "===== $tag m=$m cap=$cap I=$I dtype=$DTYPE =====" | tee -a "$LOG_DIR/all.log"
    python "$SCRIPT" \
        --num-tokens "$m" \
        --num-padded-tokens "$cap" \
        --intermediate-size "$I" \
        --dtype "$DTYPE" \
        --warmup-iters "$WARMUP" \
        --iters "$ITERS" \
        --kernels-per-sample 200 \
        2>&1 | tee "$log"

    cat "$log" >> "$LOG_DIR/all.log"
    echo "" >> "$LOG_DIR/all.log"
}

echo "Writing logs to: $LOG_DIR"

# 1) scaling curve: cap = M, no padding
# for m in 16 32 64 100 128 192 256 384 512 768 1024 1536 2048 3072 4096 8192; do
#     run_one "$m" "$m" "nopad"
# done

# 2) fixed active M=100, sweep padded capacity
# for cap in 128 192 256 384 512 768 1024 1536 2048 3072 4096 8192; do
#     run_one 100 "$cap" "pad_sweep"
# done

# 3) fixed padding factor 1.28x, roughly matching 100 -> 128
# for m in 100 128 192 256 384 512 768 1024 1536 2048 3072 4096; do
#     cap=$(( (m * 128 + 99) / 100 ))
#     run_one "$m" "$cap" "factor128"
# done

for m in 1 2 4 8 16 24 32 40 48 56 64 80 96 112 64 128 192 256 512 1024 2048; do
    # 2x padding factor
    n=$((m * 16))
    cap=$((n * 2))
    run_one "$n" "$cap" "pad_half_sweep_g16"
done

echo "Done. Combined log: $LOG_DIR/all.log"
