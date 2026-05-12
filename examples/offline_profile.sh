#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 开启离线模式标志
export TRANSFORMERS_OFFLINE=1

# 定义全局请求数
GLOBAL_BATCH_SIZE=384

echo "Starting Offline Profiling with Data Parallel Size 8..."

export VLLM_VERSION=0.16.0

python $SCRIPT_DIR/offline_profile.py \
    --model /root/models/Qwen/Qwen3-30B-A3B \
    --seed 42 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --enable-expert-parallel \
    --data-parallel-size 8 \
    --speculative-config '{"model": "/root/models/Qwen/Qwen3-0.6B", "method": "draft_model", "num_speculative_tokens": 3}' \
    --no-async-scheduling \
    --dataset /root/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --max-output-len 32 \
    --timeout 1200 \
    --compilation_config.cudagraph_mode FULL_DECODE_ONLY \
    --additional_config.ascend_compilation_config.fuse_qknorm_rope false \
    --profile \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/vllm_profile/current"}' \

# python examples/offline_inference/data_parallel.py \
#     --model /tmp/misc/Qwen3-30B-A3B \
#     --seed 42 \
#     --max-model-len 4096 \
#     --gpu-memory-utilization 0.9 \
#     --all2all-backend deepep_low_latency \
#     -dp=8 -ep \
#     --speculative-config '{"model": "/tmp/misc/Qwen3-0.6B", "method": "draft_model", "num_speculative_tokens": 3, "disable_padded_drafter_batch": false}' \
#     --enforce-eager