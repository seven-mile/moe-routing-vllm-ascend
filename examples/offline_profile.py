# SPDX-License-Identifier: Apache-2.0
"""
Offline Inference Profiling Script for EP + Speculative Decoding + DP
"""
import os
import json
import random
from time import sleep
from multiprocessing import Process

from vllm import LLM, EngineArgs, SamplingParams
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.network_utils import get_open_port

def create_parser():
    parser = FlexibleArgumentParser(description="Offline Profile for EP + SpecDecode")

    # 载入所有 vLLM Engine 参数 (包含了模型路径、SpecDecode、EP等配置)
    EngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=128,
        help="Total number of requests to process across all DP ranks.",
    )
    parser.add_argument(
        "--max-output-len", 
        type=int, 
        default=128, 
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Path to the ShareGPT JSON dataset."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling.",
    )

    # DP 协调参数
    parser.add_argument("--dp-num-nodes", type=int, default=1)
    parser.add_argument("--dp-node-rank", type=int, default=0)
    parser.add_argument("--dp-master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--dp-master-port", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)

    return parser

def main(
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    global_batch_size,
    prompts,
    max_output_len,
    profile,
    engine_args,
):
    # 配置 DP 所需的环境变量
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # 按照 DP Size 均分 prompts
    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    def get_start_idx(rank):
        return rank * floor + min(rank, remainder)

    start_idx = get_start_idx(global_dp_rank)
    end_idx = get_start_idx(global_dp_rank + 1)
    local_prompts = prompts[start_idx:end_idx]

    if not local_prompts:
        local_prompts = ["Placeholder prompt for empty rank."]

    print(f"[DP Rank {global_dp_rank}] Starting initialization. Allocated {len(local_prompts)} prompts.")

    from vllm.utils.udf import UserDefinedFunctionConfig

    cfg_baseline = UserDefinedFunctionConfig(
        file="/root/mazhi/ppl_to_ks.py",
        function="baseline",
    )

    cfg_lossless = UserDefinedFunctionConfig(
        file="/root/mazhi/ppl_to_ks.py",
        function="spec_with_list_layer_range",
        args=[
            [16.342797244301977, 16.28686882612569, 16.28686882612569, 14.62508916637383],
            [0, 0],
        ],
    )

    cfg_optimum = UserDefinedFunctionConfig(
        file="/root/mazhi/ppl_to_ks.py",
        function="spec_with_list_layer_range",
        args=[
            [16.22663120720821, 16.22663120720821, 11.861973917295447, 11.861973917295447, 7.394487721217839],
            [0, 0],
        ],
    )

    # 采样参数：使用传入的 max_output_len 替代硬编码
    sampling_params = SamplingParams(
        temperature=0.0, # 设为 0 以保证确定性，有助于 debug
        max_tokens=max_output_len, 
        ignore_eos=True,  # 忽略结束符，强制生成指定长度方便衡量 throughput
        dyn_assisted_action_config_str=cfg_baseline.dumps(),
    )

    # 初始化 LLM 引擎
    llm = LLM(**engine_args)
    
    print(f"[DP Rank {global_dp_rank}] Engine initialized. Starting generation...")
    
    # 同步推理
    if profile:
        llm.start_profile()

    outputs = llm.generate(local_prompts, sampling_params)

    if profile:
        llm.stop_profile()
    
    # 展示前 5 个输出
    print(f"\n--- [DP Rank {global_dp_rank}] Top 5 Outputs ---")
    for i, output in enumerate(outputs):
        if i >= 5:
            break
        generated_text = output.outputs[0].text
        print(f"[{global_dp_rank}-{i}] Generated: {generated_text!r}...\n")

if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())

    # 提取并移除 DP、Profile 以及新增的特有参数
    dp_size = args.pop("data_parallel_size", 1)

    dp_num_nodes = args.pop("dp_num_nodes")
    dp_node_rank = args.pop("dp_node_rank")
    dp_master_addr = args.pop("dp_master_addr")
    dp_master_port = args.pop("dp_master_port")
    timeout = args.pop("timeout")
    global_batch_size = args.pop("global_batch_size")
    
    # 获取新增的数据集相关参数
    dataset_path = args.pop("dataset")
    # Don't pop seed, it's also used by engine args.
    seed = args["seed"]
    max_output_len = args.pop("max_output_len")

    profile = args.pop("profile")

    # Prepare dataset prompts.
    # 读取 ShareGPT 数据集
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # 提取所有有效的 prompt (即 conversations 数组中的第一个 value)
    all_prompts = [
        item["conversations"][0]["value"] 
        for item in dataset 
        if "conversations" in item and len(item["conversations"]) > 0
    ]

    if not all_prompts:
        raise ValueError(f"No valid prompts found in the dataset: {dataset_path}")

    # 使用固定的种子进行全局统一的采样
    random.seed(seed)
    if global_batch_size <= len(all_prompts):
        prompts = random.sample(all_prompts, global_batch_size)
    else:
        # 如果请求数超过了数据集大小，则有放回地随机采样
        prompts = random.choices(all_prompts, k=global_batch_size)

    # Z DEBUG
    # prompts = [
    #     "Please write a detailed essay about the future of Artificial Intelligence and its impact on human society.",
    #     "What is the future of Artificial Intelligence and its impact on human society? Answer:",
    # ] * (global_batch_size // 2)
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ] * 100

    engine_args = args

    if dp_num_nodes == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port_val = get_open_port() if dp_master_port == 0 else dp_master_port
    else:
        dp_master_ip = dp_master_addr
        dp_master_port_val = dp_master_port

    assert dp_size % dp_num_nodes == 0, "dp_size must be divisible by dp_num_nodes"
    dp_per_node = dp_size // dp_num_nodes

    if current_platform.is_rocm():
        from multiprocessing import set_start_method
        set_start_method("spawn", force=True)

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(dp_node_rank * dp_per_node, (dp_node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            args=(
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port_val,
                global_batch_size,
                prompts,
                max_output_len,
                profile,
                engine_args,
            ),
        )
        proc.start()
        procs.append(proc)
        
    exit_code = 0
    for proc in procs:
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            print(f"Killing unresponsive process {proc.pid}.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    if exit_code != 0 or not profile:
        exit(exit_code)
