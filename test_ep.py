from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch_npu
from torch.multiprocessing import Process

LOGGER = logging.getLogger("moe_dispatch_cli")
LOGGER.setLevel(logging.INFO)


@dataclass(frozen=True)
class Config:
    quant_mode: int = 2
    input_dtype: str = "bfloat16"

    server_num: int = 1
    server_index: int = 0
    port: int = 50001
    master_ip: str = "127.0.0.1"
    dev_num: int = 8

    shared_expert_rank_num: int = 0
    moe_expert_num: int = 32
    batch_size: int = 8
    hidden_size: int = 7168
    topk: int = 8
    tp_world_size: int = 1

    zero_expert_num: int = 0
    copy_expert_num: int = 0

    expert_shard_type: int = 0
    warmup_global_bs: int = 16
    log_level: str = "INFO"

    @property
    def is_quant(self) -> bool:
        return self.quant_mode > 0

    @property
    def is_dispatch_scales(self) -> bool:
        return self.is_quant

    @property
    def dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = self.input_dtype.lower()
        if key not in mapping:
            raise ValueError(f"Unsupported input dtype: {self.input_dtype}")
        return mapping[key]

    @property
    def world_size(self) -> int:
        return self.server_num * self.dev_num

    @property
    def rank_per_dev(self) -> int:
        return self.world_size // self.server_num

    @property
    def ep_world_size(self) -> int:
        return self.world_size // self.tp_world_size

    @property
    def moe_rank_num(self) -> int:
        return self.ep_world_size - self.shared_expert_rank_num

    @property
    def local_moe_expert_num(self) -> int:
        return self.moe_expert_num // self.moe_rank_num

    @property
    def global_bs(self) -> int:
        return self.batch_size * self.ep_world_size

    def validate(self) -> None:
        if self.world_size % self.server_num != 0:
            raise ValueError("world_size must be divisible by server_num")
        if self.tp_world_size <= 0:
            raise ValueError("tp_world_size must be positive")
        if self.world_size % self.tp_world_size != 0:
            raise ValueError("world_size must be divisible by tp_world_size")
        if self.tp_world_size != 1 and self.local_moe_expert_num > 1:
            raise ValueError("Unsupported configuration: tp_world_size != 1 and local_moe_expert_num > 1")
        if self.shared_expert_rank_num > self.ep_world_size:
            raise ValueError("shared_expert_rank_num must not exceed ep_world_size")
        if self.shared_expert_rank_num > 0 and self.ep_world_size % self.shared_expert_rank_num != 0:
            raise ValueError("ep_world_size must be divisible by shared_expert_rank_num")
        if self.moe_rank_num <= 0:
            raise ValueError("moe_rank_num must be positive")
        if self.moe_expert_num % self.moe_rank_num != 0:
            raise ValueError("moe_expert_num must be divisible by moe_rank_num")


@dataclass(frozen=True)
class RuntimeContext:
    cfg: Config
    global_rank: int
    local_rank: int

    @property
    def ep_rank_id(self) -> int:
        return self.global_rank // self.cfg.tp_world_size

    @property
    def tp_rank_id(self) -> int:
        return self.global_rank % self.cfg.tp_world_size


@dataclass(frozen=True)
class CommHandles:
    ep_group: Any
    tp_group: Any
    ep_hcomm: Any
    tp_hcomm: Any


@dataclass(frozen=True)
class DispatchResult:
    expand_x: torch.Tensor
    dynamic_scales: Optional[torch.Tensor]
    assist_info_for_combine: Any
    expert_token_nums: torch.Tensor
    ep_recv_counts: torch.Tensor
    tp_recv_counts: torch.Tensor
    expand_scales: Optional[torch.Tensor]


@dataclass(frozen=True)
class Inputs:
    x: torch.Tensor
    expert_ids: torch.Tensor
    expert_scales: torch.Tensor
    scales: Optional[torch.Tensor]
    x_active_mask: torch.Tensor


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Run torch_npu MoE dispatch/combine with a clean CLI interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quant-mode", type=int, default=0)
    parser.add_argument("--input-dtype", type=str, default="bfloat16")
    parser.add_argument("--server-num", type=int, default=1)
    parser.add_argument("--server-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--master-ip", type=str, default="127.0.0.1")
    parser.add_argument("--dev-num", type=int, default=8)
    parser.add_argument("--shared-expert-rank-num", type=int, default=0)
    parser.add_argument("--moe-expert-num", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tp-world-size", type=int, default=1)
    parser.add_argument("--zero-expert-num", type=int, default=0)
    parser.add_argument("--copy-expert-num", type=int, default=0)
    parser.add_argument("--expert-shard-type", type=int, default=0)
    parser.add_argument("--warmup-global-bs", type=int, default=16)
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    cfg = Config(
        quant_mode=args.quant_mode,
        input_dtype=args.input_dtype,
        server_num=args.server_num,
        server_index=args.server_index,
        port=args.port,
        master_ip=args.master_ip,
        dev_num=args.dev_num,
        shared_expert_rank_num=args.shared_expert_rank_num,
        moe_expert_num=args.moe_expert_num,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        topk=args.topk,
        tp_world_size=args.tp_world_size,
        zero_expert_num=args.zero_expert_num,
        copy_expert_num=args.copy_expert_num,
        expert_shard_type=args.expert_shard_type,
        warmup_global_bs=args.warmup_global_bs,
        log_level=args.log_level,
    )
    cfg.validate()
    return cfg


def create_groups(global_rank: int, cfg: Config) -> tuple[Any, Any]:
    ep_group_for_rank = None
    tp_group_for_rank = None

    for i in range(cfg.tp_world_size):
        ep_ranks = [x * cfg.tp_world_size + i for x in range(cfg.ep_world_size)]
        group = dist.new_group(backend="hccl", ranks=ep_ranks)
        if global_rank in ep_ranks:
            ep_group_for_rank = group
            LOGGER.info("rank=%s ep_ranks=%s", global_rank, ep_ranks)

    for i in range(cfg.ep_world_size):
        tp_ranks = [x + cfg.tp_world_size * i for x in range(cfg.tp_world_size)]
        group = dist.new_group(backend="hccl", ranks=tp_ranks)
        if global_rank in tp_ranks:
            tp_group_for_rank = group
            LOGGER.info("rank=%s tp_ranks=%s", global_rank, tp_ranks)

    if ep_group_for_rank is None or tp_group_for_rank is None:
        raise RuntimeError(f"Failed to create communication groups for rank {global_rank}")
    return ep_group_for_rank, tp_group_for_rank


def get_hcomm_name(rank: int, comm_group: Any) -> Any:
    if torch.__version__ > "2.0.1":
        return comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return comm_group.get_hccl_comm_name(rank)


def init_distributed(local_rank: int, cfg: Config) -> RuntimeContext:
    torch_npu.npu.set_device(local_rank)
    global_rank = local_rank + cfg.dev_num * cfg.server_index
    dist.init_process_group(
        backend="hccl",
        rank=global_rank,
        world_size=cfg.world_size,
        init_method=f"tcp://{cfg.master_ip}:{cfg.port}",
    )
    return RuntimeContext(cfg=cfg, global_rank=global_rank, local_rank=local_rank)


def build_comm_handles(ctx: RuntimeContext) -> CommHandles:
    ep_group, tp_group = create_groups(ctx.global_rank, ctx.cfg)
    ep_hcomm = get_hcomm_name(ctx.global_rank, ep_group)
    tp_hcomm = get_hcomm_name(ctx.global_rank, tp_group)
    return CommHandles(ep_group=ep_group, tp_group=tp_group, ep_hcomm=ep_hcomm, tp_hcomm=tp_hcomm)


def create_inputs(cfg: Config) -> Inputs:
    x = torch.randn(cfg.batch_size, cfg.hidden_size, dtype=cfg.dtype).npu()
    expert_ids = torch.randint(
        0,
        cfg.moe_expert_num + cfg.zero_expert_num + cfg.copy_expert_num,
        (cfg.batch_size, cfg.topk),
        dtype=torch.int32,
    ).npu()
    expert_scales = torch.randn(cfg.batch_size, cfg.topk, dtype=torch.float32).npu()

    scales_shape = (1 + cfg.moe_expert_num, cfg.hidden_size) if cfg.shared_expert_rank_num else (cfg.moe_expert_num, cfg.hidden_size)
    scales = torch.randn(scales_shape, dtype=torch.float32).npu() if cfg.is_dispatch_scales else None

    x_active_mask = (expert_ids >= 0) & (expert_ids < cfg.moe_expert_num)
    # x_active_mask[:] = True

    return Inputs(
        x=x,
        expert_ids=expert_ids,
        expert_scales=expert_scales,
        scales=scales,
        x_active_mask=x_active_mask,
    )


def build_dispatch_kwargs(
    *,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    x_active_mask: Optional[torch.Tensor],
    scales: Optional[torch.Tensor],
    ctx: RuntimeContext,
    comm: CommHandles,
    global_bs: int,
) -> dict[str, Any]:
    cfg = ctx.cfg
    return {
        "x": x,
        "expert_ids": expert_ids,
        "x_active_mask": x_active_mask,
        "group_ep": comm.ep_hcomm,
        "group_tp": comm.tp_hcomm,
        "ep_rank_id": ctx.ep_rank_id,
        "tp_rank_id": ctx.tp_rank_id,
        "ep_world_size": cfg.ep_world_size,
        "tp_world_size": cfg.tp_world_size,
        "expert_shard_type": cfg.expert_shard_type,
        "shared_expert_num": 0,
        "shared_expert_rank_num": cfg.shared_expert_rank_num,
        "moe_expert_num": cfg.moe_expert_num,
        "scales": scales,
        "quant_mode": cfg.quant_mode,
        "global_bs": global_bs,
        "zero_expert_num": cfg.zero_expert_num,
        "copy_expert_num": cfg.copy_expert_num,
    }


def warmup_dispatch(ctx: RuntimeContext, comm: CommHandles) -> DispatchResult:
    cfg = ctx.cfg
    warmup_x = torch.empty((1, cfg.hidden_size), dtype=cfg.dtype).uniform_(-1024, 1024).to(cfg.dtype).npu()
    warmup_expert_ids = torch.arange(0, cfg.topk, dtype=torch.int32).unsqueeze(0).npu()

    outputs = torch_npu.npu_moe_distribute_dispatch_v2(
        **build_dispatch_kwargs(
            x=warmup_x,
            expert_ids=warmup_expert_ids,
            x_active_mask=None,
            scales=None,
            ctx=ctx,
            comm=comm,
            global_bs=cfg.warmup_global_bs,
        )
    )
    return DispatchResult(*outputs)


def dispatch(inputs: Inputs, ctx: RuntimeContext, comm: CommHandles) -> DispatchResult:
    outputs = torch_npu.npu_moe_distribute_dispatch_v2(
        **build_dispatch_kwargs(
            x=inputs.x,
            expert_ids=inputs.expert_ids,
            x_active_mask=inputs.x_active_mask,
            scales=inputs.scales,
            ctx=ctx,
            comm=comm,
            global_bs=ctx.cfg.global_bs,
        )
    )
    return DispatchResult(*outputs)


def combine(inputs: Inputs, dispatched: DispatchResult, ctx: RuntimeContext, comm: CommHandles) -> torch.Tensor:
    cfg = ctx.cfg
    expand_x = dispatched.expand_x.to(cfg.dtype) if cfg.is_quant else dispatched.expand_x

    return torch_npu.npu_moe_distribute_combine_v2(
        expand_x=expand_x,
        expert_ids=inputs.expert_ids,
        x_active_mask=inputs.x_active_mask,
        assist_info_for_combine=dispatched.assist_info_for_combine,
        ep_send_counts=dispatched.ep_recv_counts,
        tp_send_counts=dispatched.tp_recv_counts,
        expert_scales=inputs.expert_scales,
        group_ep=comm.ep_hcomm,
        group_tp=comm.tp_hcomm,
        ep_world_size=cfg.ep_world_size,
        tp_world_size=cfg.tp_world_size,
        ep_rank_id=ctx.ep_rank_id,
        tp_rank_id=ctx.tp_rank_id,
        expert_shard_type=cfg.expert_shard_type,
        shared_expert_rank_num=cfg.shared_expert_rank_num,
        moe_expert_num=cfg.moe_expert_num,
        global_bs=cfg.global_bs,
        ori_x=inputs.x,
        zero_expert_num=cfg.zero_expert_num,
        copy_expert_num=cfg.copy_expert_num,
    )


def worker(local_rank: int, cfg: Config) -> None:
    ctx = init_distributed(local_rank, cfg)
    comm = build_comm_handles(ctx)
    inputs = create_inputs(cfg)

    warmup_dispatch(ctx, comm)
    output = combine(inputs, dispatch(inputs, ctx, comm), ctx, comm)

    torch.npu.synchronize()
    LOGGER.info(
        "rank=%s ep_rank_id=%s tp_rank_id=%s output_shape=%s finished",
        ctx.global_rank,
        ctx.ep_rank_id,
        ctx.tp_rank_id,
        tuple(output.shape),
    )


def log_config(cfg: Config) -> None:
    LOGGER.info(
        "Config: batch_size=%s global_bs=%s hidden_size=%s topk=%s quant_mode=%s "
        "moe_expert_num=%s local_moe_expert_num=%s tp_world_size=%s ep_world_size=%s "
        "shared_expert_rank_num=%s",
        cfg.batch_size,
        cfg.global_bs,
        cfg.hidden_size,
        cfg.topk,
        cfg.quant_mode,
        cfg.moe_expert_num,
        cfg.local_moe_expert_num,
        cfg.tp_world_size,
        cfg.ep_world_size,
        cfg.shared_expert_rank_num,
    )


def main() -> None:
    cfg = parse_args()
    setup_logging(cfg.log_level)
    log_config(cfg)

    os.environ.setdefault("MASTER_ADDR", cfg.master_ip)
    os.environ.setdefault("MASTER_PORT", str(cfg.port))

    processes: list[Process] = []
    for local_rank in range(cfg.rank_per_dev):
        process = Process(target=worker, args=(local_rank, cfg))
        process.start()
        processes.append(process)

    exit_codes = []
    for process in processes:
        process.join()
        exit_codes.append(process.exitcode)

    failed = [code for code in exit_codes if code not in (0, None)]
    if failed:
        raise RuntimeError(f"One or more worker processes failed: {exit_codes}")

    LOGGER.info("Run completed successfully.")


if __name__ == "__main__":
    main()
