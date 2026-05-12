#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Benchmark driver for torch_bsa.blitz_sparse_attention.

Measures:
*  latency (usec) - NPU timer
*  correctness comparison with a reference PyTorch implementation
"""

import math
import itertools
import logging
from typing import Callable, List, Tuple
import torch
import torch_npu
import torch_bsa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Global Definitions
DEVICE = 'npu'
DTYPE = torch.bfloat16
INPUT_LAYOUT = "BNSD"  # [batch_size, num_heads, seq_len, head_dim]

# Parameter sweeps (adjust as needed)
B_VALS = [1]
H_VALS = [3]
S_VALS = [118_806]  # s_q = s_kv
D_VALS = [128]   # head dimension

N_REPEATS = 10
N_WARMUP = 2

# Kind of attention matrix pattern
# "dense" - dense attention 
# "lower_triangular"
# "band"
# "custom"
# 'sparse_block_all_same' is the same, but with all masks which are the same.
# "blocks_optimized" - blocks 
# 'blocks_optimized_batched' is the new optimized version written by us
ATTENTION_MATRIX = "blocks_optimized_batched"

# For (block / vertical / band) mask - fraction of attention elements that is retained for computation
SPARSITY_VALS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# For the block mask
BLOCK_SIZE_Q = 128
BLOCK_SIZE_KV = 512
BLOCK_MASK_SEED = 1234
USE_FRAME = True

# For the band mask
BAND_PRE_TOKENS = 8
BAND_POST_TOKENS = 2

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False
# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

RUN_REFERENCE = False  # True <--> enables accuracy compariston
TORCH_REFERENCE = False  # If False, will instead run the torch_npu reference

torch.set_printoptions(
    threshold=100_000_000,
    linewidth=400,
    edgeitems=3,
    precision=4,
    sci_mode=False
)


def ref_blitz_sparse_attention_launcher(
    torch_reference: bool, 
    pfa_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, torch.Tensor, str],
    force_dense_sm: bool
) -> torch.Tensor:
    """
    runs a reference prompt_flash attention, for correctness comparisons and for baseline time measurements.
    torch_reference = True - launch our custom pythonic model "ref_blitz_sparse_attention_fp32" 
                      False -> launch the torch_npu.npu_fusion_attention
    force_dense_sm - relevant only for torch_interface=False:
                      True - apply sparse mode 0 (dense) and don't use the provided atten_mask
                      False - apply sparse mode 1 (sparse token mask) and use the provided atten_mask
     
    """
    q, k, v, head_num, scale, atten_mask, input_layout = pfa_inputs
    if torch_reference: # use our custom pythonic model
        return ref_blitz_sparse_attention_fp32(q, k, v, scale, atten_mask=atten_mask)
    else:
        return torch_npu.npu_fusion_attention(q, k, v, head_num=head_num, input_layout=input_layout, 
                                              scale=scale, pre_tockens=0, next_tockens=0, 
                                              atten_mask=atten_mask,
                                              sparse_mode=0 if force_dense_sm else 1)[0]    


def generate_sparse_blocks_by_row(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    seed: int,
    use_frame: bool = False,
) -> list[list[int]]:
    """
    Returns sparse block columns per block-row.

    Output shape: [n_block_rows][num_kept_elems]
      - n_block_rows = ceil(s_q / block_size)
      - num_kept_elems = round(n_block_cols * (1 - sparsity)), clamped to [1, n_block_cols]
      - values in [0, n_block_cols)
    """

    if use_frame:
        return generate_sparse_blocks_by_row_with_frame(s_q, s_kv, block_size_q, block_size_kv, sparsity, seed)

    sparsity = max(0.0, min(1.0, float(sparsity)))

    n_block_rows = math.ceil(s_q / block_size_q)
    n_block_cols = math.ceil(s_kv / block_size_kv)
    if n_block_rows == 0 or n_block_cols == 0:
        return []

    num_kept_elems = int(round(n_block_cols * (1.0 - sparsity)))
    num_kept_elems = max(1, min(num_kept_elems, n_block_cols))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    rows: list[list[int]] = []
    for _ in range(n_block_rows):
        cols = torch.randperm(n_block_cols, generator=g)[:num_kept_elems].tolist()
        rows.append(cols)

    return rows


def generate_sparse_blocks_by_row_with_frame(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    seed: int,
    pad_value: int = -1,
) -> list[list[int]]:
    """
    Constraints (post-sparsity selection):
      - first 8 block-cols of each row are always included (if they exist)
      - first 29 block-rows contain all blocks
      - last row contains all blocks
      - each row always includes the last block-col
    Output:
      - shape [n_block_rows][n_block_cols]
      - each row: sorted selected cols, then pad_value to the end
    """
    sparsity = max(0.0, min(1.0, float(sparsity)))

    n_block_rows = math.ceil(s_q / block_size_q)
    n_block_cols = math.ceil(s_kv / block_size_kv)
    if n_block_rows == 0 or n_block_cols == 0:
        return []

    last_col = n_block_cols - 1

    # Base target count from sparsity.
    num_kept_elems = int(round(n_block_cols * (1.0 - sparsity)))
    num_kept_elems = max(1, min(num_kept_elems, n_block_cols))

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # Forced columns for EVERY non-fully-dense row:
    # first 8 cols (or fewer if n_block_cols < 8) + last col
    forced_prefix = set(range(min(8, n_block_cols)))
    forced_always = set(forced_prefix)
    forced_always.add(last_col)

    # Ensure "num_kept_elems" is at least large enough to contain the forced set (unless fully dense anyway)
    num_kept_elems = max(num_kept_elems, len(forced_always))

    rows: list[list[int]] = []
    for r in range(n_block_rows):
        fully_dense = (r < 29) or (r == n_block_rows - 1)

        if fully_dense:
            cols = list(range(n_block_cols))
        else:
            forced = set(forced_always)

            need = num_kept_elems - len(forced)
            if need <= 0:
                cols = sorted(forced)
            else:
                candidates = [c for c in range(n_block_cols) if c not in forced]
                # sample without replacement
                perm = torch.randperm(len(candidates), generator=g).tolist()
                extra = [candidates[i] for i in perm[:need]]
                cols = sorted(list(forced) + extra)

        cols = cols + [pad_value] * (n_block_cols - len(cols))
        rows.append(cols)

    return rows


def make_block_mask(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row: list[list[int]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, 1, s_q, s_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    n_block_rows = math.ceil(s_q / block_size_q)
    n_block_cols = math.ceil(s_kv / block_size_kv)

    mask = torch.ones((s_q, s_kv), dtype=torch.bool, device=device)

    rows_to_process = min(n_block_rows, len(sparse_blocks_by_row))
    for r in range(rows_to_process):
        row_start = r * block_size_q
        row_end = min(row_start + block_size_q, s_q)

        for c in sparse_blocks_by_row[r]:
            if 0 <= c < n_block_cols:
                col_start = c * block_size_kv
                col_end = min(col_start + block_size_kv, s_kv)
                mask[row_start:row_end, col_start:col_end] = False

    return mask.unsqueeze(0).unsqueeze(0)


def generate_sparse_blocks_by_row_per_head(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    num_heads: int,
    base_seed: int,
) -> list[list[list[int]]]:
    """
    Returns sparse blocks per head.

    Output shape:
      [num_heads][n_block_rows][num_kept_elems]
    """
    return [
        generate_sparse_blocks_by_row(
            s_q=s_q,
            s_kv=s_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparsity=sparsity,
            seed=base_seed + h,
            use_frame=USE_FRAME,
        )
        for h in range(num_heads)
    ]


def make_block_mask_per_head(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row_per_head: list[list[list[int]]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, num_heads, s_q, s_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    head_masks = [
        make_block_mask(
            s_q=s_q,
            s_kv=s_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparse_blocks_by_row=sparse_blocks_by_row,
            device=device,
        ).squeeze(0).squeeze(0)  # [s_q, s_kv]
        for sparse_blocks_by_row in sparse_blocks_by_row_per_head
    ]

    return torch.stack(head_masks, dim=0).unsqueeze(0)


def make_lower_triangular_mask(
    s_q: int,
    s_kv: int,
    device: str = "npu:0",
) -> torch.Tensor:
    """
    Create an *exact* token-level lower-triangular (causal) mask.

    Mask positions where key index > query index (future attention).
    Works best when s_q == s_kv; for s_q != s_kv this still produces the
    correct "no-looking-forward" relation over indices.

    Returns:
        atten_mask: Bool tensor of shape [1, 1, s_q, s_kv] (BNSD),
                    where True means "masked" (disallowed).
    """
    q = torch.arange(s_q, device=device)[:, None]   # [s_q, 1]
    k = torch.arange(s_kv, device=device)[None, :]  # [1, s_kv]
    mask_2d = k > q                                # [s_q, s_kv]
    return mask_2d.unsqueeze(0).unsqueeze(0)


def make_dense_mask(
    s_q: int,
    s_kv: int,
    device: str = "npu:0",
):
    mask_2d = torch.zeros(s_q, s_kv, dtype=torch.bool, device=device)
    return mask_2d.unsqueeze(0).unsqueeze(0)


def make_band_mask(
    s_q: int,
    s_kv: int,
    pre_tokens: int,
    post_tokens: int = 0,
    device: str = "npu:0",
) -> torch.Tensor:
    """
    Band (sliding-window) mask:
      query i can attend only to keys in [i-pre_tokens, i+post_tokens].

    Returns:
        atten_mask: Bool tensor of shape [1, 1, s_q, s_kv] (BNSD),
                    where True means "masked" (disallowed).
    """
    pre_tokens = int(pre_tokens)
    post_tokens = int(post_tokens)
    if pre_tokens < 0:
        raise ValueError(f"pre_tokens must be >= 0, got {pre_tokens}")
    if post_tokens < 0:
        raise ValueError(f"post_tokens must be >= 0, got {post_tokens}")

    q = torch.arange(s_q, device=device)[:, None]    # [s_q, 1]
    k = torch.arange(s_kv, device=device)[None, :]   # [1, s_kv]

    left_bound = q - pre_tokens
    right_bound = q + post_tokens

    # Mask anything outside the allowed window:
    mask_2d = (k < left_bound) | (k > right_bound)
    return mask_2d.unsqueeze(0).unsqueeze(0)


def make_custom_mask(
    s_q: int,
    s_kv: int,
    device: str = "npu:0",
) -> torch.Tensor:
    # single block mask
    mask_2d = torch.zeros(s_q, s_kv, device=device)
    mask_2d[:BLOCK_SIZE_Q, :BLOCK_SIZE_KV] = 1
    return mask_2d.unsqueeze(0).unsqueeze(0).bool()


def block_allclose_map(
    out: torch.Tensor,
    ref: torch.Tensor,
    block_h: int,
    block_w: int,
    rtol: float = 0.05,
    atol: float = 0.05,
    print_map: bool = False,
) -> torch.Tensor:
    """
    Compare two 4D tensors [dim0, dim1, dim2, dim3] in (block_h x block_w) blocks over the last two dims.
    Returns a boolean tensor of shape [dim0, dim1, nby, nbx] where each entry indicates whether the
    entire block is allclose. Optionally prints a 0/1 block matrix per (batch, head).

    - First two dims: batch and head (iterated and printed as headers)
    - Last two dims: "drawn" dimensions (split into blocks)

    Blocks at the edges can be smaller if dim2 or dim3 is not divisible by block sizes.
    """
    if out.shape != ref.shape:
        raise ValueError(f"Shape mismatch: out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    if out.ndim != 4:
        raise ValueError(f"Expected 4D tensors [dim0, dim1, dim2, dim3], got out.ndim={out.ndim}")

    if block_h <= 0 or block_w <= 0:
        raise ValueError("block_h and block_w must be positive integers")

    dim0, dim1, dim2, dim3 = out.shape
    nby = (dim2 + block_h - 1) // block_h
    nbx = (dim3 + block_w - 1) // block_w

    # Elementwise closeness map: [dim0, dim1, dim2, dim3] boolean
    close = torch.isclose(out, ref, rtol=rtol, atol=atol)

    # Blockwise result: [dim0, dim1, nby, nbx]
    block_ok = close.new_empty((dim0, dim1, nby, nbx), dtype=torch.bool)

    for b, h, by, bx in itertools.product(range(dim0), range(dim1), range(nby), range(nbx)):
        y0, x0 = by * block_h, bx * block_w
        y1, x1 = min(y0 + block_h, dim2), min(x0 + block_w, dim3)
        block_ok[b, h, by, bx] = close[b, h, y0:y1, x0:x1].all()

    if print_map:
        for b, h in itertools.product(range(dim0), range(dim1)):
            logger.info(f"batch={b}, head={h}")
            # Print as 0/1 grid
            grid = block_ok[b, h].to(dtype=torch.int32)
            for by in range(nby):
                row = " ".join(str(int(v)) for v in grid[by].tolist())
                logger.info(row)
            logger.info()  # blank line between heads

    return block_ok


def ref_blitz_sparse_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_value: float,
    atten_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reference implementation for correctness comparisons.
    Scaled dot-product attention in float32 throughout.

    Assumes:
    - input_layout == "BNSD"
    - atten_mask (if provided) is bool broadcastable
      to [batch_size, num_heads, s_q, s_kv], where True means "masked out".
    """

    # Convert to fp32 for computation
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    #    [batch_size, num_heads, s_q, head_dim] x [batch_size, num_heads, head_dim, s_kv]
    # -> [batch_size, num_heads, s_q, s_kv]
    attn_scores = torch.matmul(q_f, k_f.transpose(-1, -2))
    attn_scores = attn_scores * scale_value

    if atten_mask is not None:
        if atten_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(
                atten_mask, torch.finfo(attn_scores.dtype).min
            )
        elif atten_mask.dtype in (torch.int8, torch.uint8):
            attn_scores = attn_scores.masked_fill(
                atten_mask == 1, torch.finfo(attn_scores.dtype).min
            )

    # Softmax in fp32
    attn_probs = torch.softmax(attn_scores, dim=-1)

    #    [batch_size, num_heads, s_q, s_kv] x [batch_size, num_heads, s_kv, head_dim] 
    # -> [batch_size, num_heads, s_q, head_dim]
    out = torch.matmul(attn_probs, v_f)

    # Cast back to original dtype for comparison
    return out.to(dtype=q.dtype)


def gen_pfa_inputs(
    batch_size: int, num_heads: int, s_q: int, s_kv: int, head_dim: int,
    device: str = "npu:0", dtype: torch.dtype = DTYPE
):
    """
    Generate common inputs for blitz_sparse_attention or prompt_flash_attention.
    """
    q = torch.randn(batch_size, num_heads, s_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, s_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, s_kv, head_dim, dtype=dtype, device=device)

    # For now we assume all sequences are full-length
    actseqlen = [s_q] * batch_size
    actseqlenkv = [s_kv] * batch_size

    return q, k, v, actseqlen, actseqlenkv


def create_attention_mask(b, h, s_q, s_kv, d, sparsity, attention_matrix, device, emit_atten_mask: bool = True):
    """
    Creates the (token)-level attention mask and the sabi - necessary for the blocks_optimized_batched
    sparse patterns
    """
    scale = 1.0 / math.sqrt(float(d))
    pre_tok, post_tok = 2147483647, 0
    sabi = pfa_atten_mask = bsa_atten_mask = None
    sm = 0

    if attention_matrix == "sparse_block_all_same":
        if sparsity > 0:
            block_indices = generate_sparse_blocks_by_row(s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, sparsity, 
                                                          seed=BLOCK_MASK_SEED)
            pfa_atten_mask = bsa_atten_mask = make_block_mask(s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, block_indices, 
                                                          device=device)
            sm = 1
    elif attention_matrix == "lower_triangular":
        if sparsity > 0:
            pfa_atten_mask = make_lower_triangular_mask(s_q, s_kv, device=device)
            bsa_atten_mask = make_lower_triangular_mask(2048, 2048, device=device)
            sm = 2
    elif attention_matrix == "band":
        if sparsity > 0:
            pfa_atten_mask = make_band_mask(s_q, s_kv, pre_tokens=BAND_PRE_TOKENS, post_tokens=BAND_POST_TOKENS, 
                                        device=device)
            bsa_atten_mask = make_lower_triangular_mask(2048, 2048, device=device)
            sm, pre_tok, post_tok = 4, BAND_PRE_TOKENS, BAND_POST_TOKENS
    elif attention_matrix == "blocks_optimized":
        if emit_atten_mask and sparsity > 0:
            per_head_block_ids = generate_sparse_blocks_by_row_per_head(s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, 
                                                                            sparsity, h, BLOCK_MASK_SEED)
            pfa_atten_mask = make_block_mask_per_head(s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, per_head_block_ids, 
                                                      device)
            sabi = torch.tensor(per_head_block_ids, dtype=torch.uint16, device=device)
    elif attention_matrix == "blocks_optimized_batched":
        per_batch_head_block_indices = [
            generate_sparse_blocks_by_row_per_head(s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, sparsity, h, 
                                                   BLOCK_MASK_SEED + bidx)
            for bidx in range(b)
        ]
        sabi = torch.tensor(per_batch_head_block_indices, dtype=torch.uint16, device=device)
        if emit_atten_mask and sparsity > 0:
            pfa_atten_mask = torch.cat([make_block_mask_per_head(
                                     s_q, s_kv, BLOCK_SIZE_Q, BLOCK_SIZE_KV, bi, device=device
                                     ) for bi in per_batch_head_block_indices], dim=0)
    elif attention_matrix == "custom":
        if sparsity > 0:
            pfa_atten_mask = make_custom_mask(s_q, s_kv, device=device)
            sm = 1
        bsa_atten_mask = pfa_atten_mask
    else:
        if attention_matrix != "dense":
            raise ValueError(f"Attention matrix type {attention_matrix} is not implemented, for dense use 'dense'")
    
    ret = (pfa_atten_mask, bsa_atten_mask, sabi, sm, scale, pre_tok, post_tok)
    
    return ret


def _run_timed(kernel_fn: Callable, input_sets: List, n_warmup: int, n_repeat: int):
    """
    Utility function to run a kernel multiple times on the provided inputs 
    and report the average latency in microseconds. Warmup is appplied first (not timed).
    """
    for args in input_sets[:n_warmup]:
        kernel_fn(*args)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for args in input_sets[n_warmup:]:
        kernel_fn(*args)
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / n_repeat * 1000.0  # ms -> μs


def _check_correctness(our_fn: Callable, ref_fn: Callable, b: int, h: int, s_q: int, s_kv: int, d: int, device: str
    ) -> str:
    """Run both implementations on shared inputs and return 'yes'/'no'."""
    q, k, v, seq, seqkv = gen_pfa_inputs(b, h, s_q, s_kv, d, device=device, dtype=DTYPE)
    out_our = our_fn(q, k, v, seq, seqkv).cpu()
    out_ref = ref_fn(q, k, v, seq, seqkv).cpu()

    if PRINT_OUTPUTS:
        logger.info("OURS: ", out_our.shape)
        logger.info(out_our)
        logger.info("REF:  ", out_ref.shape)
        logger.info(out_ref)

    equal = torch.allclose(out_our, out_ref, rtol=0.02, atol=0.02)
    if not equal and PRINT_BLOCK_EQUALITY:
        block_allclose_map(out_our, out_ref,
                           block_h=PRINT_HEIGHT, block_w=PRINT_WIDTH,
                           rtol=0.02, atol=0.02, print_map=True)
    return "yes" if equal else "no"


def _fmt_or_na(value, width, spec=".2f"):
    """Format a number with given width/spec, or right-align 'N/A' if None."""
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:{width}{spec}}"


def _make_our_fn(sabi, h, scale, atten_mask, sparsity_params):
    sm, pre_tok, post_tok = sparsity_params
    
    def fn(q, k, v, seq, seqkv):
        return torch_bsa.blitz_sparse_attention(
            q, k, v,
            sabi=sabi, actual_seq_lengths=seq,
            actual_seq_lengths_kv=seqkv, num_heads=h, num_key_value_heads=h,
            input_layout=INPUT_LAYOUT, scale_value=scale,
            atten_mask=atten_mask, sparse_mode=sm,
            pre_tokens=pre_tok, next_tokens=post_tok,
        )
    return fn


def _make_ref_fn(h, scale, atten_mask, run_ref_sparsity_0):
    def fn(q, k, v, seq, seqkv):
        return ref_blitz_sparse_attention_launcher(
                torch_reference=TORCH_REFERENCE, 
                pfa_inputs=(q, k, v, h, scale, atten_mask, INPUT_LAYOUT),
                force_dense_sm=run_ref_sparsity_0,
            )
    return fn


def benchmark_blitz_sparse_attention():
    run_our = True          # blitz_sparse_attention
    run_ref = RUN_REFERENCE # PyTorch reference
    n_warmup = N_WARMUP
    n_repeat = N_REPEATS

    if not run_our and not run_ref:
        logger.info("Nothing to run, must set run_our=True or run_ref=True")
        return

    # Print table header
    logger.info("=" * 90)
    logger.info(f"  {DTYPE=}  {INPUT_LAYOUT=}  {ATTENTION_MATRIX=}")
    logger.info("=" * 90)
    logger.info(
        f"{'H':>3} {'B':>3} {'s_q':>6} {'s_kv':>6} {'D':>4} {'sparsity':>9} {'Outputs_equal':>15} "
        f"{'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18}"
    )
    logger.info("-" * 90)

    for b, h, s_kv, d, sparsity in itertools.product(B_VALS, H_VALS, S_VALS, D_VALS, SPARSITY_VALS):
        s_q = s_kv

        # Build attention mask and related parameters for this configuration
        pfa_atten_mask, bsa_atten_mask, sabi, sm, scale, pre_tok, post_tok = create_attention_mask(
            b, h, s_q, s_kv, d, sparsity, ATTENTION_MATRIX, device=DEVICE, emit_atten_mask=run_ref)
        if PRINT_MASK and pfa_atten_mask is not None:
            logger.info(pfa_atten_mask.int())
            logger.info(pfa_atten_mask.shape)

        # When sparsity=0, always run reference as a dense baseline for sanity check
        run_ref_sparsity_0 = sparsity == 0

        our_fn = _make_our_fn(sabi, h, scale, bsa_atten_mask, (sm, pre_tok, post_tok))
        ref_fn = _make_ref_fn(h, scale, pfa_atten_mask, run_ref_sparsity_0)

        # Correctness: compare our output vs reference on shared inputs
        are_equal_ref = "N/A"
        if run_our and run_ref or run_ref_sparsity_0:
            are_equal_ref = _check_correctness(our_fn, ref_fn, b, h, s_q, s_kv, d, device=DEVICE)

        # Timing: generate fresh input sets for each implementation
        inputs = [gen_pfa_inputs(b, h, s_q, s_kv, d, device=DEVICE, dtype=DTYPE)
                  for _ in range(n_warmup + n_repeat)]

        our_duration = None
        if run_our:
            our_duration = _run_timed(our_fn, inputs, n_warmup, n_repeat)

        ref_duration = None
        if run_ref or run_ref_sparsity_0:
            ref_duration = _run_timed(ref_fn, inputs, n_warmup, n_repeat)

        # Print one results row
        logger.info(
            f"{h:>3} {b:>3} {s_q:>6} {s_kv:>6} {d:>4} {sparsity:>9.2f} {are_equal_ref:>15} "
            f"{_fmt_or_na(ref_duration, 18)} {_fmt_or_na(our_duration, 18)}"
        )

    logger.info("=" * 90)


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    benchmark_blitz_sparse_attention()
