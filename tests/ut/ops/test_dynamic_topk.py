from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe.experts_selector import _apply_token_top_ks
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEPrepareOutput
from vllm_ascend.sample.rejection_sampler import _build_output_token_top_ks


def test_apply_token_top_ks_masks_routes_per_layer():
    topk_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    topk_weights = torch.ones(2, 4)
    token_top_ks = torch.tensor([[2, 3], [1, 2]], dtype=torch.int32)

    with patch.dict("os.environ", {"VLLM_DYN_TOPKS_NO_DROP_TOKENS": "0"}):
        _apply_token_top_ks(
            topk_indices,
            topk_weights,
            layer_idx=1,
            num_experts=8,
            token_top_ks=token_top_ks,
        )

    assert topk_indices.tolist() == [[0, 1, 2, 8], [4, 5, 8, 8]]
    assert topk_weights.tolist() == [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]


def test_apply_token_top_ks_no_drop_keeps_indices():
    topk_indices = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    topk_weights = torch.ones(1, 3)

    with patch.dict("os.environ", {"VLLM_DYN_TOPKS_NO_DROP_TOKENS": "1"}):
        _apply_token_top_ks(
            topk_indices,
            topk_weights,
            layer_idx=None,
            num_experts=3,
            token_top_ks=torch.tensor([1]),
        )

    assert topk_indices.tolist() == [[0, 1, 2]]
    assert topk_weights.tolist() == [[1.0, 0.0, 0.0]]


def test_apply_token_top_ks_rejects_misaligned_shape():
    with pytest.raises(ValueError, match="shape must match"):
        _apply_token_top_ks(
            torch.zeros(2, 4, dtype=torch.int32),
            torch.ones(2, 4),
            layer_idx=None,
            num_experts=4,
            token_top_ks=torch.ones(3, dtype=torch.int32),
        )


def test_build_output_token_top_ks_aligns_variable_draft_lengths():
    draft_token_top_ks = torch.tensor([[2, 3], [4, 5], [6, 7]], dtype=torch.int32)

    output = _build_output_token_top_ks(
        draft_token_top_ks=draft_token_top_ks,
        num_draft_tokens=[2, 1],
        max_spec_len=2,
        num_moe_layers=2,
        base_top_k=8,
    )

    assert output.tolist() == [
        [[2, 3], [4, 5], [8, 8]],
        [[6, 7], [8, 8], [8, 8]],
    ]


def test_moe_prepare_output_preserves_pertoken_scale_position():
    hidden_states = torch.ones(2, 4)
    router_logits = torch.ones(2, 8)
    pertoken_scale = torch.ones(2)

    output = MoEPrepareOutput(
        hidden_states,
        router_logits,
        None,
        None,
        pertoken_scale,
    )

    assert output.pertoken_scale is pertoken_scale
    assert output.token_top_ks is None
