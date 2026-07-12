# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression test for issue #48423.

MiniCPMV4_6ViTWindowAttentionSelfAttn.load_weights must correctly load
separate q_proj / k_proj / v_proj checkpoint weights into the fused
qkv_proj parameter.  A prior regression (PR #47058) replaced the manual
load_weights with WeightsMapper + AutoWeightsLoader, but the un-dotted
mapper keys cause a substring cascade:

    "q_proj" in "q_proj.weight" → key becomes "qkv_proj.weight"
    "v_proj" in "qkv_proj.weight" → key becomes "qkqkv_proj.weight"  (BUG)

This test verifies the mapping is correct.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_mock_tp_group():
    g = MagicMock()
    g.rank_in_group = 0
    g.world_size = 1
    return g


@pytest.mark.cpu_test
def test_minicpmv46_vit_self_attn_loads_separate_qkv(default_vllm_config):
    """load_weights must handle separate q/k/v projection weights without
    raising ValueError from a cascaded substring mapping."""
    mock_group = _make_mock_tp_group()

    patches = [
        patch(
            "vllm.model_executor.models.minicpmv4_6.is_vit_use_data_parallel",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.models.minicpmv4_6."
            "get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tp_group",
            return_value=mock_group,
        ),
    ]

    for p in patches:
        p.start()

    try:
        from vllm.model_executor.models.minicpmv4_6 import (
            MiniCPMV4_6ViTWindowAttentionSelfAttn,
        )

        config = SimpleNamespace(hidden_size=64, num_attention_heads=4)
        module = MiniCPMV4_6ViTWindowAttentionSelfAttn(config)

        embed_dim = 64
        weights = [
            ("q_proj.weight", torch.randn(embed_dim, embed_dim)),
            ("q_proj.bias", torch.randn(embed_dim)),
            ("k_proj.weight", torch.randn(embed_dim, embed_dim)),
            ("k_proj.bias", torch.randn(embed_dim)),
            ("v_proj.weight", torch.randn(embed_dim, embed_dim)),
            ("v_proj.bias", torch.randn(embed_dim)),
            ("out_proj.weight", torch.randn(embed_dim, embed_dim)),
            ("out_proj.bias", torch.randn(embed_dim)),
        ]

        # Before fix: raises ValueError about 'qkqkv_proj' not existing.
        # After fix: succeeds and returns the set of loaded weight names.
        loaded = module.load_weights(iter(weights))
        assert loaded is not None
        assert len(loaded) > 0
    finally:
        for p in reversed(patches):
            p.stop()
