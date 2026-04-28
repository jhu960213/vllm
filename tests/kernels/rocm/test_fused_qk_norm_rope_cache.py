# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact correctness test for `fused_qk_norm_rope_cache`.

Compares the vLLM-native ROCm op (with the dim-major LDS staging fast path on
the SHUFFLE write layout) against AITER's reference
``fused_qk_norm_rope_cache_pts_quant_shuffle`` across:

* Two layouts (FLASH, SHUFFLE)
* Two RoPE styles (neox, gptj-interleaved)
* Two dtypes (bf16, fp16)
* Four batch shapes:
  - (a) single-seq prefill aligned to 8 (every group fast-paths)
  - (b) multi-seq prefill (each seq fast-paths its full groups, falls back on
        its tail)
  - (c) seq-tail with len_i % 8 != 0
  - (d) decode-only batch (always fallback)

Both paths must produce **byte-identical** K and V cache contents — the
fallback inlines AITER's exact write, and the fast path only triggers on full
8-token aligned groups (per the bench's correctness contract).
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform


pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm only"
)


def _shuffle_k_shape(num_blocks, num_kv_heads, head_dim, block_size, x):
    return (num_blocks, num_kv_heads, head_dim // x, block_size, x)


def _shuffle_v_shape(num_blocks, num_kv_heads, head_dim, block_size, x):
    return (num_blocks, num_kv_heads, block_size // x, head_dim, x)


def _flash_shape(num_blocks, num_kv_heads, head_dim, block_size):
    return (2, num_blocks, block_size, num_kv_heads, head_dim)


def _make_inputs(
    seq_lens: list[int],
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    use_shuffle: bool,
    seed: int = 0xC0FFEE,
):
    """Construct qkv / cos_sin / positions / slot_mapping / kv_cache pair."""
    torch.manual_seed(seed)
    device = torch.device("cuda")

    num_tokens = sum(seq_lens)
    total_heads = num_heads_q + num_heads_k + num_heads_k  # K and V same head count
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)

    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)

    max_pos = 8192
    cos_sin = torch.randn(max_pos, head_dim, dtype=dtype, device=device)

    positions = torch.empty(num_tokens, dtype=torch.int64, device=device)
    slot_mapping = torch.empty(num_tokens, dtype=torch.int64, device=device)
    qsl_list = [0]
    cur_block = 0
    cur_token = 0
    for L in seq_lens:
        # Each seq starts on a fresh block (mod-block_size aligned slot 0).
        positions_seq = torch.arange(L, dtype=torch.int64, device=device)
        positions[cur_token : cur_token + L] = positions_seq
        slot_base = cur_block * block_size
        slot_mapping[cur_token : cur_token + L] = (
            torch.arange(L, dtype=torch.int64, device=device) + slot_base
        )
        # Reserve enough blocks for this seq so spillover (if any) is in-block.
        blocks_for_seq = (L + block_size - 1) // block_size
        cur_block += blocks_for_seq
        cur_token += L
        qsl_list.append(cur_token)

    query_start_loc = torch.tensor(qsl_list, dtype=torch.int32, device=device)

    # Allocate one KV cache buffer big enough for cur_block blocks.
    num_blocks = max(cur_block, 1)
    x = 16 // dtype.itemsize  # bf16/fp16 -> 8

    if use_shuffle:
        # SHUFFLE layout: K and V have different shapes.
        k_cache = torch.zeros(
            _shuffle_k_shape(num_blocks, num_heads_k, head_dim, block_size, x),
            dtype=dtype,
            device=device,
        )
        v_cache = torch.zeros(
            _shuffle_v_shape(num_blocks, num_heads_k, head_dim, block_size, x),
            dtype=dtype,
            device=device,
        )
    else:
        # FLASH layout: K and V same shape, packed as kv_cache[0]/kv_cache[1].
        kv_cache = torch.zeros(
            _flash_shape(num_blocks, num_heads_k, head_dim, block_size),
            dtype=dtype,
            device=device,
        )
        k_cache, v_cache = kv_cache.unbind(0)

    return {
        "qkv": qkv,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos_sin": cos_sin,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "query_start_loc": query_start_loc,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "num_blocks": num_blocks,
        "x": x,
    }


def _call_aiter(
    inp,
    num_heads_q,
    num_heads_k,
    head_dim,
    is_neox,
    eps,
    use_shuffle,
    block_size,
):
    """Run AITER's `fused_qk_norm_rope_cache_pts_quant_shuffle` on a fresh
    cache buffer (deep-copy of inp's k_cache / v_cache zeros)."""
    from aiter.ops.fused_qk_norm_rope_cache_quant import (
        fused_qk_norm_rope_cache_pts_quant_shuffle,
    )

    qkv = inp["qkv"].clone()
    k_cache = torch.zeros_like(inp["k_cache"])
    v_cache = torch.zeros_like(inp["v_cache"])

    q_out = torch.empty(
        qkv.shape[0],
        num_heads_q,
        head_dim,
        dtype=qkv.dtype,
        device=qkv.device,
    )
    k_scale = torch.tensor(1.0, dtype=torch.float32)
    v_scale = torch.tensor(1.0, dtype=torch.float32)

    fused_qk_norm_rope_cache_pts_quant_shuffle(
        qkv,
        inp["q_weight"],
        inp["k_weight"],
        inp["cos_sin"],
        inp["positions"],
        qkv.shape[0],
        num_heads_q,
        num_heads_k,
        num_heads_k,
        head_dim,
        is_neox,
        eps,
        q_out,
        k_cache,
        v_cache,
        inp["slot_mapping"],
        k_scale,
        v_scale,
        None,  # k_out
        None,  # v_out
        False,  # return_kv
        use_shuffle,
        block_size,
        inp["x"],
    )
    return q_out, k_cache, v_cache


def _call_vllm(
    inp,
    num_heads_q,
    num_heads_k,
    head_dim,
    is_neox,
    eps,
    use_shuffle,
    block_size,
):
    """Run the new vLLM-native ROCm op on a fresh cache buffer."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.rocm_attn import _build_kv_block_to_seq

    qkv = inp["qkv"].clone()
    k_cache = torch.zeros_like(inp["k_cache"])
    v_cache = torch.zeros_like(inp["v_cache"])

    q_out = torch.empty(
        qkv.shape[0],
        num_heads_q,
        head_dim,
        dtype=qkv.dtype,
        device=qkv.device,
    )
    k_scale = torch.tensor(1.0, dtype=torch.float32)
    v_scale = torch.tensor(1.0, dtype=torch.float32)

    block_to_seq, block_to_group_in_seq = _build_kv_block_to_seq(
        inp["query_start_loc"]
    )

    ops.fused_qk_norm_rope_cache(
        qkv=qkv,
        q_weight=inp["q_weight"],
        k_weight=inp["k_weight"],
        cos_sin_cache=inp["cos_sin"],
        positions=inp["positions"],
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_k,
        head_dim=head_dim,
        is_neox=is_neox,
        eps=eps,
        q_out=q_out,
        k_cache=k_cache,
        v_cache=v_cache,
        slot_mapping=inp["slot_mapping"],
        query_start_loc=inp["query_start_loc"],
        block_to_seq=block_to_seq,
        block_to_group_in_seq=block_to_group_in_seq,
        per_tensor_k_scale=k_scale,
        per_tensor_v_scale=v_scale,
        kv_cache_dtype="auto",
        k_out=None,
        v_out=None,
        return_kv=False,
        use_shuffle_layout=use_shuffle,
        block_size=block_size,
        x=inp["x"],
    )
    return q_out, k_cache, v_cache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("use_shuffle", [True, False])
@pytest.mark.parametrize(
    "case_name,seq_lens",
    [
        ("single_seq_aligned", [4096]),  # all groups fast-path
        ("multi_seq_full", [128, 256, 512]),  # multi-seq, all len % 8 == 0
        ("multi_seq_tail", [125, 251, 511]),  # multi-seq, partial tails
        ("decode_only", [1, 1, 1, 1]),  # always fallback
    ],
)
def test_bit_exact_vs_aiter(case_name, seq_lens, use_shuffle, is_neox, dtype):
    """vLLM op output must equal AITER op output byte-for-byte."""
    aiter = pytest.importorskip("aiter")  # noqa: F841

    num_heads_q = 32
    num_heads_k = 4
    head_dim = 128
    block_size = 16
    eps = 1e-5

    inp = _make_inputs(
        seq_lens=seq_lens,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        dtype=dtype,
        block_size=block_size,
        use_shuffle=use_shuffle,
    )

    q_a, k_a, v_a = _call_aiter(
        inp, num_heads_q, num_heads_k, head_dim, is_neox, eps, use_shuffle,
        block_size,
    )
    q_b, k_b, v_b = _call_vllm(
        inp, num_heads_q, num_heads_k, head_dim, is_neox, eps, use_shuffle,
        block_size,
    )
    torch.cuda.synchronize()

    # Q: same RMS+RoPE math, must match exactly.
    assert torch.equal(q_a, q_b), (
        f"[{case_name} shuffle={use_shuffle} neox={is_neox} {dtype}] "
        "q_out mismatch"
    )
    assert torch.equal(k_a, k_b), (
        f"[{case_name} shuffle={use_shuffle} neox={is_neox} {dtype}] "
        "k_cache mismatch"
    )
    assert torch.equal(v_a, v_b), (
        f"[{case_name} shuffle={use_shuffle} neox={is_neox} {dtype}] "
        "v_cache mismatch"
    )
