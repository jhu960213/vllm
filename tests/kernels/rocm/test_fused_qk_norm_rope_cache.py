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


def _resolve_cache_dtype(kv_cache_dtype: str, qkv_dtype: torch.dtype):
    """Map the vLLM kv_cache_dtype string to (storage_dtype, per_elem_bytes).

    For fp8, picks the platform-appropriate variant: `float8_e4m3fnuz` on
    CDNA3/MI300X, `float8_e4m3fn` (OCP) on CDNA4/MI350X.
    """
    if kv_cache_dtype in ("auto", ""):
        return qkv_dtype, qkv_dtype.itemsize
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return current_platform.fp8_dtype(), 1
    raise ValueError(f"unsupported kv_cache_dtype: {kv_cache_dtype}")


def _make_inputs(
    seq_lens: list[int],
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    use_shuffle: bool,
    kv_cache_dtype: str = "auto",
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
    cache_dtype, cache_elem_bytes = _resolve_cache_dtype(kv_cache_dtype, dtype)
    x = 16 // cache_elem_bytes  # bf16/fp16 -> 8, fp8 -> 16

    if use_shuffle:
        # SHUFFLE layout: K and V have different shapes.
        k_cache = torch.zeros(
            _shuffle_k_shape(num_blocks, num_heads_k, head_dim, block_size, x),
            dtype=cache_dtype,
            device=device,
        )
        v_cache = torch.zeros(
            _shuffle_v_shape(num_blocks, num_heads_k, head_dim, block_size, x),
            dtype=cache_dtype,
            device=device,
        )
    else:
        # FLASH layout: K and V same shape, packed as kv_cache[0]/kv_cache[1].
        kv_cache = torch.zeros(
            _flash_shape(num_blocks, num_heads_k, head_dim, block_size),
            dtype=cache_dtype,
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
        "kv_cache_dtype": kv_cache_dtype,
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
    # Use a non-trivial scale to actually exercise the per-tensor scale path.
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.5, dtype=torch.float32)

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
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.5, dtype=torch.float32)

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
        kv_cache_dtype=inp["kv_cache_dtype"],
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
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "case_name,seq_lens",
    [
        ("single_seq_aligned", [4096]),  # all groups fast-path
        ("multi_seq_full", [128, 256, 512]),  # multi-seq, all len % 8 == 0
        ("multi_seq_tail", [125, 251, 511]),  # multi-seq, partial tails
        ("decode_only", [1, 1, 1, 1]),  # always fallback
    ],
)
def test_bit_exact_vs_aiter(
    case_name, seq_lens, kv_cache_dtype, use_shuffle, is_neox, dtype
):
    """vLLM op output must equal AITER op output byte-for-byte across all
    (batch shape, kv_cache_dtype, layout, RoPE style, qkv dtype) combos."""
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
        kv_cache_dtype=kv_cache_dtype,
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

    label = (
        f"[{case_name} shuffle={use_shuffle} neox={is_neox} "
        f"kv={kv_cache_dtype} {dtype}]"
    )
    # Q: same RMS+RoPE math, must match exactly. Q output is in qkv's dtype
    # (not affected by kv_cache_dtype).
    assert torch.equal(q_a, q_b), f"{label} q_out mismatch"
    # K/V: includes the per-tensor quant for fp8. The fast path's dim-major
    # LDS staging and the fallback's per-warp scattered store must both
    # reproduce AITER's bytes exactly.
    assert torch.equal(k_a, k_b), f"{label} k_cache mismatch"
    assert torch.equal(v_a, v_b), f"{label} v_cache mismatch"


# ---------------------------------------------------------------------------
# Performance microbench
# ---------------------------------------------------------------------------
#
# Run with:
#   pytest tests/kernels/rocm/test_fused_qk_norm_rope_cache.py::test_perf_qwen3_30b -s
#
# Confirms the bench-hbm hypothesis: the dim-major LDS fast path delivers a
# meaningful HBM-write speedup over AITER's scattered writes on the SHUFFLE
# layout for the headline Qwen3-30B-A3B prefill workload, while staying
# bit-exact (and ~free) on decode.
#
# Workload: ISL=1000, OSL=1000, concurrency=16, Qwen3-30B-A3B head config:
#   num_heads_q = 32  num_heads_kv = 4  head_dim = 128  block_size = 16
#   bf16 inputs, bf16 KV cache (default Qwen3-30B-A3B configuration).
# The kernel is invoked once per attention layer per forward step. We time
# each invocation in isolation (warmup + many iterations + hipEvent timing).


def _time_op(fn, iters: int, warmup: int = 5) -> float:
    """Return mean elapsed time per call in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    return (elapsed_ms / iters) * 1000.0


def _make_aiter_runner(
    inp, num_heads_q, num_heads_k, head_dim, is_neox, eps, use_shuffle,
    block_size,
):
    """Build a closure that runs AITER's op into pre-allocated outputs."""
    from aiter.ops.fused_qk_norm_rope_cache_quant import (
        fused_qk_norm_rope_cache_pts_quant_shuffle,
    )

    qkv_orig = inp["qkv"]
    qkv = qkv_orig.clone()
    k_cache = torch.zeros_like(inp["k_cache"])
    v_cache = torch.zeros_like(inp["v_cache"])
    q_out = torch.empty(
        qkv.shape[0], num_heads_q, head_dim,
        dtype=qkv_orig.dtype, device=qkv.device,
    )
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.5, dtype=torch.float32)
    num_tokens = qkv.shape[0]

    def run():
        # Restore qkv each call because AITER overwrites in-place.
        qkv.copy_(qkv_orig)
        fused_qk_norm_rope_cache_pts_quant_shuffle(
            qkv, inp["q_weight"], inp["k_weight"], inp["cos_sin"],
            inp["positions"], num_tokens, num_heads_q, num_heads_k,
            num_heads_k, head_dim, is_neox, eps, q_out, k_cache, v_cache,
            inp["slot_mapping"], k_scale, v_scale, None, None, False,
            use_shuffle, block_size, inp["x"],
        )

    return run


def _make_vllm_runner(
    inp, num_heads_q, num_heads_k, head_dim, is_neox, eps, use_shuffle,
    block_size,
):
    """Build a closure that runs the new vLLM op into pre-allocated outputs."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backends.rocm_attn import _build_kv_block_to_seq

    qkv_orig = inp["qkv"]
    qkv = qkv_orig.clone()
    k_cache = torch.zeros_like(inp["k_cache"])
    v_cache = torch.zeros_like(inp["v_cache"])
    q_out = torch.empty(
        qkv.shape[0], num_heads_q, head_dim,
        dtype=qkv_orig.dtype, device=qkv.device,
    )
    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.5, dtype=torch.float32)
    block_to_seq, block_to_group_in_seq = _build_kv_block_to_seq(
        inp["query_start_loc"]
    )

    def run():
        qkv.copy_(qkv_orig)
        ops.fused_qk_norm_rope_cache(
            qkv=qkv, q_weight=inp["q_weight"], k_weight=inp["k_weight"],
            cos_sin_cache=inp["cos_sin"], positions=inp["positions"],
            num_heads_q=num_heads_q, num_heads_k=num_heads_k,
            num_heads_v=num_heads_k, head_dim=head_dim, is_neox=is_neox,
            eps=eps, q_out=q_out, k_cache=k_cache, v_cache=v_cache,
            slot_mapping=inp["slot_mapping"],
            query_start_loc=inp["query_start_loc"],
            block_to_seq=block_to_seq,
            block_to_group_in_seq=block_to_group_in_seq,
            per_tensor_k_scale=k_scale, per_tensor_v_scale=v_scale,
            kv_cache_dtype=inp["kv_cache_dtype"], k_out=None, v_out=None,
            return_kv=False, use_shuffle_layout=use_shuffle,
            block_size=block_size, x=inp["x"],
        )

    return run


def _time_scenario(
    label, seq_lens, num_heads_q, num_heads_k, head_dim, dtype,
    kv_cache_dtype, block_size, is_neox, use_shuffle, eps, iters,
):
    inp = _make_inputs(
        seq_lens=seq_lens, num_heads_q=num_heads_q,
        num_heads_k=num_heads_k, head_dim=head_dim, dtype=dtype,
        block_size=block_size, use_shuffle=use_shuffle,
        kv_cache_dtype=kv_cache_dtype,
    )
    aiter_run = _make_aiter_runner(
        inp, num_heads_q, num_heads_k, head_dim, is_neox, eps, use_shuffle,
        block_size,
    )
    vllm_run = _make_vllm_runner(
        inp, num_heads_q, num_heads_k, head_dim, is_neox, eps, use_shuffle,
        block_size,
    )

    aiter_us = _time_op(aiter_run, iters=iters)
    vllm_us = _time_op(vllm_run, iters=iters)

    # Effective KV write traffic: 2 (K+V) * num_tokens * num_heads_kv *
    # head_dim * cache_elem_bytes (real-token bytes only).
    cache_elem_bytes = inp["k_cache"].element_size()
    real_bytes = (
        2 * sum(seq_lens) * num_heads_k * head_dim * cache_elem_bytes
    )

    def gbs(us):
        return real_bytes / (us * 1e-6) / 1e9 if us > 0 else 0.0

    print(
        f"  {label:<40s} "
        f"aiter={aiter_us:8.2f}us  vllm={vllm_us:8.2f}us  "
        f"speedup={aiter_us / vllm_us:5.2f}x  "
        f"effBW(aiter)={gbs(aiter_us):7.1f}GB/s  "
        f"effBW(vllm)={gbs(vllm_us):7.1f}GB/s"
    )
    return aiter_us, vllm_us


def test_perf_qwen3_30b():
    """Microbench at Qwen3-30B-A3B shape (ISL=1000, OSL=1000, conc=16).

    Hypothesis to confirm:
      * SHUFFLE prefill: vLLM op (dim-major LDS fast path) is meaningfully
        faster than AITER (scattered writes). bench-hbm predicts ~5x at
        kernel-write-only level; here we see the full kernel speedup which
        is bounded above by the write-time fraction of the kernel.
      * SHUFFLE decode: vLLM op falls back to per-warp scattered writes
        (bit-identical to AITER), so should be neutral (within noise).
      * FLASH prefill/decode: both go through the same vector-store path
        already; vLLM op should be neutral too.
    """
    pytest.importorskip("aiter")

    # Qwen3-30B-A3B attention config.
    num_heads_q = 32
    num_heads_k = 4
    head_dim = 128
    block_size = 16
    eps = 1e-5
    is_neox = True
    iters = 200

    # ISL=1000 OSL=1000 concurrency=16 reduces to two kernel-input shapes:
    #   * Prefill step: chunk of N tokens (typically chunked at
    #     max_num_batched_tokens = 8192). For ISL=1000 conc=16, the full
    #     prefill fits in a single chunk: 16 seqs x 1000 tokens.
    #   * Decode step: 16 tokens (1 per active seq), repeated 1000 times.
    scenarios = [
        # Pure prefill, single seq, ISL=1000 — fully fast-pathable.
        ("prefill_1seq_1000tok",  [1000]),
        # Multi-seq prefill batch (e.g., 16 prefills concurrent in 1 step).
        ("prefill_16seqs_x_1000", [1000] * 16),
        # Decode step: 16 seqs x 1 token. Always fallback (bit-identical).
        ("decode_16seqs_x_1tok",  [1] * 16),
    ]

    layout_combos = [
        ("SHUFFLE",     True,  "auto", torch.bfloat16),
        ("SHUFFLE-fp8", True,  "fp8",  torch.bfloat16),
        ("FLASH",       False, "auto", torch.bfloat16),
    ]

    print("\n")
    print("=" * 110)
    print(
        f"Qwen3-30B-A3B: num_heads_q={num_heads_q} num_heads_kv={num_heads_k}"
        f" head_dim={head_dim} block_size={block_size}  iters={iters}"
    )
    print("=" * 110)
    for layout_name, use_shuffle, kv_cache_dtype, dtype in layout_combos:
        print(f"\n[{layout_name}]")
        for label, seq_lens in scenarios:
            _time_scenario(
                label=label, seq_lens=seq_lens, num_heads_q=num_heads_q,
                num_heads_k=num_heads_k, head_dim=head_dim, dtype=dtype,
                kv_cache_dtype=kv_cache_dtype, block_size=block_size,
                is_neox=is_neox, use_shuffle=use_shuffle, eps=eps, iters=iters,
            )
    print("=" * 110)
