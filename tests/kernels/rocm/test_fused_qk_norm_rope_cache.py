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
# Bit-exact summary table (human-readable equivalent of the parametrized
# test above; one PASS/FAIL per (config, workload) row covering both
# RoPE styles).
# ---------------------------------------------------------------------------
def _max_abs_diff_bytes(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count mismatched bytes between two tensors (cheap, host-side after
    a single equal() check)."""
    if torch.equal(a, b):
        return 0
    return int((a != b).sum().item())


def test_bit_exact_table_vs_aiter():
    """One-shot bit-exact verification across all (config, workload) combos,
    formatted as a readable summary table similar to the perf table.

    For each row we run BOTH RoPE styles (neox + gptj) on identical inputs
    and report PASS only if all 6 outputs (q_out / k_cache / v_cache for
    each RoPE style) are byte-identical to AITER's.

    Run with:
        pytest tests/kernels/rocm/test_fused_qk_norm_rope_cache.py::test_bit_exact_table_vs_aiter -s
    """
    pytest.importorskip("aiter")

    num_heads_q = 32
    num_heads_k = 4
    head_dim = 128
    block_size = 16
    eps = 1e-5

    bf16, fp16 = torch.bfloat16, torch.float16

    def cfg(layout, qkv, kv):
        return f"{layout} in={qkv} kv={kv}"

    # 8 (layout, qkv_dtype, kv_cache_dtype) configurations.
    configs = [
        (cfg("shuffle", "bf16", "bf16"), True,  "auto", bf16),
        (cfg("shuffle", "fp16", "fp16"), True,  "auto", fp16),
        (cfg("shuffle", "bf16", "fp8"),  True,  "fp8",  bf16),
        (cfg("shuffle", "fp16", "fp8"),  True,  "fp8",  fp16),
        (cfg("flash",   "bf16", "bf16"), False, "auto", bf16),
        (cfg("flash",   "fp16", "fp16"), False, "auto", fp16),
        (cfg("flash",   "bf16", "fp8"),  False, "fp8",  bf16),
        (cfg("flash",   "fp16", "fp8"),  False, "fp8",  fp16),
    ]

    workloads = [
        ("single_seq_aligned", [4096]),
        ("multi_seq_full",     [128, 256, 512]),
        ("multi_seq_tail",     [125, 251, 511]),
        ("decode_only",        [1, 1, 1, 1]),
    ]

    rope_styles = [("neox", True), ("gptj", False)]

    width = 130
    print("\n")
    print("=" * width)
    print("Bit-exact verification: vLLM op output vs AITER op output "
          "(matched layout + dtype, identical inputs).")
    print("Each row tests both RoPE styles (neox + gptj). 'mismatch bytes' "
          "counts diff bytes across q_out + k_cache + v_cache (sum of both styles).")
    print("=" * width)
    print(
        f"  {'config':<25s}  {'workload':<22s}  "
        f"{'q_out':>8s}  {'k_cache':>8s}  {'v_cache':>8s}  "
        f"{'mismatch bytes':>14s}  result"
    )
    print("-" * width)

    n_pass = 0
    n_total = 0
    for cfg_label, use_shuffle, kv_cache_dtype, dtype in configs:
        for case_name, seq_lens in workloads:
            inp = _make_inputs(
                seq_lens=seq_lens, num_heads_q=num_heads_q,
                num_heads_k=num_heads_k, head_dim=head_dim, dtype=dtype,
                block_size=block_size, use_shuffle=use_shuffle,
                kv_cache_dtype=kv_cache_dtype,
            )
            q_match = k_match = v_match = True
            mismatch_bytes = 0
            for _, is_neox in rope_styles:
                q_a, k_a, v_a = _call_aiter(
                    inp, num_heads_q, num_heads_k, head_dim, is_neox, eps,
                    use_shuffle, block_size,
                )
                q_b, k_b, v_b = _call_vllm(
                    inp, num_heads_q, num_heads_k, head_dim, is_neox, eps,
                    use_shuffle, block_size,
                )
                torch.cuda.synchronize()
                q_diff = _max_abs_diff_bytes(q_a, q_b)
                k_diff = _max_abs_diff_bytes(k_a, k_b)
                v_diff = _max_abs_diff_bytes(v_a, v_b)
                if q_diff: q_match = False
                if k_diff: k_match = False
                if v_diff: v_match = False
                mismatch_bytes += q_diff + k_diff + v_diff
            n_total += 1
            row_pass = q_match and k_match and v_match
            if row_pass:
                n_pass += 1
            print(
                f"  {cfg_label:<25s}  {case_name:<22s}  "
                f"{'match' if q_match else 'DIFF':>8s}  "
                f"{'match' if k_match else 'DIFF':>8s}  "
                f"{'match' if v_match else 'DIFF':>8s}  "
                f"{mismatch_bytes:>14d}  "
                f"{'PASS' if row_pass else 'FAIL'}"
            )

    print("-" * width)
    print(
        f"  Total: {n_pass}/{n_total} rows passed "
        f"({len(configs)} configs x {len(workloads)} workloads x "
        f"{len(rope_styles)} RoPE styles = "
        f"{len(configs) * len(workloads) * len(rope_styles)} sub-cases)"
    )
    print("=" * width)
    assert n_pass == n_total, (
        f"{n_total - n_pass}/{n_total} bit-exact rows failed; see table above"
    )


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


def _expected_vllm_path(seq_lens, use_shuffle, block_size):
    """Predict which path the vLLM op will take per group based on inputs.

    Returns one of: "fast" (every group fast-paths), "mixed" (some groups
    fast, some fallback), "fallback" (every group fallback).
    """
    if not use_shuffle:
        # FLASH layout has no fast/fallback distinction — always per-warp
        # vector store.
        return "flash"

    # Shuffle: fast path requires (slot_first % 8 == 0) AND (num_real == 8).
    # In our test setup each seq starts on a fresh block boundary, so
    # slot_first is always mod-block_size aligned (mod-8). The remaining
    # gate is whether the group is full.
    full_groups = sum(L // 8 for L in seq_lens)
    tail_groups = sum(1 for L in seq_lens if L % 8 != 0)
    short_groups = sum(1 for L in seq_lens if L < 8)
    total_groups = full_groups + max(tail_groups, short_groups)
    if total_groups == 0:
        return "n/a"
    if full_groups == total_groups:
        return "fast"
    if full_groups == 0:
        return "fallback"
    return "mixed"


def _time_scenario(
    workload_label, seq_lens, num_heads_q, num_heads_k, head_dim, dtype,
    vllm_kv_cache_dtype, aiter_kv_cache_dtype, block_size, is_neox,
    vllm_use_shuffle, aiter_use_shuffle, eps, iters, comparison_label,
):
    """Time AITER vs the vLLM op on a workload. Each side may have its own
    layout / kv_cache_dtype to support cross-layout comparisons (e.g. our
    shuffle vs AITER's flash)."""
    aiter_inp = _make_inputs(
        seq_lens=seq_lens, num_heads_q=num_heads_q,
        num_heads_k=num_heads_k, head_dim=head_dim, dtype=dtype,
        block_size=block_size, use_shuffle=aiter_use_shuffle,
        kv_cache_dtype=aiter_kv_cache_dtype,
    )
    vllm_inp = _make_inputs(
        seq_lens=seq_lens, num_heads_q=num_heads_q,
        num_heads_k=num_heads_k, head_dim=head_dim, dtype=dtype,
        block_size=block_size, use_shuffle=vllm_use_shuffle,
        kv_cache_dtype=vllm_kv_cache_dtype,
    )
    aiter_run = _make_aiter_runner(
        aiter_inp, num_heads_q, num_heads_k, head_dim, is_neox, eps,
        aiter_use_shuffle, block_size,
    )
    vllm_run = _make_vllm_runner(
        vllm_inp, num_heads_q, num_heads_k, head_dim, is_neox, eps,
        vllm_use_shuffle, block_size,
    )

    aiter_us = _time_op(aiter_run, iters=iters)
    vllm_us = _time_op(vllm_run, iters=iters)

    # Effective KV-write bandwidth: real-token K+V bytes / time. We count
    # against each side's own cache element size since the two ops may write
    # caches of different element widths (bf16 vs fp8) in a cross-dtype row.
    def gbs(us, inp):
        elem_bytes = inp["k_cache"].element_size()
        bytes_ = 2 * sum(seq_lens) * num_heads_k * head_dim * elem_bytes
        return bytes_ / (us * 1e-6) / 1e9 if us > 0 else 0.0

    speedup = aiter_us / vllm_us if vllm_us > 0 else 0.0
    path = _expected_vllm_path(seq_lens, vllm_use_shuffle, block_size)

    # comparison_label is a (vllm_cfg, aiter_cfg) tuple.
    vllm_cfg, aiter_cfg = comparison_label
    print(
        f"  {vllm_cfg:<25s}  {aiter_cfg:<25s}  {workload_label:<28s}  "
        f"{aiter_us:8.2f}   {vllm_us:8.2f}    {speedup:5.2f}x    "
        f"{gbs(aiter_us, aiter_inp):8.1f}    "
        f"{gbs(vllm_us, vllm_inp):8.1f}    {path}"
    )
    return aiter_us, vllm_us


def test_perf_qwen3_30b():
    """Microbench at Qwen3-30B-A3B shape (ISL=1000, OSL=1000, conc=16).

    Each row compares AITER vs the vLLM op on identical inputs and identical
    layout/dtype settings — i.e. AITER's shuffle path vs vLLM's shuffle path,
    AITER's flash path vs vLLM's flash path. The "vllm path" column shows
    which internal path the vLLM op ended up taking:

        fast      every 8-token group hits the dim-major LDS fast path
        mixed     full groups go fast, tail groups fall back
        fallback  every group falls back (decode, or len_i < 8)
        flash     FLASH layout — single per-warp vector-store, no branching

    Hypothesis to confirm:
      * SHUFFLE prefill: vLLM (dim-major LDS) >> AITER (scattered writes).
        bench-hbm predicts ~5x at kernel-write-only level; here we see the
        full-kernel speedup, bounded above by write-time fraction of kernel.
      * SHUFFLE decode: vLLM falls back; should match or beat AITER (single
        kernel launch vs AITER's single launch, no LDS staging overhead).
      * FLASH everywhere: both ops share the optimal vector-store path;
        speedup comes purely from launch-overhead savings + slightly better
        wave occupancy.
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
        ("prefill 1seq x 1000 tok",   [1000]),
        # Multi-seq prefill batch (e.g., 16 prefills concurrent in 1 step).
        ("prefill 16seqs x 1000 tok", [1000] * 16),
        # Decode step: 16 seqs x 1 token. Always fallback (bit-identical).
        ("decode  16seqs x 1 tok",    [1] * 16),
    ]

    # Comparison entry tuple:
    #   (group_header, vllm_cfg_label, vllm_use_shuffle, vllm_kv_cache_dtype,
    #                  aiter_cfg_label, aiter_use_shuffle, aiter_kv_cache_dtype,
    #    qkv_dtype)
    # Label convention: "<layout> in=<qkv_dtype> kv=<cache_dtype>"
    # For example:
    #   shuffle in=bf16 kv=bf16   shuffle layout, bf16 QKV input,  bf16 KV cache
    #   shuffle in=bf16 kv=fp8    shuffle layout, bf16 QKV input,  fp8 KV cache
    #                             (bf16 inputs are quantized to fp8 before caching)
    #   flash in=fp16 kv=fp16     flash layout, fp16 input + fp16 cache
    bf16, fp16 = torch.bfloat16, torch.float16

    def cfg(layout: str, qkv: str, kv: str) -> str:
        return f"{layout} in={qkv} kv={kv}"

    comparisons = [
        # ----- Same-layout: vLLM op vs AITER on the same layout -----
        ("Same-layout (vLLM beats AITER on its own turf)",
         cfg("shuffle", "bf16", "bf16"), True, "auto",
         cfg("shuffle", "bf16", "bf16"), True, "auto", bf16),
        (None,
         cfg("shuffle", "fp16", "fp16"), True, "auto",
         cfg("shuffle", "fp16", "fp16"), True, "auto", fp16),
        (None,
         cfg("shuffle", "bf16", "fp8"),  True, "fp8",
         cfg("shuffle", "bf16", "fp8"),  True, "fp8",  bf16),
        (None,
         cfg("shuffle", "fp16", "fp8"),  True, "fp8",
         cfg("shuffle", "fp16", "fp8"),  True, "fp8",  fp16),
        (None,
         cfg("flash",   "bf16", "bf16"), False, "auto",
         cfg("flash",   "bf16", "bf16"), False, "auto", bf16),
        (None,
         cfg("flash",   "fp16", "fp16"), False, "auto",
         cfg("flash",   "fp16", "fp16"), False, "auto", fp16),
        (None,
         cfg("flash",   "bf16", "fp8"),  False, "fp8",
         cfg("flash",   "bf16", "fp8"),  False, "fp8",  bf16),
        (None,
         cfg("flash",   "fp16", "fp8"),  False, "fp8",
         cfg("flash",   "fp16", "fp8"),  False, "fp8",  fp16),
        # ----- Cross-layout: vLLM SHUFFLE vs AITER FLASH (matched dtypes) -----
        # The killer question: does our SHUFFLE op match or beat AITER's
        # optimal FLASH path? If yes, the shuffle penalty is fully closed
        # at the kernel level — shuffle is no longer a bad layout choice
        # for prefill performance.
        ("Cross-layout (vLLM shuffle vs AITER's optimal flash, matched dtypes)",
         cfg("shuffle", "bf16", "bf16"), True, "auto",
         cfg("flash",   "bf16", "bf16"), False, "auto", bf16),
        (None,
         cfg("shuffle", "fp16", "fp16"), True, "auto",
         cfg("flash",   "fp16", "fp16"), False, "auto", fp16),
        (None,
         cfg("shuffle", "bf16", "fp8"),  True, "fp8",
         cfg("flash",   "bf16", "fp8"),  False, "fp8",  bf16),
        (None,
         cfg("shuffle", "fp16", "fp8"),  True, "fp8",
         cfg("flash",   "fp16", "fp8"),  False, "fp8",  fp16),
    ]

    width = 160
    print("\n")
    print("=" * width)
    print(
        f"Qwen3-30B-A3B microbench  -  num_heads_q={num_heads_q}  "
        f"num_heads_kv={num_heads_k}  head_dim={head_dim}  "
        f"block_size={block_size}  iters={iters}"
    )
    print("Each row pairs an AITER kernel run vs a vLLM kernel run on "
          "identical inputs. Times in microseconds.")
    print("Config format: <layout> in=<qkv dtype> kv=<cache dtype>. "
          "EffKV BW = real-token K+V bytes / time, GB/s (per side).")
    print("=" * width)
    print(
        f"  {'vLLM config':<25s}  {'AITER config':<25s}  {'workload':<28s}  "
        f"{'AITER':>8s}   {'vLLM':>8s}    speedup     "
        f"{'AITER':>8s}    {'vLLM':>8s}    vllm path"
    )
    print(
        f"  {'':<25s}  {'':<25s}  {'':<28s}  "
        f"{'(us)':>8s}   {'(us)':>8s}              "
        f"{'(GB/s)':>8s}    {'(GB/s)':>8s}"
    )
    print("-" * width)
    for entry in comparisons:
        (group_header, vllm_cfg, vllm_shuf, vllm_kvd,
         aiter_cfg, aiter_shuf, aiter_kvd, dtype) = entry
        if group_header is not None:
            print()
            print(f"  --- {group_header} ---")
        for workload_label, seq_lens in scenarios:
            _time_scenario(
                workload_label=workload_label, seq_lens=seq_lens,
                num_heads_q=num_heads_q, num_heads_k=num_heads_k,
                head_dim=head_dim, dtype=dtype,
                vllm_kv_cache_dtype=vllm_kvd,
                aiter_kv_cache_dtype=aiter_kvd,
                block_size=block_size, is_neox=is_neox,
                vllm_use_shuffle=vllm_shuf, aiter_use_shuffle=aiter_shuf,
                eps=eps, iters=iters,
                comparison_label=(vllm_cfg, aiter_cfg),
            )
    print("=" * width)
