/*
 * Copyright (c) 2025, vLLM contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// vLLM-native ROCm port of AITER's fused_qk_norm_rope_cache_pts_quant_shuffle.
// Adds the dim-major LDS staging optimization on the SHUFFLE write path
// (validated in /app/bench-hbm: ~5x kernel-write speedup for full 8-token
// groups, AITER-equivalent cost everywhere else).
//
// SINGLE FUSED KERNEL with concatenated 1D grid: blockIdx.x < q_blocks_count
// runs the Q phase (1 warp per (token, q_head), packed AITER-style); the
// remaining blocks run the KV phase (8 warps per (8-token group, 1 kv_head),
// cooperative LDS staging fast path).
//
// Q phase: warp-per-(token, q_head), 8 packed per block → fully utilized
// even on tiny decode shapes.
// KV phase: per-block-per-(group, kv_head) ownership enables the dim-major
// LDS staging fast path on the SHUFFLE layout.
//   Fast path (dim-major LDS) when (slot_first % 8 == 0) AND (num_real == 8).
//   Fallback path (per-warp scalar shuffle) otherwise — bit-identical to
//   AITER's existing path.
//   Per-seq grid via int32 block_to_seq + block_to_group_in_seq lookup,
//   built host-side from query_start_loc.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>

#include "../attention/dtype_fp8.cuh"
#include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
#include "qk_norm_rope_cache.h"

namespace vllm {
namespace qk_norm_rope_cache {

// ===========================================================================
// File-local helpers
// ===========================================================================

template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T data[N];
  __device__ __forceinline__ T& operator[](int i) { return data[i]; }
  __device__ __forceinline__ T const& operator[](int i) const { return data[i]; }
  __device__ __forceinline__ void load(const T* p) {
    *this = *reinterpret_cast<const Vec<T, N>*>(p);
  }
  __device__ __forceinline__ void store(T* p) const {
    *reinterpret_cast<Vec<T, N>*>(p) = *this;
  }
  __device__ __forceinline__ void fill(T v) {
#pragma unroll
    for (int i = 0; i < N; ++i) data[i] = v;
  }
};

template <typename T>
__device__ __forceinline__ T warp_reduce_sum_(T val) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) val += __shfl_xor(val, o, 32);
  return val;
}

template <typename T>
__device__ __forceinline__ T warp_broadcast_(T val, int src) {
  return __shfl(val, src, 32);
}

// Warp-wide value shuffle for a Vec<T, N>. Backed by uint32_t shuffles.
template <typename T, int N>
__device__ __forceinline__ Vec<T, N> warp_shfl_vec_(const Vec<T, N>& v,
                                                    int src) {
  static_assert((sizeof(T) * N) % sizeof(uint32_t) == 0,
                "Vec must be u32-aligned for warp shuffle");
  constexpr int ITERS = (sizeof(T) * N) / sizeof(uint32_t);
  Vec<T, N> out;
  const uint32_t* in_u = reinterpret_cast<const uint32_t*>(&v);
  uint32_t* out_u = reinterpret_cast<uint32_t*>(&out);
#pragma unroll
  for (int i = 0; i < ITERS; ++i) {
    out_u[i] = __shfl(in_u[i], src, 32);
  }
  return out;
}

// In-place RMSNorm: x <- gamma * x / sqrt(mean(x^2) + eps).
// Reduction is over the warp (32 lanes covering one head_dim).
template <typename T, int N>
__device__ __forceinline__ void warp_rms_norm_(Vec<T, N>& x,
                                               const Vec<T, N>& gamma,
                                               float head_dim, float eps) {
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    float v = static_cast<float>(x[i]);
    acc += v * v;
  }
  acc = warp_reduce_sum_<float>(acc);
  acc = warp_broadcast_<float>(acc, 0);
  const float inv = rsqrtf(acc / head_dim + eps);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    x[i] = static_cast<T>(static_cast<float>(x[i]) * inv *
                          static_cast<float>(gamma[i]));
  }
}

// Apply RoPE to one warp's vec.
template <typename T, int N, bool IS_NEOX, int HEAD_SIZE>
__device__ __forceinline__ void warp_apply_rope_(
    Vec<T, N>& x, const T* cos_sin_for_token, int access_in_head,
    int neighbor_offset_lane, int lane_id) {
  constexpr int HALF = HEAD_SIZE / 2;
  if constexpr (IS_NEOX) {
    Vec<T, N> cs;
    cs.load(cos_sin_for_token + access_in_head);
    Vec<T, N> nb_cs = warp_shfl_vec_<T, N>(cs, lane_id + neighbor_offset_lane);
    Vec<T, N> nb_x = warp_shfl_vec_<T, N>(x, lane_id + neighbor_offset_lane);
    if (neighbor_offset_lane > 0) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        x[i] = static_cast<T>((float)x[i] * (float)cs[i] -
                              (float)nb_x[i] * (float)nb_cs[i]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        x[i] = static_cast<T>((float)x[i] * (float)nb_cs[i] +
                              (float)nb_x[i] * (float)cs[i]);
      }
    }
  } else {
    Vec<T, N> cv, sv;
    cv.load(cos_sin_for_token + access_in_head / 2);
    sv.load(cos_sin_for_token + access_in_head / 2 + HALF);
    Vec<T, N> y;
#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      y[2 * i + 0] = static_cast<T>((float)x[2 * i + 0] * (float)cv[i] -
                                    (float)x[2 * i + 1] * (float)sv[i]);
      y[2 * i + 1] = static_cast<T>((float)x[2 * i + 1] * (float)cv[i] +
                                    (float)x[2 * i + 0] * (float)sv[i]);
    }
    x = y;
  }
}

// Element-wise per-tensor quant: T (input) -> CACHE_T (cache).
// kAuto: identity copy (T == CACHE_T enforced by dispatch).
// kFp8E4M3: vllm::fp8::scaled_convert. For T == _Float16 we bit-cast to
// uint16_t first because vllm::fp8::scaled_vec_conversion's half->fp8
// specialization is keyed on the half-as-uint16 bit pattern (see
// csrc/quantization/w8a8/fp8/amd/quant_utils.cuh:487-495).
template <typename CACHE_T, typename T, vllm::Fp8KVCacheDataType KV_DTYPE>
__device__ __forceinline__ CACHE_T quant_one_(const T& x, float scale) {
  if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
    return x;
  } else if constexpr (std::is_same_v<T, _Float16>) {
    uint16_t bits;
    __builtin_memcpy(&bits, &x, sizeof(uint16_t));
    return vllm::fp8::scaled_convert<CACHE_T, uint16_t, KV_DTYPE>(bits, scale);
  } else {
    return vllm::fp8::scaled_convert<CACHE_T, T, KV_DTYPE>(x, scale);
  }
}

template <typename CACHE_T, typename T, int N,
          vllm::Fp8KVCacheDataType KV_DTYPE>
__device__ __forceinline__ Vec<CACHE_T, N> quant_vec_(const Vec<T, N>& src,
                                                      float scale) {
  Vec<CACHE_T, N> out;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    out[i] = quant_one_<CACHE_T, T, KV_DTYPE>(src[i], scale);
  }
  return out;
}

// Shuffle K layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
template <int HEAD_SIZE>
__device__ __forceinline__ int64_t shuffle_k_base_(int64_t slot_id,
                                                   int block_size,
                                                   int num_kv_heads,
                                                   int head_id, int access_id,
                                                   int x) {
  const int block_id = static_cast<int>(slot_id / block_size);
  const int block_offset = static_cast<int>(slot_id % block_size);
  const int k_head_stride = HEAD_SIZE * block_size;
  const int chunk_id = access_id / x;
  const int block_size_x = block_size * x;
  return static_cast<int64_t>(block_id) * num_kv_heads * k_head_stride +
         head_id * k_head_stride + chunk_id * block_size_x +
         block_offset * x + (access_id % x);
}

// Shuffle V layout: [num_blocks, num_kv_heads, block_size/x, head_size, x]
template <int HEAD_SIZE>
__device__ __forceinline__ int64_t shuffle_v_base_(int64_t slot_id,
                                                   int block_size,
                                                   int num_kv_heads,
                                                   int head_id, int x) {
  const int block_id = static_cast<int>(slot_id / block_size);
  const int block_offset = static_cast<int>(slot_id % block_size);
  const int v_head_stride = (block_size / x) * HEAD_SIZE * x;
  const int v_slot_chunk = block_offset / x;
  const int v_slot_in_chunk = block_offset % x;
  return static_cast<int64_t>(block_id) * num_kv_heads * v_head_stride +
         head_id * v_head_stride + v_slot_chunk * HEAD_SIZE * x +
         v_slot_in_chunk;
}

// ===========================================================================
// Single fused kernel: Q phase + KV phase, concatenated 1D grid.
//   blockIdx.x in [0, q_blocks_count)               => Q phase
//   blockIdx.x in [q_blocks_count, total_blocks)    => KV phase
// Each phase keeps its naturally-optimal per-block work distribution:
//   * Q phase  : 1 warp = 1 (token, q_head), 8 packed per block.
//   * KV phase : 1 block = 1 (8-token group, 1 kv_head), 8 cooperating warps.
// ===========================================================================
template <typename T, typename CACHE_T, int HEAD_SIZE, bool IS_NEOX,
          vllm::Fp8KVCacheDataType KV_DTYPE>
__launch_bounds__(256, 2) __global__ void fused_qk_norm_rope_cache_kernel(
    const T* __restrict__ qkv, const T* __restrict__ q_weight,
    const T* __restrict__ k_weight, const T* __restrict__ cos_sin_cache,
    const int64_t* __restrict__ positions, int64_t positions_stride,
    const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_to_seq,
    const int32_t* __restrict__ block_to_group_in_seq,
    const int32_t* __restrict__ query_start_loc, int num_heads_q,
    int num_heads_k, int num_heads_v, float eps, float per_tensor_k_scale,
    float per_tensor_v_scale, bool use_shuffle_layout, int block_size,
    int x_kv, T* __restrict__ q_out, CACHE_T* __restrict__ k_cache,
    CACHE_T* __restrict__ v_cache, T* __restrict__ k_out,
    T* __restrict__ v_out, int num_tokens, int q_blocks_count) {
  constexpr int WARP_SIZE_ = 32;
  constexpr int TOKENS_PER_GROUP = 8;
  constexpr int VEC_SIZE = HEAD_SIZE / WARP_SIZE_;
  constexpr int HALF = HEAD_SIZE / 2;
  static_assert(HEAD_SIZE == 64 || HEAD_SIZE == 128,
                "fused_qk_norm_rope_cache_kernel: HEAD_SIZE must be 64 or 128");

  const int warp = threadIdx.x / WARP_SIZE_;
  const int lane = threadIdx.x % WARP_SIZE_;
  const int access = lane * VEC_SIZE;
  const int neighbor = (access < HALF ? HALF : -HALF) / VEC_SIZE;
  const int total_heads = num_heads_q + num_heads_k + num_heads_v;

  // -----------------------------------------------------------------------
  // Q phase: 1 warp = 1 (token, q_head), 8 packed per block.
  // -----------------------------------------------------------------------
  if (blockIdx.x < q_blocks_count) {
    const int wpb = blockDim.x / WARP_SIZE_;
    const int gw = blockIdx.x * wpb + warp;
    const int total_q = num_tokens * num_heads_q;
    if (gw >= total_q) return;

    const int token = gw / num_heads_q;
    const int head = gw % num_heads_q;
    const T* qkv_ptr =
        qkv + (static_cast<int64_t>(token) * total_heads + head) * HEAD_SIZE;

    Vec<T, VEC_SIZE> w_vec, x_vec;
    w_vec.load(q_weight + access);
    x_vec.load(qkv_ptr + access);

    warp_rms_norm_<T, VEC_SIZE>(x_vec, w_vec, static_cast<float>(HEAD_SIZE),
                                eps);

    const int64_t pos = positions[token * positions_stride];
    const T* cs_ptr = cos_sin_cache + pos * HEAD_SIZE;
    warp_apply_rope_<T, VEC_SIZE, IS_NEOX, HEAD_SIZE>(x_vec, cs_ptr, access,
                                                      neighbor, lane);

    T* q_ptr = q_out + (static_cast<int64_t>(token) * num_heads_q + head) *
                           HEAD_SIZE;
    x_vec.store(q_ptr + access);
    return;
  }

  // -----------------------------------------------------------------------
  // KV phase: 8 warps per (8-token group, 1 kv_head). LDS staging fast path
  // when (slot_first % 8 == 0) AND (num_real == 8); otherwise per-warp
  // scalar-shuffle fallback (bit-identical to AITER).
  // -----------------------------------------------------------------------
  // x = 16 / sizeof(CACHE_T): bf16/fp16 -> 8, fp8 -> 16. Mailbox is 16 B.
  constexpr int X_KV = 16 / sizeof(CACHE_T);
  constexpr int NUM_CHUNKS = HEAD_SIZE / X_KV;
  constexpr int K_COMMIT_THREADS = NUM_CHUNKS * TOKENS_PER_GROUP;
  constexpr int V_COMMIT_THREADS = HEAD_SIZE;
  // V mailbox bytes: TOKENS_PER_GROUP * sizeof(CACHE_T) (16 for bf16, 8 for fp8).
  constexpr int V_MAILBOX_BYTES = TOKENS_PER_GROUP * sizeof(CACHE_T);
  // Dim-major V LDS row width (+1 pad to break 2-way bank conflict).
  constexpr int LDS_V_ROW = TOKENS_PER_GROUP + 1;

  // Decode flat KV block index to (group_id_global, kv_head). Layout:
  // kv_flat = group_id_global * num_heads_k + head_idx, so head_idx varies
  // fastest and consecutive blocks share the same group_id_global → all
  // kv_heads of a group probe the same slot_mapping[group_first_token],
  // friendly for the slot probe load.
  const int kv_flat = blockIdx.x - q_blocks_count;
  const int group_id_global = kv_flat / num_heads_k;
  const int head_idx = kv_flat - group_id_global * num_heads_k;

  const int seq_idx = block_to_seq[group_id_global];
  const int group_idx_in_seq = block_to_group_in_seq[group_id_global];

  const int seq_token_offset = query_start_loc[seq_idx];
  const int seq_query_len = query_start_loc[seq_idx + 1] - seq_token_offset;
  const int group_first_token =
      seq_token_offset + group_idx_in_seq * TOKENS_PER_GROUP;
  const int num_real_in_group =
      min(TOKENS_PER_GROUP, seq_query_len - group_idx_in_seq * TOKENS_PER_GROUP);

  const int k_head_in_qkv = num_heads_q + head_idx;
  const int v_head_in_qkv = num_heads_q + num_heads_k + head_idx;

  // Phase 1: probe alignment in thread 0; broadcast via shared.
  __shared__ int s_use_fast;
  __shared__ int64_t s_slot_first;
  if (threadIdx.x == 0) {
    int64_t slot_first = slot_mapping[group_first_token];
    s_slot_first = slot_first;
    bool aligned = (slot_first >= 0) && (slot_first % TOKENS_PER_GROUP == 0);
    bool full = (num_real_in_group == TOKENS_PER_GROUP);
    s_use_fast = (use_shuffle_layout && aligned && full) ? 1 : 0;
  }
  __syncthreads();
  const bool use_fast = s_use_fast != 0;
  const int64_t slot_first_val = s_slot_first;

  // Phase 2A: per-warp compute. Real warps load qkv, compute K/V into CACHE_T
  // register vectors. Dummy warps zero-fill (no qkv read => no OOB).
  const bool warp_is_real = (warp < num_real_in_group);
  const int my_token = group_first_token + warp;

  Vec<CACHE_T, VEC_SIZE> out_k_kv, out_v_kv;
  out_k_kv.fill(static_cast<CACHE_T>(0));
  out_v_kv.fill(static_cast<CACHE_T>(0));
  Vec<T, VEC_SIZE> k_vec_T;
  Vec<T, VEC_SIZE> v_vec_T;
  k_vec_T.fill(static_cast<T>(0));
  v_vec_T.fill(static_cast<T>(0));

  if (warp_is_real) {
    const T* qkv_k_ptr = qkv + (static_cast<int64_t>(my_token) * total_heads +
                                k_head_in_qkv) *
                                   HEAD_SIZE;
    const T* qkv_v_ptr = qkv + (static_cast<int64_t>(my_token) * total_heads +
                                v_head_in_qkv) *
                                   HEAD_SIZE;

    Vec<T, VEC_SIZE> w_vec;
    w_vec.load(k_weight + access);
    k_vec_T.load(qkv_k_ptr + access);
    warp_rms_norm_<T, VEC_SIZE>(k_vec_T, w_vec, static_cast<float>(HEAD_SIZE),
                                eps);
    const int64_t pos = positions[my_token * positions_stride];
    const T* cs_ptr = cos_sin_cache + pos * HEAD_SIZE;
    warp_apply_rope_<T, VEC_SIZE, IS_NEOX, HEAD_SIZE>(k_vec_T, cs_ptr, access,
                                                      neighbor, lane);
    out_k_kv = quant_vec_<CACHE_T, T, VEC_SIZE, KV_DTYPE>(k_vec_T,
                                                          per_tensor_k_scale);

    v_vec_T.load(qkv_v_ptr + access);
    out_v_kv = quant_vec_<CACHE_T, T, VEC_SIZE, KV_DTYPE>(v_vec_T,
                                                          per_tensor_v_scale);

    // Optional flat-layout outputs for return_kv path.
    if (k_out != nullptr) {
      T* dst = k_out +
               (static_cast<int64_t>(my_token) * num_heads_k + head_idx) *
                   HEAD_SIZE;
      k_vec_T.store(dst + access);
    }
    if (v_out != nullptr) {
      T* dst = v_out +
               (static_cast<int64_t>(my_token) * num_heads_v + head_idx) *
                   HEAD_SIZE;
      v_vec_T.store(dst + access);
    }
  }

  if (use_fast) {
    // Stage K flash-major, V dim-major to LDS.
    __shared__ CACHE_T lds_k[TOKENS_PER_GROUP][HEAD_SIZE];
    __shared__ CACHE_T lds_v[HEAD_SIZE][LDS_V_ROW];

#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      lds_k[warp][access + i] = out_k_kv[i];
      lds_v[access + i][warp] = out_v_kv[i];
    }
    __syncthreads();

    // Phase 2B: K + V commit (256 threads total, some idle for fp8).
    if (threadIdx.x < K_COMMIT_THREADS) {
      const int chunk_id = threadIdx.x / TOKENS_PER_GROUP;
      const int t_in_g = threadIdx.x % TOKENS_PER_GROUP;
      const int64_t slot_t = slot_first_val + t_in_g;
      const int64_t k_dst = shuffle_k_base_<HEAD_SIZE>(
          slot_t, block_size, num_heads_k, head_idx,
          /*access_id=*/chunk_id * X_KV, x_kv);
      uint4 mailbox =
          *reinterpret_cast<const uint4*>(&lds_k[t_in_g][chunk_id * X_KV]);
      *reinterpret_cast<uint4*>(&k_cache[k_dst]) = mailbox;
    } else if (threadIdx.x < K_COMMIT_THREADS + V_COMMIT_THREADS) {
      const int d = threadIdx.x - K_COMMIT_THREADS;
      const int64_t v_dst_base = shuffle_v_base_<HEAD_SIZE>(
          slot_first_val, block_size, num_heads_v, head_idx, x_kv);
      const int64_t v_dst = v_dst_base + d * x_kv;
      if constexpr (V_MAILBOX_BYTES == 16) {
        uint4 mailbox = *reinterpret_cast<const uint4*>(&lds_v[d][0]);
        *reinterpret_cast<uint4*>(&v_cache[v_dst]) = mailbox;
      } else if constexpr (V_MAILBOX_BYTES == 8) {
        uint2 mailbox = *reinterpret_cast<const uint2*>(&lds_v[d][0]);
        *reinterpret_cast<uint2*>(&v_cache[v_dst]) = mailbox;
      } else {
#pragma unroll
        for (int t = 0; t < TOKENS_PER_GROUP; ++t) {
          v_cache[v_dst + t] = lds_v[d][t];
        }
      }
    }
    return;
  }

  // Phase 2B (fallback): per-warp commit, real warps only, slot_id<0 skip.
  // Bit-identical to AITER's K vector store + V scalar loop.
  if (!warp_is_real) return;

  const int64_t slot_id = slot_mapping[my_token];
  if (slot_id < 0) return;

  if (use_shuffle_layout) {
    const int64_t k_dst = shuffle_k_base_<HEAD_SIZE>(
        slot_id, block_size, num_heads_k, head_idx, /*access_id=*/access, x_kv);
    out_k_kv.store(&k_cache[k_dst]);
    const int64_t v_dst_base = shuffle_v_base_<HEAD_SIZE>(
        slot_id, block_size, num_heads_v, head_idx, x_kv);
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      const int offset_in_head = access + i;
      v_cache[v_dst_base + offset_in_head * x_kv] = out_v_kv[i];
    }
  } else {
    const int64_t k_off =
        (slot_id * num_heads_k + head_idx) * HEAD_SIZE + access;
    out_k_kv.store(&k_cache[k_off]);
    const int64_t v_off =
        (slot_id * num_heads_v + head_idx) * HEAD_SIZE + access;
    out_v_kv.store(&v_cache[v_off]);
  }
}

// ===========================================================================
// Typed launcher
// ===========================================================================
template <typename T, typename CACHE_T, vllm::Fp8KVCacheDataType KV_DTYPE>
void launch_typed(
    const T* qkv, const T* q_weight, const T* k_weight, const T* cos_sin_cache,
    const int64_t* positions, int64_t positions_stride,
    const int64_t* slot_mapping, const int32_t* block_to_seq,
    const int32_t* block_to_group_in_seq, const int32_t* query_start_loc,
    int num_tokens, int num_heads_q, int num_heads_k, int num_heads_v,
    int head_dim, bool is_neox, float eps, float per_tensor_k_scale,
    float per_tensor_v_scale, bool use_shuffle_layout, int block_size, int x,
    int total_kv_blocks, T* q_out, CACHE_T* k_cache, CACHE_T* v_cache,
    T* k_out, T* v_out, hipStream_t stream) {
  constexpr int kBlock = 256;
  constexpr int kWarpsPerBlock = kBlock / 32;

  const int total_q_warps = num_tokens * num_heads_q;
  const int q_blocks =
      (total_q_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int kv_blocks =
      (num_heads_k > 0) ? (total_kv_blocks * num_heads_k) : 0;
  const int total_blocks = q_blocks + kv_blocks;
  if (total_blocks <= 0) return;

  dim3 grid(total_blocks, 1, 1);

#define LAUNCH(HS)                                                            \
  do {                                                                        \
    if (is_neox) {                                                            \
      fused_qk_norm_rope_cache_kernel<T, CACHE_T, HS, true, KV_DTYPE>         \
          <<<grid, kBlock, 0, stream>>>(                                      \
              qkv, q_weight, k_weight, cos_sin_cache, positions,              \
              positions_stride, slot_mapping, block_to_seq,                   \
              block_to_group_in_seq, query_start_loc, num_heads_q,            \
              num_heads_k, num_heads_v, eps, per_tensor_k_scale,              \
              per_tensor_v_scale, use_shuffle_layout, block_size, x, q_out,   \
              k_cache, v_cache, k_out, v_out, num_tokens, q_blocks);          \
    } else {                                                                  \
      fused_qk_norm_rope_cache_kernel<T, CACHE_T, HS, false, KV_DTYPE>        \
          <<<grid, kBlock, 0, stream>>>(                                      \
              qkv, q_weight, k_weight, cos_sin_cache, positions,              \
              positions_stride, slot_mapping, block_to_seq,                   \
              block_to_group_in_seq, query_start_loc, num_heads_q,            \
              num_heads_k, num_heads_v, eps, per_tensor_k_scale,              \
              per_tensor_v_scale, use_shuffle_layout, block_size, x, q_out,   \
              k_cache, v_cache, k_out, v_out, num_tokens, q_blocks);          \
    }                                                                         \
  } while (0)

  if (head_dim == 64) {
    LAUNCH(64);
  } else if (head_dim == 128) {
    LAUNCH(128);
  } else {
    TORCH_CHECK(false, "fused_qk_norm_rope_cache: unsupported head_dim ",
                head_dim);
  }
#undef LAUNCH
}

}  // namespace qk_norm_rope_cache
}  // namespace vllm

// ===========================================================================
// Public C++ entry point
// ===========================================================================
void fused_qk_norm_rope_cache(
    torch::Tensor& qkv, torch::Tensor& q_weight, torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache, torch::Tensor& positions, int64_t num_heads_q,
    int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, bool is_neox,
    double eps, torch::Tensor& q_out, torch::Tensor& k_cache,
    torch::Tensor& v_cache, torch::Tensor& slot_mapping,
    torch::Tensor& query_start_loc, torch::Tensor& block_to_seq,
    torch::Tensor& block_to_group_in_seq, torch::Tensor& per_tensor_k_scale,
    torch::Tensor& per_tensor_v_scale, const std::string& kv_cache_dtype,
    const std::optional<torch::Tensor>& k_out,
    const std::optional<torch::Tensor>& v_out, bool return_kv,
    bool use_shuffle_layout, int64_t block_size, int64_t x) {
  TORCH_CHECK(qkv.is_cuda(), "qkv must be on CUDA/ROCm device");
  TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Long,
              "positions must be int64");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int64");
  TORCH_CHECK(block_to_seq.scalar_type() == at::ScalarType::Int,
              "block_to_seq must be int32");
  TORCH_CHECK(block_to_group_in_seq.scalar_type() == at::ScalarType::Int,
              "block_to_group_in_seq must be int32");
  TORCH_CHECK(query_start_loc.scalar_type() == at::ScalarType::Int,
              "query_start_loc must be int32");
  TORCH_CHECK(block_to_seq.is_contiguous() &&
                  block_to_group_in_seq.is_contiguous() &&
                  query_start_loc.is_contiguous(),
              "block mapping tensors must be contiguous");
  TORCH_CHECK(head_dim == 64 || head_dim == 128,
              "fused_qk_norm_rope_cache supports head_dim 64 or 128, got ",
              head_dim);
  TORCH_CHECK(per_tensor_k_scale.numel() == 1 &&
                  per_tensor_v_scale.numel() == 1,
              "scales must be scalar tensors");

  const int num_tokens = static_cast<int>(qkv.size(0));
  const int total_kv_blocks = static_cast<int>(block_to_seq.size(0));

  const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
  hipStream_t stream = at::cuda::getCurrentCUDAStream();

  const int64_t pos_stride = positions.stride(0);
  const float k_scale_f = per_tensor_k_scale.item<float>();
  const float v_scale_f = per_tensor_v_scale.item<float>();

#define DISPATCH_BODY(KV_T, CACHE_T_TYPE, KV_DTYPE_VAL)                       \
  vllm::qk_norm_rope_cache::launch_typed<KV_T, CACHE_T_TYPE, KV_DTYPE_VAL>(   \
      reinterpret_cast<const KV_T*>(qkv.data_ptr()),                          \
      reinterpret_cast<const KV_T*>(q_weight.data_ptr()),                     \
      reinterpret_cast<const KV_T*>(k_weight.data_ptr()),                     \
      reinterpret_cast<const KV_T*>(cos_sin_cache.data_ptr()),                \
      positions.data_ptr<int64_t>(), pos_stride,                              \
      slot_mapping.data_ptr<int64_t>(), block_to_seq.data_ptr<int32_t>(),     \
      block_to_group_in_seq.data_ptr<int32_t>(),                              \
      query_start_loc.data_ptr<int32_t>(), num_tokens,                        \
      static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),           \
      static_cast<int>(num_heads_v), static_cast<int>(head_dim), is_neox,     \
      static_cast<float>(eps), k_scale_f, v_scale_f, use_shuffle_layout,      \
      static_cast<int>(block_size), static_cast<int>(x), total_kv_blocks,     \
      reinterpret_cast<KV_T*>(q_out.data_ptr()),                              \
      reinterpret_cast<CACHE_T_TYPE*>(k_cache.data_ptr()),                    \
      reinterpret_cast<CACHE_T_TYPE*>(v_cache.data_ptr()),                    \
      (return_kv && k_out.has_value())                                        \
          ? reinterpret_cast<KV_T*>(k_out->data_ptr())                        \
          : nullptr,                                                          \
      (return_kv && v_out.has_value())                                        \
          ? reinterpret_cast<KV_T*>(v_out->data_ptr())                        \
          : nullptr,                                                          \
      stream)

  vllm::Fp8KVCacheDataType kv_dtype_enum =
      vllm::get_fp8_kv_cache_data_type(kv_cache_dtype);

  if (kv_dtype_enum == vllm::Fp8KVCacheDataType::kAuto) {
    if (qkv.scalar_type() == at::ScalarType::BFloat16) {
      DISPATCH_BODY(__nv_bfloat16, __nv_bfloat16,
                    vllm::Fp8KVCacheDataType::kAuto);
    } else if (qkv.scalar_type() == at::ScalarType::Half) {
      DISPATCH_BODY(_Float16, _Float16, vllm::Fp8KVCacheDataType::kAuto);
    } else {
      TORCH_CHECK(false, "Unsupported qkv dtype: ", qkv.scalar_type());
    }
  } else if (kv_dtype_enum == vllm::Fp8KVCacheDataType::kFp8E4M3) {
    if (qkv.scalar_type() == at::ScalarType::BFloat16) {
      DISPATCH_BODY(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (qkv.scalar_type() == at::ScalarType::Half) {
      DISPATCH_BODY(_Float16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else {
      TORCH_CHECK(false, "Unsupported qkv dtype: ", qkv.scalar_type());
    }
  } else {
    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
  }

#undef DISPATCH_BODY
}
