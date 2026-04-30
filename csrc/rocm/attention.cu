/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_fp8.h>
#include <hip/hip_bf16.h>
#include "../cuda_compat.h"

#include <algorithm>
#include "../attention/dtype_fp8.cuh"
#include "../quantization/w8a8/fp8/amd/quant_utils.cuh"

// ROCm 6.2 compatibility: map OCP fp8 types to FNUZ variants if OCP is absent
#if !defined(HIP_FP8_TYPE_OCP)
using __hip_fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __hip_fp8_e5m2 = __hip_fp8_e5m2_fnuz;
#endif

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__FP8MFMA__
#endif

#if defined(__HIPCC__) && (defined(__gfx1100__) || defined(__gfx1101__) || \
                           defined(__gfx1150__) || defined(__gfx1151__))
  #define __HIP__GFX11__
#endif

#if defined(__HIPCC__) && (defined(__gfx1200__) || defined(__gfx1201__))
  #define __HIP__GFX12__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

enum class MFMAType {
  F16 = 0,
  Fp8 = 1,
  Fp4 = 2,
};

#if defined(__HIP__GFX9__)

  #define GCN_MFMA_INSTR1 __builtin_amdgcn_mfma_f32_16x16x4f32
  #define GCN_MFMA_INSTR __builtin_amdgcn_mfma_f32_4x4x4f16

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
typedef float16x2 _Half2;
typedef struct _Half8 {
  _Half4 xy[2];
} _Half8;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8 {
  _B16x4 xy[2];
} _B16x8;

using _B8x8 = uint2;
using _B8x4 = int32_t;  // used in builtins
using bit8_t = uint8_t;

// NOTE(jhu96): CDNA4 (gfx950) hardware-accelerated LDS transposed loads
// (ds_read_b64_tr_b16, ds_read_b64_tr_b8) are intentionally NOT integrated
// into this kernel — both CDNA3 (gfx942) and CDNA4 (gfx950) use the
// software LDS transpose path in Phase 2 (V-fetch).
//
// Reason: the hw-tr instructions deliver a per-lane layout that does not
// match what our SV-MFMA operand A expects. Empirical diagnostic
// (see /home/jackhu12/vllm-fusion-runs/diagnose_ds_read_tr*.hip):
//
//   ds_read_tr16_b64 returns, per 16-lane block, a tile in
//   "groups-of-4 lanes share head_dim, 4 lane-groups share token-chunk"
//   form. Our SV-MFMA expects "16 unique head_dims across 16 lanes, all
//   sharing the same 4 tokens" (row K = head_dim K, K-axis = tokens).
//
// Bridging the two requires either (a) cross-lane DPP/permute after the
// hw-tr (kills the perf win), or (b) cascading layout changes through
// v_lds, Vlocal, the SV-MFMA call site, and softmax (multi-kernel
// refactor). Until (b) is implemented the software LDS transpose is the
// correct & verified path. Cost: ~7 extra scalar 16-bit ds_read
// instructions per Vlocal slot per lane, executed in LDS-bandwidth-bound
// time — sub-percent end-to-end vs hw-tr.
//
// Empirical correctness: software path produces FLASH-vs-PAGED attention
// output cosim = 1.0000 on identical K/V, while the hw-tr path gave 0.25
// (catastrophic — gsm8k collapses to random baseline ~0.01).

typedef struct _B8x16 {
  _B8x8 xy[2];
} _B8x16;

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma4x4x4_instr(const _B16x4& inpA,
                                                       const _B16x4& inpB,
                                                       const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4f16(inpA, inpB, inpC, absz, cbid,
                                              blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(inpA, inpB, inpC, absz, cbid,
                                                  blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                          const _B16x4& inpB,
                                                          const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(inpA, inpB, inpC, absz, cbid,
                                                 blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz,
                                                     cbid, blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x32_instr(const long& inpA,
                                                          const long& inpB,
                                                          const floatx4& inpC) {
  if constexpr (std::is_same<T, __hip_fp8_e4m3>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(inpA, inpB, inpC, absz,
                                                      cbid, blgp);
  } else if constexpr (std::is_same<T, __hip_fp8_e5m2>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(inpA, inpB, inpC, absz,
                                                      cbid, blgp);
  } else {
    static_assert(false, "unsupported 8b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (_Float16)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __float2bfloat16(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4(const floatx4& inp) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      __half2 h2[2];
      _B16x4 b16x4;
    } u;
    u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
    u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
    return u.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        uint32_t u32;
        float f32;
      } u;
      u.f32 = inp[i];
      u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // BF16 RNE with no nan/inf check
      ret[i] = uint16_t(u.u32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 addx4(const _B16x4& inp1,
                                        const _B16x4& inp2) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      _B16x4 b16x4;
      __half2 h2[2];
    } u1, u2, s;
    u1.b16x4 = inp1;
    u2.b16x4 = inp2;
    s.h2[0] = u1.h2[0] + u2.h2[0];
    s.h2[1] = u1.h2[1] + u2.h2[1];
    return s.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        float f32;
        uint32_t i32;
      } u1, u2, s;
      u1.i32 = uint32_t(inp1[i]) << 16;
      u2.i32 = uint32_t(inp2[i]) << 16;
      s.f32 = u1.f32 + u2.f32;
      ret[i] = uint16_t(s.i32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

__device__ __forceinline__ floatx4 to_float_fp8x4(const _B8x4& inp) {
  // From MI300+ platforms, we have v_cvt_pk_f32_fp8 instruction
  // to convert 2 packed fp8 to 2 packed fp32 values.
  // However, in MI200 platforms, we only have v_cvt_f32_fp8
  // to convert fp8 values individually. So we added
  // #else case for fewer instructions (# inst=2) in MI300+,
  // and fallback to
  // #if case for other platforms (# inst=4).
  #if defined(__gfx90a__)
  float4 f32x4 = vllm::fp8::vec_conversion<float4, uint32_t>(
      *reinterpret_cast<const uint32_t*>(&inp));
  return *reinterpret_cast<floatx4*>(&f32x4);
  #else  // MI3xx+ optimized builtins
  const auto f0 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, false);
  const auto f1 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, true);
  floatx4 ret;
  ret[0] = f0[0];
  ret[1] = f0[1];
  ret[2] = f1[0];
  ret[3] = f1[1];
  return ret;
  #endif
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4_rtz(const floatx4& inp) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      _Half2 h2[2];
      _B16x4 b16x4;
    } u;
    u.h2[0] = __builtin_amdgcn_cvt_pkrtz(inp[0], inp[1]);
    u.h2[1] = __builtin_amdgcn_cvt_pkrtz(inp[2], inp[3]);
    return u.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        uint32_t i32;
        float f32;
      } u;
      u.f32 = inp[i];
      ret[i] = uint16_t(u.i32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x8 convert_b8x8_custom(const _B8x8 input) {
  union {
    _B8x8 b8x8;
    _B8x4 b8x4[2];
  } tmp;
  tmp.b8x8 = input;
  _B16x8 ret;
  for (int i = 0; i < 2; i++) {
    ret.xy[i] = from_floatx4_rtz<T>(to_float_fp8x4(tmp.b8x4[i]));
  }
  return ret;
}

typedef union u64_cvt {
  half f16x4[4];
  int16_t b16x4[4];
  _B8x8 b8x8;
  _B16x4 b64;
  int64_t i64;
} _T8x8;

__device__ __forceinline__ _B8x8 convert_b16x8(const _B16x8& input,
                                               _T8x8& Mtemp) {
  _T8x8 Qtmp8x8;

  for (int i = 0; i < 2; i++) {
    floatx4 q_out = {0, 0, 0, 0};
    q_out = gcn_mfma16x16x16_instr<_Float16, 0, 0, 0>(Mtemp.b64, input.xy[i],
                                                      q_out);
    Qtmp8x8.b16x4[i * 2] =
        __builtin_amdgcn_cvt_pk_fp8_f32(q_out[0], q_out[1], 0, false);
    Qtmp8x8.b16x4[i * 2 + 1] =
        __builtin_amdgcn_cvt_pk_fp8_f32(q_out[2], q_out[3], 0, false);
  }
  return Qtmp8x8.b8x8;
}

__device__ float warpReduceMax(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = max(
        val, __shfl_down(val, offset, WARP_SIZE));  // Using max() for reduction
  }
  return val;
}

// grid (num_seqs, num_partitions,num_kv_heads)
// block (256)
// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO, MFMAType MFMA_TYPE,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS, 5) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, ...] layout depends on KV_CACHE_LAYOUT
    const cache_t* __restrict__ v_cache,    // [num_blocks, ...] layout depends on KV_CACHE_LAYOUT
    const int num_kv_heads,   
    const float scale,    
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,   // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,             // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,           // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  // clang-format on
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;
  const int lane16id = laneid % 16;
  const int rowid = laneid / 16;

  const auto seq_idx = blockIdx.x;
  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx]) != 1) {
    return;
  }

  const auto partition_idx = blockIdx.y;

  constexpr int T_PAR_SIZE = 256;  // token partition size set to 256

  const auto max_num_partitions = gridDim.y;

  const int seq_len = seq_lens[seq_idx];

  const int partition_start_token_idx =
      partition_idx * T_PAR_SIZE;  // partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= seq_len) {
    return;
  }

  constexpr int GQA_RATIO4 = DIVIDE_ROUND_UP(GQA_RATIO, 4);

  // shared_logits is used for multiple purposes; for KV_CACHE_LAYOUT==1 (flash),
  // the same memory is reused as a V transpose buffer during the V-fetch phase
  // (shared_logits is dead between Q broadcast and cross-warp softmax reduction).
  __shared__ _B16x4 shared_logits[NWARPS][4][16][4];

  constexpr int SHARED_LOGITS_BYTES =
      NWARPS * 4 * 16 * 4 * static_cast<int>(sizeof(_B16x4));
  constexpr int V_LDS_BLOCK_BYTES =
      BLOCK_SIZE * HEAD_SIZE * static_cast<int>(sizeof(cache_t));
  constexpr int V_LDS_BLOCKS_PER_BATCH = SHARED_LOGITS_BYTES / V_LDS_BLOCK_BYTES;
  static_assert(V_LDS_BLOCKS_PER_BATCH >= 1,
                "Single V block must fit within shared_logits LDS region");

  // for QK mfma16x16, layout is QHead/Tokenx16 across every 16 lanes, 16 Bytes
  // HeadElements in each lane, 4x16B HeadElements across 4 rows of warp
  constexpr int ROWS_PER_WARP =
      WARP_SIZE / 16;  // rows refers to 16 lanes; refer DDP (Data Parallel
                       // Processing) terminology
  constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD =
      16 / sizeof(cache_t);  // 8 for 16 bit cache type, 16 for 8 bit types
  constexpr int QKHE_PER_FETCH =
      CONTIGUOUS_KV_ELEMS_16B_LOAD *
      ROWS_PER_WARP;  // each fetch across a warp fetches these many elements
  constexpr int QK_SIZE_RATIO =
      sizeof(scalar_t) /
      sizeof(cache_t);  // 1 for 16bit types, 2 for 8bit types
  constexpr int QKHELOOP = HEAD_SIZE / QKHE_PER_FETCH;  // 4xQKHE_16B across
                                                        // warp

  _B16x8 Qlocal[QKHELOOP]
               [QK_SIZE_RATIO];  // note that 16 contiguous elements of Q should
                                 // be fetched per lane for 8 bit cache types :
                                 // QK_SIZE_RATIO changes for this

  constexpr int CONTIGUOUS_SCALAR_ELEMS_16B = 16 / sizeof(scalar_t);

  constexpr int TOKENS_PER_WARP =
      T_PAR_SIZE /
      NWARPS;  // sub partition of tokens per warp for qk calculation
  constexpr int TLOOP =
      TOKENS_PER_WARP /
      16;  // each mfma16x16x16 instruction processes 16 tokens

  // can be interpreted as B8x16 for 8 bit types
  _B16x8 Klocal[TLOOP][QKHELOOP];

  const auto wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const auto wg_start_kv_head_idx = blockIdx.z;
  const auto total_num_heads = gridDim.z * GQA_RATIO;

  // for QK mfma, tokens in multiples of TOKENS_PER_WARP are spread across warps
  // each mfma takes QH16xT16x16HE across warp
  // repeat mfmas across QKHELOOP dimension
  // output layout from QKmfma : QH16xT4x4 16 qheads across 16 lanes, 16 tokens
  // across 4 rows x 4 tokens per lane

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int last_seq_block = num_seq_blocks - 1;

  const int* block_table_seq = block_tables + seq_idx * max_num_blocks_per_seq;

  int kphysical_block_number[TLOOP];
  #if defined(__HIP__FP8MFMA__)
  float q_max = 0;
  float q_scale = 1.0;
  #endif

  // fetch k physical block numbers
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int klocal_token_idx =
        TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
    const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
    const int kblock_idx = (kglobal_token_idx < seq_len)
                               ? kglobal_token_idx / BLOCK_SIZE
                               : last_seq_block;
    kphysical_block_number[token_depth] = block_table_seq[kblock_idx];
  }

  // fetch Q in shared across warps and then write to registers
  const int local_qhead_idx = 4 * warpid + rowid;
  const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
  const scalar_t* q_ptr =
      q + query_start_off * q_stride + global_qhead_idx * HEAD_SIZE;

  const int qhead_element = lane16id * CONTIGUOUS_SCALAR_ELEMS_16B;
  if ((local_qhead_idx < GQA_RATIO) && (qhead_element < HEAD_SIZE)) {
    const scalar_t* q_fetch_ptr = q_ptr + qhead_element;
    const _B16x8* q_fetch_ptr_16B =
        reinterpret_cast<const _B16x8*>(q_fetch_ptr);
    _B16x8 tmp = *q_fetch_ptr_16B;
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const int offset1 =
          lane16id /
          4;  // 16 contiguous chunks of head elems are spread across 4x4lanes
      shared_logits[offset1][lane4id][local_qhead_idx][0] = tmp.xy[0];
      shared_logits[offset1][lane4id][local_qhead_idx][1] = tmp.xy[1];
    } else {
      for (int i = 0; i < 2; i++) {
        const int head_elem = lane16id * 2 + i;  // element id in _B16x4 terms
        const int offset3 = head_elem % 4;
        const int offset2 = (head_elem / 4) % 4;
        const int offset1 = head_elem / 4 / 4;
        shared_logits[offset1][offset2][local_qhead_idx][offset3] = tmp.xy[i];
      }
    }
  }
  __syncthreads();
  for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
    for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
      for (int i = 0; i < 2; i++) {
        Qlocal[qkhe_depth][qkratio].xy[i] =
            shared_logits[qkhe_depth][rowid][lane16id % GQA_RATIO]
                         [2 * qkratio + i];
  #if defined(__HIP__FP8MFMA__)
        if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto &&
                      MFMA_TYPE == MFMAType::Fp8) {
          scalar_t* qptr =
              reinterpret_cast<scalar_t*>(&Qlocal[qkhe_depth][qkratio].xy[i]);
          for (int k = 0; k < 4; k++)
            q_max = fmax(fabs(to_float<scalar_t>(qptr[k])), q_max);
        }
  #endif
      }
    }
  }

  constexpr int KX =
      16 / sizeof(cache_t);  // vLLM defines x as 16 Bytes of kv cache elements

  const int row_head_elem = rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;
  // fetch K values
  if constexpr (KV_CACHE_LAYOUT == 1) {
    // Flash K: [blocks, block_size, heads, head_dim] — head_dim contiguous
    // kv_head_stride holds the token stride (num_kv_heads * HEAD_SIZE)
    const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * HEAD_SIZE;
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      const int64_t kblock_number =
          static_cast<int64_t>(kphysical_block_number[token_depth]);
      const cache_t* k_ptr2 = k_ptr + kblock_number * kv_block_stride;
      const int klocal_token_idx =
          TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
      const int kphysical_block_offset = klocal_token_idx % BLOCK_SIZE;
      const cache_t* k_ptr3 = k_ptr2 + kphysical_block_offset * kv_head_stride;
      for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        const int head_elem = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
        Klocal[token_depth][qkhe_depth] =
            *reinterpret_cast<const _B16x8*>(k_ptr3 + head_elem);
      }
    }
  } else {
    // Paged/Shuffle K: [blocks, heads, hd/x, block_size, x]
    const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * kv_head_stride;
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      const int64_t kblock_number =
          static_cast<int64_t>(kphysical_block_number[token_depth]);
      const cache_t* k_ptr2 = k_ptr + kblock_number * kv_block_stride;
      const int klocal_token_idx =
          TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
      const int kphysical_block_offset = klocal_token_idx % BLOCK_SIZE;
      const cache_t* k_ptr3 = k_ptr2 + kphysical_block_offset * KX;
      for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        const int head_elem = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
        const int offset1 = head_elem / KX;
        const int offset2 = head_elem % KX;
        const cache_t* k_fetch_ptr =
            k_ptr3 + offset1 * BLOCK_SIZE * KX + offset2;
        Klocal[token_depth][qkhe_depth] =
            *reinterpret_cast<const _B16x8*>(k_fetch_ptr);
      }
    }
  }

  float alibi_slope;
  if constexpr (ALIBI_ENABLED) {
    const int alibi_head_idx = wg_start_head_idx + lane16id;
    alibi_slope = (lane16id < GQA_RATIO) ? alibi_slopes[alibi_head_idx] : 0.f;
  }

  constexpr int VTOKENS_PER_LANE =
      TOKENS_PER_WARP / ROWS_PER_WARP;  // 64/4 = 16 contiguous vtokens per lane
  constexpr int VBLOCKS_PER_LANE =
      1;  // assumes block size >=16, each lane can correspond to 1 block only
  constexpr int VTLOOP = NWARPS;  // corresponds to tokens across warps
  constexpr int VTLANELOOP = DIVIDE_ROUND_UP(
      VTOKENS_PER_LANE,
      CONTIGUOUS_KV_ELEMS_16B_LOAD);  // optimized for 16B fetches; assumes
                                      // minimum block size is 16
  constexpr int VHELOOP = HEAD_SIZE / 16 / NWARPS;

  int vphysical_block_number[VTLOOP][VBLOCKS_PER_LANE];

  // fetch v physical block numbers
  for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
    for (int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE;
         vblock_depth++) {
      const int vlocal_token_idx =
          vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP +
          rowid * VTOKENS_PER_LANE + vblock_depth * BLOCK_SIZE;
      // Safe to use an int32_t here assuming we are working with < 2 billion
      // tokens
      const int vglobal_token_idx =
          partition_start_token_idx + vlocal_token_idx;
      const int vblock_idx = (vglobal_token_idx < seq_len)
                                 ? vglobal_token_idx / BLOCK_SIZE
                                 : last_seq_block;
      vphysical_block_number[vtoken_depth][vblock_depth] =
          block_table_seq[vblock_idx];
    }
  }

  _B16x8 Vlocal[VTLOOP][VHELOOP][VTLANELOOP];  // this could be B8x16 too

  // KX_V = number of contiguous token elements per interleaved chunk
  // (e.g. 8 for bf16, 16 for fp8). This is also the 'x' dimension size.
  constexpr int KX_V = 16 / sizeof(cache_t);

  if constexpr (KV_CACHE_LAYOUT == 1) {
    // Flash V: [blocks, block_size, heads, head_dim] — head_dim contiguous
    // Cooperative coalesced load into LDS + transposed read to produce
    // token-ordered Vlocal registers for the SV MFMA.
    // shared_logits memory is reused as the transpose buffer (dead at this point).
    cache_t (*v_lds)[HEAD_SIZE] = reinterpret_cast<cache_t (*)[HEAD_SIZE]>(
        shared_logits);

    constexpr int TOTAL_V_BLOCKS = VTLOOP * ROWS_PER_WARP;
    constexpr int V_BATCHES = DIVIDE_ROUND_UP(TOTAL_V_BLOCKS, V_LDS_BLOCKS_PER_BATCH);

    constexpr int LANES_PER_TOKEN =
        (HEAD_SIZE * static_cast<int>(sizeof(cache_t))) / 16;
    constexpr int TOKENS_PER_LOAD = NUM_THREADS / LANES_PER_TOKEN;
    constexpr int LOADS_PER_BLOCK = DIVIDE_ROUND_UP(BLOCK_SIZE, TOKENS_PER_LOAD);

    const int thread_token_base =
        static_cast<int>(threadIdx.x) / LANES_PER_TOKEN;
    const int thread_dim_start =
        (static_cast<int>(threadIdx.x) % LANES_PER_TOKEN) *
        (16 / static_cast<int>(sizeof(cache_t)));

    const cache_t* v_ptr_head = v_cache + wg_start_kv_head_idx * HEAD_SIZE;

    // Optimization 4: Hoist per-thread step-2 column-base LDS pointers out
    // of the batch loop. v_lds_col_base[vhe] points to v_lds row 0 at this
    // thread's unique head_dim — depends only on warpid/lane16id/vhe_depth
    // (all loop-invariant), so it's computed once per kernel invocation and
    // its address-arithmetic ALU latency is overlapped with the
    // step-1-write → step-2-read __syncthreads() barrier wait below.
    cache_t* v_lds_col_base[VHELOOP];
    #pragma unroll
    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
      const int vhead_elem =
          vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
      v_lds_col_base[vhe_depth] = &v_lds[0][vhead_elem];
    }

    for (int batch = 0; batch < V_BATCHES; batch++) {
      const int batch_block_start = batch * V_LDS_BLOCKS_PER_BATCH;
      const int batch_block_count =
          ((batch_block_start + V_LDS_BLOCKS_PER_BATCH) <= TOTAL_V_BLOCKS)
              ? V_LDS_BLOCKS_PER_BATCH
              : (TOTAL_V_BLOCKS - batch_block_start);

      // Step 1: All 256 threads cooperatively load blocks into LDS.
      // Flash layout has head_dim contiguous per token, so each thread
      // loads 16B from one token's head_dim row via global_load_dwordx4.
      for (int b = 0; b < batch_block_count; b++) {
        const int flat_block_idx = batch_block_start + b;
        const int target_vtoken_depth = flat_block_idx / ROWS_PER_WARP;
        const int target_row = flat_block_idx % ROWS_PER_WARP;

        const int vlocal =
            target_vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP +
            target_row * VTOKENS_PER_LANE;
        const int vglobal = partition_start_token_idx + vlocal;
        const int vblock_idx =
            (vglobal < seq_len) ? vglobal / BLOCK_SIZE : last_seq_block;
        const int64_t phys_block =
            static_cast<int64_t>(block_table_seq[vblock_idx]);
        const cache_t* v_block_ptr =
            v_ptr_head + phys_block * kv_block_stride;

        for (int load = 0; load < LOADS_PER_BLOCK; load++) {
          const int token_in_block =
              load * TOKENS_PER_LOAD + thread_token_base;
          if (token_in_block < BLOCK_SIZE) {
            *reinterpret_cast<uint4*>(
                &v_lds[b * BLOCK_SIZE + token_in_block][thread_dim_start]) =
                *reinterpret_cast<const uint4*>(
                    &v_block_ptr[token_in_block * kv_head_stride +
                                 thread_dim_start]);
          }
        }
      }

      __syncthreads();

      // Step 2: Software LDS transpose — each row reads its block's data
      // from LDS in token-ordered fashion to produce the Vlocal register
      // layout expected by the SV MFMA.
      //
      // All reads below are from LDS (populated by Step 1's cooperative
      // coalesced HBM-to-LDS load) — no global gathers. Each lane issues
      // CONTIGUOUS_KV_ELEMS_16B_LOAD strided LDS halfword/byte reads that
      // collectively form the token-dim → head-dim transpose.
      //
      // Both CDNA3 (gfx942) and CDNA4 (gfx950) use this path. The CDNA4
      // hardware LDS transpose intrinsics (ds_read_b64_tr_b16,
      // ds_read_b64_tr_b8) are not integrated — see the top-of-file note
      // for the layout-mismatch reason and empirical correctness data.
      for (int b = 0; b < batch_block_count; b++) {
        const int flat_block_idx = batch_block_start + b;
        const int target_vtoken_depth = flat_block_idx / ROWS_PER_WARP;
        const int target_row = flat_block_idx % ROWS_PER_WARP;

        if (rowid == target_row) {
          // All reads below are from LDS (populated by Step 1's cooperative
          // coalesced load) — not from global memory. Each lane issues
          // CONTIGUOUS_KV_ELEMS_16B_LOAD strided LDS halfword/byte reads
          // that collectively form the token-dim → head-dim transpose.
          //
          // Optimization 1: pair-friendly addressing for ds_read2_b16
          // emission. The hoisted v_lds_col_base[vhe_depth] gives this
          // lane its unique head_dim base in v_lds row 0. We then offset
          // by the row stride (HEAD_SIZE elements per token row) for each
          // pair of consecutive tokens.
          //
          // Inner pair loop pattern:
          //   pair_base = col_ptr + p * HEAD_SIZE   // bumps every 2 rows
          //   temp[p]   = pair_base[0]              // row p's halfword
          //   temp[p+1] = pair_base[HEAD_SIZE]      // row p+1's halfword
          //
          // The two reads within a pair have offsets (0, HEAD_SIZE elems)
          // = (0, 128 halfwords / 256 B) for HD=128 bf16. Both fit in the
          // ds_read2_b16 8-bit halfword-scaled offset slots (≤ 255). With
          // pair_base rebased every 2 rows the compiler's
          // SILoadStoreOptimizer pass can lower each pair to a single
          // ds_read2_b16 (one LDS instruction = two 16-bit reads).
          // Result: 8 scalar ds_read_u16 → 4 ds_read2_b16 per Vlocal slot.
          //
          // For fp8 (cache_t = uint8_t) AMDGPU has no ds_read2_b8; the
          // compiler keeps individual ds_read_u8 here. No benefit, no
          // regression.
          #pragma unroll
          for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
            cache_t* col_base = v_lds_col_base[vhe_depth] +
                                b * BLOCK_SIZE * HEAD_SIZE;
            #pragma unroll
            for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP;
                 vfetch_depth++) {
              cache_t* col_ptr =
                  col_base +
                  vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD * HEAD_SIZE;
              cache_t temp[CONTIGUOUS_KV_ELEMS_16B_LOAD];
              #pragma unroll
              for (int p = 0; p < CONTIGUOUS_KV_ELEMS_16B_LOAD; p += 2) {
                cache_t* pair_base = col_ptr + p * HEAD_SIZE;
                temp[p]     = pair_base[0];
                temp[p + 1] = pair_base[HEAD_SIZE];
              }
              Vlocal[target_vtoken_depth][vhe_depth][vfetch_depth] =
                  *reinterpret_cast<const _B16x8*>(temp);
            }
          }
        }
      }

      // Sync before next batch overwrites LDS.
      // Last batch: no trailing sync needed — Phase 3 (QK MFMA + softmax)
      // is pure register/ALU work, and Phase 4's existing __syncthreads()
      // at the cross-warp softmax reduction guarantees all LDS reads complete.
      if (batch < V_BATCHES - 1) {
        __syncthreads();
      }
    }
  } else if constexpr (KV_CACHE_LAYOUT == 2) {
    // Shuffle V: [num_blocks, num_heads, block_size/x, head_dim, x]
    const cache_t* v_ptr_head = v_cache + wg_start_kv_head_idx * kv_head_stride;
    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
      const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
      for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
        const int vblock_depth = 0;
        const int64_t vblock_number = static_cast<int64_t>(
            vphysical_block_number[vtoken_depth][vblock_depth]);
        const cache_t* v_block_ptr =
            v_ptr_head + vblock_number * kv_block_stride;
        for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
          const int bo = ((rowid * VTOKENS_PER_LANE) % BLOCK_SIZE) +
                         vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD;
          const int bo_chunk = bo / KX_V;
          const cache_t* v_fetch_ptr =
              v_block_ptr + (bo_chunk * HEAD_SIZE + vhead_elem) * KX_V +
              (bo % KX_V);
          Vlocal[vtoken_depth][vhe_depth][vfetch_depth] =
              *reinterpret_cast<const _B16x8*>(v_fetch_ptr);
        }
      }
    }
  } else {
    // Paged V: [num_blocks, num_heads, head_dim, block_size]
    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride +
                           ((rowid * VTOKENS_PER_LANE) % BLOCK_SIZE);
    for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
      const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
      const cache_t* v_ptr2 = v_ptr + vhead_elem * BLOCK_SIZE;
      for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
        for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
          const int vblock_depth = 0;
          const int64_t vblock_number = static_cast<int64_t>(
              vphysical_block_number[vtoken_depth][vblock_depth]);
          const cache_t* v_ptr3 = v_ptr2 + (vblock_number * kv_block_stride);
          const cache_t* v_fetch_ptr =
              v_ptr3 + vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD;
          Vlocal[vtoken_depth][vhe_depth][vfetch_depth] =
              *reinterpret_cast<const _B16x8*>(v_fetch_ptr);
        }
      }
    }
  }

  // calculate post qk mfma scale
  float scale2 = scale;
  if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
    // multiply by k_scale if fp8 kv cache
    scale2 *= *k_scale;
  #if defined(__HIP__FP8MFMA__)
    q_max = warpReduceMax(q_max);
    constexpr float FP8_E4M3_SCALE_TARGET = 224.0f;
    if constexpr (MFMA_TYPE == MFMAType::Fp8) {
      q_scale = q_max > 0 ? FP8_E4M3_SCALE_TARGET / q_max : 1.0f;
      scale2 /= q_scale;
    }
  #endif
  }

  floatx4 d_out[TLOOP];
  // qk mfma
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    d_out[token_depth] = {0};
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
      if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
        for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
          for (int i = 0; i < 2; i++) {
            d_out[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                Klocal[token_depth][qkhe_depth].xy[i],
                Qlocal[qkhe_depth][qkratio].xy[i], d_out[token_depth]);
          }
        }
      } else {  // kv cache dtype fp8
        auto Ktmp = Klocal[token_depth][qkhe_depth];
        _B8x16 Ktmp8x16 = *reinterpret_cast<_B8x16*>(&Ktmp);
        for (int qkratio = 0; qkratio < QK_SIZE_RATIO; qkratio++) {
          if constexpr (MFMA_TYPE == MFMAType::F16) {
            _B8x8 Ktmp8x8 = Ktmp8x16.xy[qkratio];
            _B16x8 Klocaltmp = convert_b8x8_custom<scalar_t>(Ktmp8x8);
            for (int i = 0; i < 2; i++) {
              d_out[token_depth] = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                  Klocaltmp.xy[i], Qlocal[qkhe_depth][qkratio].xy[i],
                  d_out[token_depth]);
            }
          } else {
  #if defined(__HIP__FP8MFMA__)
            _T8x8 Ktmp8x8, Qtmp8x8;
            Ktmp8x8.b8x8 = Ktmp8x16.xy[qkratio];

            for (int n = 0; n < 2; n++) {
              scalar_t* qptr = reinterpret_cast<scalar_t*>(
                  &Qlocal[qkhe_depth][qkratio].xy[n]);

              Qtmp8x8.b16x4[n * 2] =
                  vllm::fp8::scaled_vec_conversion<uint16_t, float2>(
                      make_float2(to_float<scalar_t>(qptr[0]),
                                  to_float<scalar_t>(qptr[1])),
                      q_scale);
              Qtmp8x8.b16x4[n * 2 + 1] =
                  vllm::fp8::scaled_vec_conversion<uint16_t, float2>(
                      make_float2(to_float<scalar_t>(qptr[2]),
                                  to_float<scalar_t>(qptr[3])),
                      q_scale);
            }

            d_out[token_depth] =
                gcn_mfma16x16x32_instr<__hip_fp8_e4m3, 0, 0, 0>(
                    Ktmp8x8.i64, Qtmp8x8.i64, d_out[token_depth]);
  #else
            UNREACHABLE_CODE
  #endif
          }
        }
      }
    }
    d_out[token_depth] *= scale2;
  }

  const int qkout_token_idx =
      partition_start_token_idx + TOKENS_PER_WARP * warpid + rowid * 4;

  // apply alibi
  if constexpr (ALIBI_ENABLED) {
    for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
      const int local_token_idx = qkout_token_idx + token_depth * 16;
      const int alibi_offset = local_token_idx - seq_len + 1;
      for (int i = 0; i < 4; i++) {
        d_out[token_depth][i] += alibi_slope * (alibi_offset + i);
      }
    }
  }

  // calculate qk_max and exp_sum per warp and write to shared memory
  float qk_max = -FLT_MAX;
  float exp_sum = 0.0f;

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 4; i++) {
      const float tmp =
          (local_token_idx + i < seq_len) ? d_out[token_depth][i] : -FLT_MAX;
      qk_max = fmaxf(qk_max, tmp);
    }
  }

  for (int mask = WARP_SIZE / 2; mask >= 16; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor(qk_max, mask));
  }

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 4; i++) {
      const float tmp = (local_token_idx + i < seq_len)
                            ? __expf(d_out[token_depth][i] - qk_max)
                            : 0.0f;
      d_out[token_depth][i] = tmp;
      exp_sum += tmp;
    }
  }

  for (int mask = WARP_SIZE / 2; mask >= 16; mask /= 2) {
    exp_sum += __shfl_xor(exp_sum, mask);
  }

  __syncthreads();  // sync before writing to shared mem

  float* shared_mem = reinterpret_cast<float*>(shared_logits);
  if (laneid < 16) {
    const int qk_max_offset = warpid * 16 + lane16id;
    shared_mem[qk_max_offset] = qk_max;
    const int exp_sum_offset = NWARPS * 16 + qk_max_offset;
    shared_mem[exp_sum_offset] = exp_sum;
  }

  __syncthreads();

  // calculate partition qk_max and exp_sum
  float partition_qk_max = -FLT_MAX;
  float warp_qk_max_exp[NWARPS];
  float partition_exp_sum = 0.0f;

  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = shared_mem[w * 16 + lane16id];
    partition_qk_max = fmaxf(partition_qk_max, warp_qk_max_exp[w]);
  }

  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = __expf(warp_qk_max_exp[w] - partition_qk_max);
    partition_exp_sum +=
        shared_mem[NWARPS * 16 + w * 16 + lane16id] * warp_qk_max_exp[w];
  }

  const float inv_sum_scale =
      __fdividef(1.f, partition_exp_sum + 1e-6f) * warp_qk_max_exp[warpid];

  __syncthreads();

  // disable rtz conversion due to its impact on accuracy.
  constexpr bool LOGITS_RTZ_CONVERSION = false;

  #if defined(__HIP__FP8MFMA__)
  int rowid_8x8 = rowid / 2;
  int offset = rowid % 2;
  #endif

  // write logits to shared mem
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    d_out[token_depth] *= inv_sum_scale;
    if constexpr (MFMA_TYPE != MFMAType::Fp8) {
      if constexpr (LOGITS_RTZ_CONVERSION) {
        // use rtz conversion for better performance, with negligible impact on
        // accuracy
        shared_logits[warpid][token_depth][lane16id][rowid] =
            from_floatx4_rtz<scalar_t>(d_out[token_depth]);
      } else {
        shared_logits[warpid][token_depth][lane16id][rowid] =
            from_floatx4<scalar_t>(d_out[token_depth]);
      }
    } else {
  #if defined(__HIP__FP8MFMA__)
      // cast _B16x4* to _B8x8*
      _T8x8& logits_8x8 = *reinterpret_cast<_T8x8*>(
          &shared_logits[warpid][token_depth][lane16id][rowid_8x8]);
      logits_8x8.b16x4[offset * 2] = __builtin_amdgcn_cvt_pk_fp8_f32(
          d_out[token_depth][0], d_out[token_depth][1], 0, false);
      logits_8x8.b16x4[offset * 2 + 1] = __builtin_amdgcn_cvt_pk_fp8_f32(
          d_out[token_depth][2], d_out[token_depth][3], 0, false);
  #else
      UNREACHABLE_CODE
  #endif
    }
  }

  // write out partition max_logits and exp_sum
  if (threadIdx.x < GQA_RATIO) {
    const int qhead_idx = lane16id;
    const int64_t offset = static_cast<int64_t>(seq_idx) *
                               static_cast<int64_t>(total_num_heads) *
                               static_cast<int64_t>(max_num_partitions) +
                           (static_cast<int64_t>(wg_start_head_idx) +
                            static_cast<int64_t>(qhead_idx)) *
                               static_cast<int64_t>(max_num_partitions) +
                           static_cast<int64_t>(partition_idx);
    max_logits[offset] = partition_qk_max;
    exp_sums[offset] = partition_exp_sum;
  }

  __syncthreads();

  constexpr int ELEMS8_ELEMS4_RATIO = 8 / 4;
  constexpr int ELEMS16_ELEMS8_RATIO = 16 / 8;

  _B16x4 outelems[VHELOOP];
  // Softmax V mfma
  // v layout: 16he across lanes x 16 tokens per lane
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    floatx4 tmp_out = {0};

    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
      if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
        for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
          for (int i = 0; i < ELEMS8_ELEMS4_RATIO; i++) {
            const int offset = rowid * VTLANELOOP * ELEMS8_ELEMS4_RATIO +
                               vfetch_depth * ELEMS8_ELEMS4_RATIO + i;
            const int offset1 = offset % ROWS_PER_WARP;
            const int offset2 = offset / ROWS_PER_WARP;
            // output format is 16 qheads across 16 lanes, 16 head elems spread
            // across 4 rows
            tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                Vlocal[vtoken_depth][vhe_depth][vfetch_depth].xy[i],
                shared_logits[vtoken_depth][offset2][lane16id][offset1],
                tmp_out);
          }
        }
        // KV cache fp8
      } else {
        for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
          _B16x8 Vtmp = Vlocal[vtoken_depth][vhe_depth][vfetch_depth];
          // reinterpret V format as 16 elements of 8bits
          _B8x16 Vtmp8x16 = *reinterpret_cast<_B8x16*>(&Vtmp);
          for (int j = 0; j < ELEMS16_ELEMS8_RATIO; j++) {
            _B8x8 Vtmp8x8 = Vtmp8x16.xy[j];
            if constexpr (MFMA_TYPE == MFMAType::F16) {
              _B16x8 Vlocaltmp = convert_b8x8_custom<scalar_t>(Vtmp8x8);
              for (int i = 0; i < ELEMS8_ELEMS4_RATIO; i++) {
                const int offset =
                    rowid * ELEMS16_ELEMS8_RATIO * ELEMS8_ELEMS4_RATIO +
                    j * ELEMS8_ELEMS4_RATIO + i;
                const int offset1 = offset % ROWS_PER_WARP;
                const int offset2 = offset / ROWS_PER_WARP;
                // output format is 16 qheads across 16 lanes, 16 head elems
                // spread across 4 rows
                tmp_out = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(
                    Vlocaltmp.xy[i],
                    shared_logits[vtoken_depth][offset2][lane16id][offset1],
                    tmp_out);
              }
            } else {
  #if defined(__HIP__FP8MFMA__)
              for (int i = 0; i < ELEMS8_ELEMS4_RATIO / 2; i++) {
                const int offset =
                    rowid * ELEMS16_ELEMS8_RATIO * ELEMS8_ELEMS4_RATIO +
                    j * ELEMS8_ELEMS4_RATIO + i;
                const int offset1 = (offset % ROWS_PER_WARP) / 2;
                const int offset2 = offset / ROWS_PER_WARP;
                // output format is 16 qheads across 16 lanes, 16 head elems
                // spread across 4 rows
                tmp_out = gcn_mfma16x16x32_instr<__hip_fp8_e4m3, 0, 0, 0>(
                    reinterpret_cast<_T8x8*>(&Vtmp8x8)->i64,
                    reinterpret_cast<_T8x8*>(
                        &shared_logits[vtoken_depth][offset2][lane16id]
                                      [offset1])
                        ->i64,
                    tmp_out);
              }
  #else
              UNREACHABLE_CODE
  #endif
            }
          }
        }
      }
    }
    // apply post Softmax V mfma v_scale
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
      tmp_out *= *v_scale;
    }
    outelems[vhe_depth] = from_floatx4<scalar_t>(tmp_out);
  }

  __syncthreads();

  // store Softmax-V mfma output to shared mem
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    // lane16 id head dimension; rowid head element dimension
    shared_logits[warpid][vhe_depth][lane16id][rowid] = outelems[vhe_depth];
  }

  __syncthreads();

  // write to tmp_out with coalesced writes after reading from shared mem
  if (warpid == 0) {
    _B16x8 vout[GQA_RATIO4];
    // each lane writes out 16Bytes of tmp_out along head elem dimension
    const int head_elem_idx = lane16id * 8;
    if (head_elem_idx < HEAD_SIZE) {
      for (int h = 0; h < GQA_RATIO4; h++) {
        const int local_head_idx = 4 * h + rowid;
        const int offset1 = (head_elem_idx / 16) % 4;
        const int offset2 = head_elem_idx / 16 / NWARPS;
        const int offset3 = (head_elem_idx / 4) % 4;
        for (int i = 0; i < 2; i++) {
          vout[h].xy[i] =
              shared_logits[offset1][offset2][local_head_idx][offset3 + i];
        }
      }

      const int64_t hsz_maxp_mult =
          static_cast<int64_t>(HEAD_SIZE * max_num_partitions);
      scalar_t* out_ptr = out + seq_idx * total_num_heads * hsz_maxp_mult +
                          partition_idx * HEAD_SIZE;
      for (int h = 0; h < GQA_RATIO4; h++) {
        const int local_head_idx = 4 * h + rowid;
        if (local_head_idx < GQA_RATIO) {
          const int64_t out_head_idx =
              static_cast<int64_t>(wg_start_head_idx + local_head_idx);
          scalar_t* out_ptr2 = out_ptr + out_head_idx * hsz_maxp_mult;
          scalar_t* out_ptr3 = out_ptr2 + head_elem_idx;
          _B16x8* out_ptr_B16x8 = reinterpret_cast<_B16x8*>(out_ptr3);
          *out_ptr_B16x8 = vout[h];
        }
      }
    }
  }
}

// grid (num_seqs, num_partitions, num_kv_heads)
// block (256 : partition size)
// each WG handles 1 partition per sequence
// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma4_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, ...] layout depends on KV_CACHE_LAYOUT
    const cache_t* __restrict__ v_cache,    // [num_blocks, ...] layout depends on KV_CACHE_LAYOUT
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,   // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes, // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,             // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,           // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  // clang-format on
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const auto warpid = threadIdx.x / WARP_SIZE;
  const auto laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;

  const auto seq_idx = blockIdx.x;
  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1)) {
    return;
  }
  const auto partition_idx = blockIdx.y;
  const auto partition_size = blockDim.x;
  const auto max_num_partitions = gridDim.y;

  const int seq_len = seq_lens[seq_idx];
  const int partition_start_token_idx = partition_idx * partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= seq_len) {
    return;
  }
  // every 4 lanes fetch 4 different qheads
  // qhloop = num loops over qhead dimension
  constexpr int QHLOOP = DIVIDE_ROUND_UP(GQA_RATIO, 4);
  constexpr int GQA_RATIO4 = 4 * QHLOOP;
  __shared__ float shared_qk_max[NWARPS][GQA_RATIO4 + 1];
  __shared__ float shared_exp_sum[NWARPS][GQA_RATIO4 + 1];
  _B16x8 Qlocal[QHLOOP];
  constexpr int x = 16 / sizeof(scalar_t);
  // kheloop = num loops over head_size for 16Bytes of Q/dequantized K elements
  constexpr int KHELOOP = HEAD_SIZE / x;
  _B16x8 Klocal[KHELOOP];
  _B8x8 Klocalb8[KHELOOP];
  // for SoftMax-V Gemm, V head_size dimension is distributed across warp
  // vheloop = num loops to cover v head size dimension
  constexpr int VHELOOP = HEAD_SIZE / WARP_SIZE;
  // softmax out has warp_size tokens across warp
  // vtloop = num loops to cover warp_size(64) tokens with 16Bytes of
  // dequantized V elements
  constexpr int VTLOOP = WARP_SIZE / 8;
  // num vblocks to cover warp_size(64) v elements
  constexpr int VBLOCKS = 8 * VTLOOP / BLOCK_SIZE;
  int vphysical_blocks[VBLOCKS];
  _B16x8 Vlocal[VHELOOP][VTLOOP];
  _B8x8 Vlocalb8[VHELOOP][VTLOOP];
  floatx4 d_out[QHLOOP];
  float qk_max[QHLOOP];

  // vout_shared is used for the final cross-warp SV MFMA accumulation (Phase 6).
  // For KV_CACHE_LAYOUT == 1 (flash), the same LDS memory is reused as a
  // per-warp V transpose buffer during the V-fetch phase. These two uses are
  // temporally disjoint: V-fetch completes before Phase 4's __syncthreads(),
  // and vout_shared is first written during the SV MFMA that follows Phase 5.
  //
  // The per-warp v_warp_lds region allows each warp to cooperatively coalesced-
  // load a V block and read it back in transposed order, without any
  // __syncthreads() (safe because each warp uses its own LDS region and
  // wavefront lockstep guarantees intra-warp ordering). This avoids the
  // warp-level-branching hazard that prevents the mfma16-style cooperative
  // LDS transpose in this kernel.
  //
  // v_warp_lds is only needed for flash layout (KV_CACHE_LAYOUT == 1). For
  // paged (0) and shuffle (2) layouts the V-fetch reads directly from global
  // memory into Vlocal, so v_warp_lds is never referenced. In those cases we
  // collapse its array dimensions to 1x1x1 so the union size is governed by
  // vout_shared alone (5 KiB).
  //
  // For flash layout, we cap v_warp_lds at V_WARP_TOKENS_PER_CHUNK = 16
  // tokens per warp (independent of BLOCK_SIZE). When BLOCK_SIZE > 16, each
  // block is processed in V_SUB_BATCHES_PER_BLOCK = BLOCK_SIZE/16 chunks,
  // each reusing the same 16-token v_warp_lds region in wavefront lockstep.
  // This keeps LDS/WG ≤ 16.25 KiB for every supported config so the mfma4
  // kernel always fits the 3 WG/CU VGPR-bound occupancy target on CDNA3
  // (budget ~21.3 KiB); without this cap BLOCK_SIZE=32, HEAD_SIZE=128, bf16
  // would allocate 32 KiB v_warp_lds and drop to 1 WG/CU on CDNA3.
  //
  // The total number of load+read iterations per warp is invariant:
  //   VBLOCKS * V_SUB_BATCHES_PER_BLOCK = (64/BLOCK_SIZE) * (BLOCK_SIZE/16) = 4
  // regardless of BLOCK_SIZE, so global memory bandwidth usage is unchanged.
  //
  // All accesses to v_warp_lds are guarded by `if constexpr (KV_CACHE_LAYOUT
  // == 1)` so the 1x1x1 fallback is never addressed at runtime.
  constexpr int V_WARP_TOKENS_PER_CHUNK =
      (BLOCK_SIZE < 16) ? BLOCK_SIZE : 16;
  static_assert(BLOCK_SIZE % V_WARP_TOKENS_PER_CHUNK == 0,
                "BLOCK_SIZE must be a multiple of V_WARP_TOKENS_PER_CHUNK");
  constexpr int V_SUB_BATCHES_PER_BLOCK =
      BLOCK_SIZE / V_WARP_TOKENS_PER_CHUNK;

  constexpr int V_WARP_LDS_DIM_0 = (KV_CACHE_LAYOUT == 1) ? NWARPS : 1;
  constexpr int V_WARP_LDS_DIM_1 =
      (KV_CACHE_LAYOUT == 1) ? V_WARP_TOKENS_PER_CHUNK : 1;
  constexpr int V_WARP_LDS_DIM_2 = (KV_CACHE_LAYOUT == 1) ? HEAD_SIZE : 1;
  __shared__ union {
    _B16x4 vout_shared[QHLOOP][VHELOOP][WARP_SIZE][NWARPS + 1];
    cache_t v_warp_lds[V_WARP_LDS_DIM_0][V_WARP_LDS_DIM_1][V_WARP_LDS_DIM_2];
  };

  for (int h = 0; h < QHLOOP; h++) {
    d_out[h] = {0};
    qk_max[h] = -FLT_MAX;
  }

  const auto wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const auto wg_start_kv_head_idx = blockIdx.z;

  const int warp_start_token_idx =
      partition_start_token_idx + warpid * WARP_SIZE;

  if (warp_start_token_idx >= seq_len) {  // warp out of context
  #pragma unroll
    for (int h = 0; h < GQA_RATIO4; h++) {
      shared_qk_max[warpid][h] = -FLT_MAX;
      shared_exp_sum[warpid][h] = 0.0f;
    }
  } else {  // warp within context

    const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
    const int last_seq_block = num_seq_blocks - 1;

    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
    // token id within partition
    const auto local_token_idx = threadIdx.x;
    // token id within sequence
    const int global_token_idx = partition_start_token_idx + local_token_idx;

    // fetch block number for k
    const int block_idx = (global_token_idx < seq_len)
                              ? global_token_idx / BLOCK_SIZE
                              : last_seq_block;

    // fetch k physical block number
    //  int32 physical_block_number leads to overflow when multiplied with
    //  kv_block_stride
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // fetch vphysical block numbers up front
    const int warp_start_block_idx = warp_start_token_idx / BLOCK_SIZE;
    for (int b = 0; b < VBLOCKS; b++) {
      const int vblock_idx = warp_start_block_idx + b;
      const int vblock_idx_ctx =
          (vblock_idx <= last_seq_block) ? vblock_idx : last_seq_block;
      vphysical_blocks[b] = block_table[vblock_idx_ctx];
    }

    // fetch q elements
    // every 4 lanes fetch 8 elems, so warp fetches 8*16 = 128 elemsc
    const int64_t query_start_off = static_cast<int64_t>(
        query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
    const scalar_t* q_ptr =
        q + query_start_off * q_stride + wg_start_head_idx * HEAD_SIZE;
    const _B16x8* q_ptrh8 = reinterpret_cast<const _B16x8*>(q_ptr);
    const int qhead_elemh8 = laneid / 4;

    for (int h = 0; h < QHLOOP - 1; h++) {
      const int qhead_idx = h * 4 + lane4id;
      Qlocal[h] = q_ptrh8[qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    }
    const int final_qhead_idx = 4 * (QHLOOP - 1) + lane4id;
    if (final_qhead_idx < GQA_RATIO) {
      Qlocal[QHLOOP - 1] =
          q_ptrh8[final_qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    } else {
      Qlocal[QHLOOP - 1].xy[0] = {0};
      Qlocal[QHLOOP - 1].xy[1] = {0};
    }

    // fetch k elements
    const int physical_block_offset = local_token_idx % BLOCK_SIZE;

    if constexpr (KV_CACHE_LAYOUT == 1) {
      // Flash K: [blocks, block_size, heads, head_dim] — head_dim contiguous
      const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                             wg_start_kv_head_idx * HEAD_SIZE +
                             physical_block_offset * kv_head_stride;
      if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
        const _B16x8* k_ptrh8 = reinterpret_cast<const _B16x8*>(k_ptr);
        for (int d = 0; d < KHELOOP; d++) {
          Klocal[d] = k_ptrh8[d];
        }
      } else {
        const _B8x8* k_ptrb8 = reinterpret_cast<const _B8x8*>(k_ptr);
        for (int d = 0; d < KHELOOP; d++) {
          Klocalb8[d] = k_ptrb8[d];
        }
      }
    } else {
      // Paged/Shuffle K: [blocks, heads, hd/x, block_size, x]
      const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                             wg_start_kv_head_idx * kv_head_stride;
      if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
        const _B16x8* k_ptrh8 = reinterpret_cast<const _B16x8*>(k_ptr);
        for (int d = 0; d < KHELOOP; d++) {
          Klocal[d] = k_ptrh8[d * BLOCK_SIZE + physical_block_offset];
        }
      } else {
        constexpr int X = 16 / sizeof(cache_t);
        const cache_t* k_ptr2 = k_ptr + physical_block_offset * X;
        for (int d = 0; d < KHELOOP; d++) {
          const int head_elem = d * 8;
          const int offset1 = head_elem / X;
          const int offset2 = head_elem % X;
          const cache_t* k_ptr3 = k_ptr2 + offset1 * BLOCK_SIZE * X + offset2;
          Klocalb8[d] = *reinterpret_cast<const _B8x8*>(k_ptr3);
        }
      }
    }

    // optional alibi fetch
    float alibi_slope[QHLOOP];
    if constexpr (ALIBI_ENABLED) {
      for (int h = 0; h < QHLOOP; h++) {
        const int qhead_idx = h * 4 + lane4id;
        alibi_slope[h] = (qhead_idx < GQA_RATIO)
                             ? alibi_slopes[wg_start_head_idx + qhead_idx]
                             : 0.f;
      }
    }

    constexpr int KX_V = 16 / sizeof(cache_t);

    if constexpr (KV_CACHE_LAYOUT == 1) {
      // Flash V: [blocks, block_size, heads, head_dim] — head_dim contiguous.
      //
      // Warp-level LDS transpose: each warp cooperatively coalesced-loads a
      // 16-token chunk of a V block into its own LDS region (v_warp_lds
      // [warpid]) using 16B/lane loads, then reads it back in transposed
      // order to produce token-ordered Vlocal / Vlocalb8 for the SV MFMA.
      //
      // Sub-batching: when BLOCK_SIZE > V_WARP_TOKENS_PER_CHUNK (= 16), each
      // block is processed in V_SUB_BATCHES_PER_BLOCK passes, reusing the
      // same 16-token v_warp_lds region. The load+read iteration count is
      // invariant (always 4 per warp for WARP_SIZE=64) regardless of BS,
      // but LDS footprint is capped at 16 tokens per warp — preserving
      // 3 WG/CU occupancy on CDNA3 even for BLOCK_SIZE=32 HD=128 bf16.
      //
      // No __syncthreads() is needed: each warp uses its own LDS region and
      // wavefront lockstep guarantees that all 64 lanes complete a store
      // before any lane executes the next instruction. This sidesteps the
      // warp-level branching (out-of-context warps skip this path) that
      // prevents the mfma16-style cooperative barrier approach.
      //
      // Replaces BLOCK_SIZE * VHELOOP strided scalar loads per block with
      // (V_WARP_TOKENS_PER_CHUNK * HEAD_SIZE * sizeof(cache_t) / 16) /
      // WARP_SIZE coalesced 16B loads per warp per chunk, achieving ~100%
      // cache-line utilization.
      constexpr int V_LANES_PER_TOKEN =
          (HEAD_SIZE * static_cast<int>(sizeof(cache_t))) / 16;
      static_assert(V_LANES_PER_TOKEN > 0 &&
                        WARP_SIZE % V_LANES_PER_TOKEN == 0,
                    "HEAD_SIZE * sizeof(cache_t) must be a multiple of 16B "
                    "and V_LANES_PER_TOKEN must divide WARP_SIZE");
      constexpr int V_TOKENS_PER_LOAD = WARP_SIZE / V_LANES_PER_TOKEN;
      static_assert(V_WARP_TOKENS_PER_CHUNK % V_TOKENS_PER_LOAD == 0,
                    "V_WARP_TOKENS_PER_CHUNK must be a multiple of "
                    "V_TOKENS_PER_LOAD");
      constexpr int V_LOADS_PER_CHUNK =
          V_WARP_TOKENS_PER_CHUNK / V_TOKENS_PER_LOAD;
      constexpr int V_ELEMS_PER_LOAD = 16 / static_cast<int>(sizeof(cache_t));
      static_assert(V_WARP_TOKENS_PER_CHUNK % 8 == 0,
                    "V_WARP_TOKENS_PER_CHUNK must be a multiple of 8 "
                    "(8 tokens per _B16x8 / _B8x8 slot)");
      constexpr int V_CHUNK_SLOTS = V_WARP_TOKENS_PER_CHUNK / 8;

      const cache_t* v_ptr_head = v_cache + wg_start_kv_head_idx * HEAD_SIZE;

      for (int b = 0; b < VBLOCKS; b++) {
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const cache_t* v_block_ptr =
            v_ptr_head + vphysical_block_number * kv_block_stride;

        for (int sb = 0; sb < V_SUB_BATCHES_PER_BLOCK; sb++) {
          const int token_base_in_block = sb * V_WARP_TOKENS_PER_CHUNK;

          // Step 1: Cooperative coalesced load of V_WARP_TOKENS_PER_CHUNK
          // tokens from the current block into v_warp_lds[warpid][0..CHUNK-1].
          // Lane layout: (laneid / V_LANES_PER_TOKEN) selects a token within
          // the chunk, (laneid % V_LANES_PER_TOKEN) selects a 16B chunk of
          // head_dim. Each token's head_dim row is contiguous in flash
          // layout, so every 16B lane chunk hits the same global cache line
          // as its neighbors.
          #pragma unroll
          for (int load = 0; load < V_LOADS_PER_CHUNK; load++) {
            const int tok_in_chunk =
                load * V_TOKENS_PER_LOAD + laneid / V_LANES_PER_TOKEN;
            const int tok_in_block = token_base_in_block + tok_in_chunk;
            const int dim_elem =
                (laneid % V_LANES_PER_TOKEN) * V_ELEMS_PER_LOAD;
            *reinterpret_cast<uint4*>(
                &v_warp_lds[warpid][tok_in_chunk][dim_elem]) =
                *reinterpret_cast<const uint4*>(
                    &v_block_ptr[tok_in_block * kv_head_stride + dim_elem]);
          }
          // No __syncthreads() — wavefront lockstep guarantees that the
          // stores above are visible to all 64 lanes of this warp before
          // the reads below execute.

          // Step 2: Transposed LDS read → Vlocal / Vlocalb8.
          // Each lane owns one head_dim element (head_size_elem) and gathers
          // V_WARP_TOKENS_PER_CHUNK token values for it, producing the
          // token-ordered layout the SV MFMA expects. Destination slot:
          //   Vlocal[h][b * BLOCK_SIZE/8 + sb * V_CHUNK_SLOTS + d]
          // which for BLOCK_SIZE=16 (SB=1) matches the original b*2+d
          // formula exactly, and for BLOCK_SIZE=32 (SB=2) fills the 4
          // per-block slots across two sub-batches.
          if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
            for (int h = 0; h < VHELOOP; h++) {
              const int head_size_elem = h * WARP_SIZE + laneid;
              for (int d = 0; d < V_CHUNK_SLOTS; d++) {
                scalar_t temp_v[8];
                for (int t = 0; t < 8; t++) {
                  temp_v[t] = *reinterpret_cast<const scalar_t*>(
                      &v_warp_lds[warpid][d * 8 + t][head_size_elem]);
                }
                const int vlocal_slot =
                    b * (BLOCK_SIZE / 8) + sb * V_CHUNK_SLOTS + d;
                Vlocal[h][vlocal_slot] =
                    *reinterpret_cast<const _B16x8*>(temp_v);
              }
            }
          } else {
            for (int h = 0; h < VHELOOP; h++) {
              const int head_size_elem = h * WARP_SIZE + laneid;
              for (int d = 0; d < V_CHUNK_SLOTS; d++) {
                cache_t temp_v[8];
                for (int t = 0; t < 8; t++) {
                  temp_v[t] = v_warp_lds[warpid][d * 8 + t][head_size_elem];
                }
                const int vlocal_slot =
                    b * (BLOCK_SIZE / 8) + sb * V_CHUNK_SLOTS + d;
                Vlocalb8[h][vlocal_slot] =
                    *reinterpret_cast<const _B8x8*>(temp_v);
              }
            }
          }
          // No trailing barrier — next sub-batch's cooperative store
          // happens in the same warp and lockstep ordering keeps it
          // ordered after the reads above. vout_shared (union alias) is
          // not written until Phase 6, well after the __syncthreads() at
          // the end of this warp-in-context branch.
        }
      }
    } else {
      // Paged/Shuffle V — branch on layout within the KV_DTYPE split
      const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
      if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
        const _B16x8* v_ptrh8 = reinterpret_cast<const _B16x8*>(v_ptr);
        for (int b = 0; b < VBLOCKS; b++) {
          const int64_t vphysical_block_number =
              static_cast<int64_t>(vphysical_blocks[b]);
          const _B16x8* v_ptrh8b =
              v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
          for (int h = 0; h < VHELOOP; h++) {
            const int head_size_elem = h * WARP_SIZE + laneid;
            if constexpr (KV_CACHE_LAYOUT == 2) {
              // Shuffle V: [num_blocks, num_heads, block_size/x, head_dim, x]
              for (int bo_chunk = 0; bo_chunk < BLOCK_SIZE / KX_V;
                   bo_chunk++) {
                const _B16x8* v_chunk =
                    v_ptrh8b +
                    (bo_chunk * HEAD_SIZE + head_size_elem) * KX_V / 8;
                for (int xd = 0; xd < KX_V / 8; xd++) {
                  Vlocal[h][b * BLOCK_SIZE / 8 + bo_chunk * (KX_V / 8) + xd] =
                      v_chunk[xd];
                }
              }
            } else {
              // Paged V: [num_blocks, num_heads, head_dim, block_size]
              const _B16x8* v_ptrh8be =
                  v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
              for (int d = 0; d < BLOCK_SIZE / 8; d++) {
                Vlocal[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
              }
            }
          }
        }
      } else {
        const _B8x8* v_ptrh8 = reinterpret_cast<const _B8x8*>(v_ptr);
        for (int b = 0; b < VBLOCKS; b++) {
          const int64_t vphysical_block_number =
              static_cast<int64_t>(vphysical_blocks[b]);
          const _B8x8* v_ptrh8b =
              v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
          for (int h = 0; h < VHELOOP; h++) {
            const int head_size_elem = h * WARP_SIZE + laneid;
            if constexpr (KV_CACHE_LAYOUT == 2) {
              // Shuffle V fp8: same interleaved addressing with _B8x8 loads
              for (int bo_chunk = 0; bo_chunk < BLOCK_SIZE / KX_V;
                   bo_chunk++) {
                const _B8x8* v_chunk =
                    v_ptrh8b +
                    (bo_chunk * HEAD_SIZE + head_size_elem) * KX_V / 8;
                for (int xd = 0; xd < KX_V / 8; xd++) {
                  Vlocalb8[h][b * BLOCK_SIZE / 8 + bo_chunk * (KX_V / 8) +
                              xd] = v_chunk[xd];
                }
              }
            } else {
              // Paged V fp8
              const _B8x8* v_ptrh8be =
                  v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
              for (int d = 0; d < BLOCK_SIZE / 8; d++) {
                Vlocalb8[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
              }
            }
          }
        }
      }
    }

  #define QK_mfma(x)                                             \
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) { \
      Klocal[x] = convert_b8x8_custom<scalar_t>(Klocalb8[x]);    \
    }                                                            \
    for (int h = 0; h < QHLOOP; h++) {                           \
      d_out[h] = gcn_mfma4x4x4_instr<scalar_t, 4, x, 0>(         \
          Qlocal[h].xy[0], Klocal[x].xy[0], d_out[h]);           \
      d_out[h] = gcn_mfma4x4x4_instr<scalar_t, 4, x, 0>(         \
          Qlocal[h].xy[1], Klocal[x].xy[1], d_out[h]);           \
    }
    // QK mfma with Q mfma block broadcast
    // Q values across head_size dimension stored across lanes
    // K values across head_size dimension are stored depthwise within lane
    // Q broadcast with absz, cbid of mfma instruction
    QK_mfma(0);
    QK_mfma(1);
    QK_mfma(2);
    QK_mfma(3);
    QK_mfma(4);
    QK_mfma(5);
    QK_mfma(6);
    QK_mfma(7);
    // below only needed for head size 128
    if constexpr (KHELOOP > 8) {
      QK_mfma(8);
      QK_mfma(9);
      QK_mfma(10);
      QK_mfma(11);
      QK_mfma(12);
      QK_mfma(13);
      QK_mfma(14);
      QK_mfma(15);
    }
  #undef QK_mfma

    float scale2 = scale;
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
      // post mfma scaling for fp8
      scale2 *= *k_scale;
    }

    for (int h = 0; h < QHLOOP; h++) {
      d_out[h] *= scale2;
    }

    // transpose d_out so that 4 token ids are in each lane, and 4 heads are
    // across 4 lanes
    for (int h = 0; h < QHLOOP; h++) {
      floatx4 tmp = {0};
      for (int i = 0; i < 4; i++) {
        const float B = (lane4id == i) ? 1.0f : 0.0f;
        tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(d_out[h][i], B, tmp, 0, 0, 0);
      }
      d_out[h] = tmp;
    }

    const int lane4_token_idx = 4 * (global_token_idx >> 2);

    if constexpr (ALIBI_ENABLED) {
      const int alibi_offset = lane4_token_idx - seq_len + 1;
      for (int h = 0; h < QHLOOP; h++) {
        for (int i = 0; i < 4; i++) {
          d_out[h][i] += alibi_slope[h] * (alibi_offset + i);
        }
      }
    }

    const int bpermute_mask = 4 * (16 * ((laneid >> 2) % 4) + lane4id);

    for (int h = 0; h < QHLOOP; h++) {
      qk_max[h] = -FLT_MAX;
      for (int i = 0; i < 4; i++) {
        qk_max[h] = (lane4_token_idx + i < seq_len)
                        ? fmaxf(qk_max[h], d_out[h][i])
                        : qk_max[h];
      }

      // for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
      //   qk_max[h] = fmaxf(qk_max[h], __shfl_xor(qk_max[h], mask));
      // }
      // faster version of above code with dpp
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));

      auto tmp = __builtin_amdgcn_ds_bpermute(
          bpermute_mask, *reinterpret_cast<int*>(&qk_max[h]));
      qk_max[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
    }

    float exp_sum[QHLOOP];
    for (int h = 0; h < QHLOOP; h++) {
      exp_sum[h] = 0.0f;
      for (int i = 0; i < 4; i++) {
        d_out[h][i] = (lane4_token_idx + i < seq_len)
                          ? __expf(d_out[h][i] - qk_max[h])
                          : 0.0f;
        exp_sum[h] += d_out[h][i];
      }
      // for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
      //   exp_sum[h] += __shfl_xor(exp_sum[h], mask);
      // }
      // faster version of above code with dpp
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));

      auto tmp = __builtin_amdgcn_ds_bpermute(
          bpermute_mask, *reinterpret_cast<int*>(&exp_sum[h]));
      exp_sum[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
    }

    if (laneid < 4) {
      for (int h = 0; h < QHLOOP; h++) {
        const int head_idx = 4 * h + lane4id;
        shared_qk_max[warpid][head_idx] = qk_max[h];
        shared_exp_sum[warpid][head_idx] = exp_sum[h];
      }
    }
  }  // warp within context

  __syncthreads();

  const auto num_heads = gridDim.z * GQA_RATIO;
  float* max_logits_ptr =
      max_logits + seq_idx * num_heads * max_num_partitions + partition_idx;
  float* exp_sums_ptr =
      exp_sums + seq_idx * num_heads * max_num_partitions + partition_idx;
  // calculate qk_max and exp_sums for partition
  for (int h = 0; h < QHLOOP; h++) {
    float global_qk_max = -FLT_MAX;
    float warp_qk_max[NWARPS];
    const int head_idx = 4 * h + lane4id;
    for (int w = 0; w < NWARPS; w++) {
      warp_qk_max[w] = shared_qk_max[w][head_idx];
      global_qk_max = fmaxf(global_qk_max, warp_qk_max[w]);
    }
    float global_exp_sum = 0.0f;
    for (int w = 0; w < NWARPS; w++) {
      global_exp_sum +=
          shared_exp_sum[w][head_idx] * __expf(warp_qk_max[w] - global_qk_max);
    }
    if (head_idx < GQA_RATIO) {
      max_logits_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_qk_max;
      exp_sums_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_exp_sum;
    }
    const float global_inv_sum_scale = __fdividef(1.f, global_exp_sum + 1e-6f) *
                                       __expf(qk_max[h] - global_qk_max);
    d_out[h] *= global_inv_sum_scale;
  }
  constexpr bool LOGITS_RTZ_CONVERSION = false;
  // logits[h] -> every 4 lanes hold 4 heads, each lane holds 4 tokens, there
  // are 4x16 tokens across warp
  _B16x4 logits[QHLOOP];
  for (int h = 0; h < QHLOOP; h++) {
    if constexpr (LOGITS_RTZ_CONVERSION) {
      // use rtz for faster performance with no perceivable accuracy loss
      logits[h] = from_floatx4_rtz<scalar_t>(d_out[h]);
    } else {
      logits[h] = from_floatx4<scalar_t>(d_out[h]);
    }
  }

  if (warp_start_token_idx >= seq_len) {  // warp out of context
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout_shared[qh][vh][laneid][warpid] = {0};
      }
    }
  } else {  // warp in context
  #define SV_mfma(x)                                                  \
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {      \
      Vlocal[vh][x] = convert_b8x8_custom<scalar_t>(Vlocalb8[vh][x]); \
    }                                                                 \
    for (int qh = 0; qh < QHLOOP; qh++) {                             \
      acc[qh] = gcn_mfma4x4x4_instr<scalar_t, 4, 2 * x, 0>(           \
          logits[qh], Vlocal[vh][x].xy[0], acc[qh]);                  \
      acc[qh] = gcn_mfma4x4x4_instr<scalar_t, 4, 2 * x + 1, 0>(       \
          logits[qh], Vlocal[vh][x].xy[1], acc[qh]);                  \
    }

    for (int vh = 0; vh < VHELOOP; vh++) {
      floatx4 acc[QHLOOP];
      for (int qh = 0; qh < QHLOOP; qh++) {
        acc[qh] = {0};
      }
      // SoftMax-V calculation
      // logits -> token dimension is distributed across lanes
      // Vlocal -> token dimension is depthwise within lane
      // uses mfma instruction block broadcast for logits
      SV_mfma(0);
      SV_mfma(1);
      SV_mfma(2);
      SV_mfma(3);
      SV_mfma(4);
      SV_mfma(5);
      SV_mfma(6);
      SV_mfma(7);

      for (int qh = 0; qh < QHLOOP; qh++) {
        if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
          // post mfma v scale for fp8
          acc[qh] *= *v_scale;
        }
        vout_shared[qh][vh][laneid][warpid] = from_floatx4<scalar_t>(acc[qh]);
      }
    }

  #undef SV_mfma
  }  // warp in context

  __syncthreads();

  // final write to tmp_out after vout accumulation
  if (warpid == 0) {
    _B16x4 vout[QHLOOP][VHELOOP];
    // iterate across heads
    for (int qh = 0; qh < QHLOOP; qh++) {
      // iterate over each v head elem (within head_size)
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout[qh][vh] = {0};
        for (int w = 0; w < NWARPS; w++) {
          vout[qh][vh] =
              addx4<scalar_t>(vout[qh][vh], vout_shared[qh][vh][laneid][w]);
        }
      }
    }

    scalar_t* out_ptr = out +
                        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                        partition_idx * HEAD_SIZE;
    const int out_num_partitions = max_num_partitions;
    bit16_t* out_ptr_b16 = reinterpret_cast<bit16_t*>(out_ptr);
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        const int head_size_elem = vh * WARP_SIZE + laneid;
        for (int i = 0; i < 4; i++) {
          const int head_idx = 4 * qh + i;
          if (head_idx < GQA_RATIO) {
            out_ptr_b16[(wg_start_head_idx + head_idx) * out_num_partitions *
                            HEAD_SIZE +
                        head_size_elem] = vout[qh][vh][i];
          }
        }
      }
    }
  }  // warpid == 0
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  const auto num_heads = gridDim.x;
  const auto head_idx = blockIdx.x;
  const auto seq_idx = blockIdx.y;

  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1)) {
    return;
  }

  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  const auto warpid = threadIdx.x / WARP_SIZE;

  __shared__ float shared_global_exp_sum;
  // max num partitions supported is warp_size * NPAR_LOOPS
  __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];

  if (warpid == 0) {
    const float* max_logits_ptr = max_logits +
                                  seq_idx * num_heads * max_num_partitions +
                                  head_idx * max_num_partitions;

    // valid partition is the last valid partition in case threadid > num
    // partitions
    int valid_partition[NPAR_LOOPS];
    float reg_max_logit[NPAR_LOOPS];
    const int last_valid_partition = num_partitions - 1;

  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const auto partition_no = i * WARP_SIZE + threadIdx.x;
      valid_partition[i] =
          (partition_no < num_partitions) ? partition_no : last_valid_partition;
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      reg_max_logit[i] = max_logits_ptr[valid_partition[i]];
    }
    float max_logit = reg_max_logit[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      max_logit = fmaxf(max_logit, reg_max_logit[i]);
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
    }

    const float* exp_sums_ptr = exp_sums +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;

    float rescaled_exp_sum[NPAR_LOOPS];
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const auto partition_no = i * WARP_SIZE + threadIdx.x;
      rescaled_exp_sum[i] *= (partition_no < num_partitions)
                                 ? expf(reg_max_logit[i] - max_logit)
                                 : 0.0f;
    }
    float global_exp_sum = rescaled_exp_sum[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      global_exp_sum += rescaled_exp_sum[i];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const auto partition_no = i * WARP_SIZE + threadIdx.x;
      shared_exp_sums[partition_no] = rescaled_exp_sum[i];
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      global_exp_sum += __shfl_xor(global_exp_sum, mask);
    }
    if (threadIdx.x == 0) {
      shared_global_exp_sum = global_exp_sum;
    }
  }  // warpid == 0
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
  constexpr int MAX_NPAR = 64;
  scalar_t tmps[MAX_NPAR];
  const float dzero = 0.0f;
  #pragma unroll
  for (int j = 0; j < MAX_NPAR; j++) {
    tmps[j] = from_float<scalar_t>(dzero);
  }
  const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
  const int num_partition_offset = (num_partitions)*HEAD_SIZE;
  int idx = 0;

  constexpr int JCHUNK = 16;

  #pragma unroll
  for (int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE) {
    // lastj is last valid partition
    const int lastj_offset =
        (j < num_partition_offset) ? j : last_partition_offset;
    tmps[idx] = tmp_out_ptr[lastj_offset];
    idx++;
  }
  __syncthreads();

  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE;
         j += HEAD_SIZE) {
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }
    }
  }  // num_partitions > JCHUNK

  // Aggregate tmp_out to out.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < JCHUNK; j++) {
    acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
  }
  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK; j < 2 * JCHUNK; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
      }
    }
  }

  for (int p = 1; p < NPAR_LOOPS; p++) {
    if (num_partitions > p * MAX_NPAR) {
      idx = 0;
  #pragma unroll
      for (int j = p * MAX_NPAR * HEAD_SIZE; j < (p + 1) * MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        // lastj is last valid partition
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }

  #pragma unroll
      for (int j = 0; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + p * MAX_NPAR];
      }
    }
  }

  const float inv_global_exp_sum =
      __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
  const float out_scale =
      (fp8_out_scale_ptr != nullptr) ? 1.0f / (*fp8_out_scale_ptr) : 1.0f;
  acc *= inv_global_exp_sum;
  acc *= out_scale;
  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
  OUTT* out_ptr = out + query_start_off * num_heads * HEAD_SIZE +
                  static_cast<int64_t>(head_idx) * HEAD_SIZE;
  if constexpr (std::is_same<OUTT, bit8_t>::value) {
    out_ptr[threadIdx.x] =
        __hip_cvt_float_to_fp8(acc, vllm::fp8::fp8_type::__default_saturation,
                               vllm::fp8::fp8_type::__default_interpret);
  } else {
    out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
  }
}

#elif defined(__HIP__GFX11__)

using floatx8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;

using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
union b16x8_u {
  bit16x8 u16x8;
  _B16x4 xy[2];
};
typedef b16x8_u _B16x8;

using bit16x16 =
    __attribute__((__vector_size__(16 * sizeof(uint16_t)))) uint16_t;
union b16x16_u {
  bit16x16 u16x16;
  _B16x8 xy[2];
};
typedef b16x16_u _B16x16;

using _B8x8 = uint2;
using bit8_t = uint8_t;

typedef struct _B8x16 {
  _B8x8 xy[2];
} _B8x16;

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx8 gcn_wmma16x16x16_instr(const bit16x16& inpA,
                                                          const bit16x16& inpB,
                                                          const floatx8& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(inpA, inpB, inpC);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(inpA, inpB, inpC);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (_Float16)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __float2bfloat16(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x8 from_floatx8(const floatx8& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      __half2 h2[4];
      _B16x8 b16x8;
    } u;
    u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
    u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
    u.h2[2] = __float22half2_rn(make_float2(inp[4], inp[5]));
    u.h2[3] = __float22half2_rn(make_float2(inp[6], inp[7]));
    return u.b16x8;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    union b2cvt {
      __hip_bfloat162 b2[4];
      _B16x8 b16x8;
    } u;

    u.b2[0] = __float22bfloat162_rn(make_float2(inp[0], inp[1]));
    u.b2[1] = __float22bfloat162_rn(make_float2(inp[2], inp[3]));
    u.b2[2] = __float22bfloat162_rn(make_float2(inp[4], inp[5]));
    u.b2[3] = __float22bfloat162_rn(make_float2(inp[6], inp[7]));

    return u.b16x8;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO,
          MFMAType MFMA_TYPE,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS, 3) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,  // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  // clang-format on
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;  // 8 warps on gfx11
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;
  const int lane2id = laneid % 2;
  const int lane16id = laneid % 16;
  const int rowid = laneid / 16;

  const int seq_idx = blockIdx.x;
  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx]) != 1) {
    return;
  }

  const int partition_idx = blockIdx.y;

  constexpr int T_PAR_SIZE = 256;  // token partition size set to 256

  const int max_num_partitions = gridDim.y;

  const int seq_len = seq_lens[seq_idx];  // length of a seq

  const int partition_start_token_idx = partition_idx * T_PAR_SIZE;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= seq_len) {
    return;
  }

  constexpr int GQA_RATIO2 = DIVIDE_ROUND_UP(GQA_RATIO, 2);

  __shared__ float shared_qk_max[NWARPS][16 + 1];
  __shared__ float shared_exp_sum[NWARPS][16 + 1];
  // shared_logits is used for multiple purposes
  __shared__ _B16x16 shared_logits[NWARPS][2][16][2];

  // for QK wmma16x16, layout is QHead/Tokenx16 across every 16 lanes,
  // 32 Bytes HeadElements in each lane, 2x16B HeadElements across a row of warp
  constexpr int ROWS_PER_WARP =
      WARP_SIZE / 16 / 2;  // rows refers to 16 lanes; refer dpp terminology
  constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD =
      16 / sizeof(cache_t);  // 8 for 16 bit cache type, 16 for 8 bit types
  constexpr int QKHE_PER_FETCH =
      CONTIGUOUS_KV_ELEMS_16B_LOAD *
      ROWS_PER_WARP;  // each fetch across a warp fetches these many elements
  constexpr int QKHELOOP = HEAD_SIZE / QKHE_PER_FETCH;  // 2xQKHE_16B across
                                                        // warp

  _B16x16 Qlocal[QKHELOOP / 2];  // note that 16 contiguous elements of Q should
                                 // be fetched per lane for 16 bit cache types

  constexpr int CONTIGUOUS_SCALAR_ELEMS_16B = 16 / sizeof(scalar_t);

  constexpr int TOKENS_PER_WARP =
      T_PAR_SIZE /
      NWARPS;  // sub partition of tokens per warp for qk calculation
  constexpr int TLOOP =
      TOKENS_PER_WARP /
      16;  // each wmma16x16x16 instruction processes 16 tokens

  _B16x16 Klocal[TLOOP]
                [QKHELOOP / 2];  // can be interpreted as B8x16 for 8 bit types

  const int wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const int wg_start_kv_head_idx = blockIdx.z;
  const int total_num_heads = gridDim.z * GQA_RATIO;

  // for QK wmma, tokens in multiples of TOKENS_PER_WARP are spread across warps
  // each wmma takes QH16xT16x16HE across warp
  // repeat wmma across QKHELOOP dimension
  // output layout from QKwmma : QH16xT8x2 16 qheads across 16 lanes, 16 tokens
  // across 2 rows x 8 tokens per lane

  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);

  if (GQA_RATIO == 1) {
    const int local_qhead_idx = lane16id % GQA_RATIO;
    const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
    const scalar_t* q_ptr =
        q + query_start_off * q_stride + global_qhead_idx * HEAD_SIZE;
    if (lane16id < GQA_RATIO) {
  #pragma unroll
      for (int qkhe_depth = 0; qkhe_depth < QKHELOOP / 2; qkhe_depth++) {
        const scalar_t* q_fetch_ptr = q_ptr + qkhe_depth * QKHE_PER_FETCH * 2;
        const _B16x16* q_fetch_ptr_32B =
            reinterpret_cast<const _B16x16*>(q_fetch_ptr);
        Qlocal[qkhe_depth] = *q_fetch_ptr_32B;
      }
    }
  } else {
    // fetch Q in shared across warps and then write to registers
    const int local_qhead_idx = 2 * warpid + rowid;
    const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
    const scalar_t* q_ptr =
        q + query_start_off * q_stride + global_qhead_idx * HEAD_SIZE;

    const int qhead_element = lane16id * CONTIGUOUS_SCALAR_ELEMS_16B;
    if ((local_qhead_idx < GQA_RATIO) && (qhead_element < HEAD_SIZE)) {
      const scalar_t* q_fetch_ptr = q_ptr + qhead_element;
      const _B16x8* q_fetch_ptr_16B =
          reinterpret_cast<const _B16x8*>(q_fetch_ptr);
      _B16x8 tmp = *q_fetch_ptr_16B;

      const int offset1 =
          lane16id /
          2;  // 16 contiguous chunks of head elems are spread across 8x2lanes
      shared_logits[offset1][lane2id][local_qhead_idx][0].xy[0] = tmp;
    }

    __syncthreads();

  #pragma unroll
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP / 2; qkhe_depth++) {
      Qlocal[qkhe_depth].xy[0] =
          shared_logits[qkhe_depth][0][lane16id % GQA_RATIO][0].xy[0];
      Qlocal[qkhe_depth].xy[1] =
          shared_logits[qkhe_depth][1][lane16id % GQA_RATIO][0].xy[0];
    }
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int last_seq_block = num_seq_blocks - 1;

  const int* block_table_seq = block_tables + seq_idx * max_num_blocks_per_seq;

  int kphysical_block_number[TLOOP];

  // fetch k physical block numbers
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int klocal_token_idx =
        TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
    const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
    const int kblock_idx = (kglobal_token_idx < seq_len)
                               ? kglobal_token_idx / BLOCK_SIZE
                               : last_seq_block;
    kphysical_block_number[token_depth] = block_table_seq[kblock_idx];
  }

  constexpr int KX = 16 / sizeof(cache_t);
  const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * kv_head_stride;

  const int row_head_elem = 0;

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int64_t kblock_number =
        static_cast<int64_t>(kphysical_block_number[token_depth]);
    const cache_t* k_ptr2 = k_ptr + kblock_number * kv_block_stride;
    const int klocal_token_idx =
        TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
    const int kphysical_block_offset = klocal_token_idx % BLOCK_SIZE;
    const cache_t* k_ptr3 = k_ptr2 + kphysical_block_offset * KX;

    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
      const int head_elem = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
      const int offset1 = head_elem / KX;
      const int offset2 = head_elem % KX;
      const cache_t* k_fetch_ptr = k_ptr3 + offset1 * BLOCK_SIZE * KX + offset2;
      const _B16x8* k_fetch_ptr_16B =
          reinterpret_cast<const _B16x8*>(k_fetch_ptr);
      Klocal[token_depth][qkhe_depth / 2].xy[qkhe_depth % 2] = *k_fetch_ptr_16B;
    }
  }

  constexpr int VTOKENS_PER_LANE =
      TOKENS_PER_WARP / ROWS_PER_WARP;  // 32/1 = 32 vtokens per lane
  constexpr int VBLOCKS_PER_LANE = 2;   // assumes block size >=16
  constexpr int VTLOOP = NWARPS;        // corresponds to tokens across warps
  constexpr int VTLANELOOP = DIVIDE_ROUND_UP(
      VTOKENS_PER_LANE,
      CONTIGUOUS_KV_ELEMS_16B_LOAD);  // optimized for 16B fetches; assumes
                                      // minimum block size is 16
  constexpr int VHELOOP = DIVIDE_ROUND_UP(
      (HEAD_SIZE / 16), NWARPS);  // head_size distributed across warps; each
                                  // wmma instr works on 16 head elements

  int vphysical_block_number[VTLOOP][VBLOCKS_PER_LANE];

  // fetch v physical block numbers
  for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
    for (int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE;
         vblock_depth++) {
      const int vlocal_token_idx =
          vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP +
          vblock_depth * BLOCK_SIZE;
      const int vglobal_token_idx =
          partition_start_token_idx + vlocal_token_idx;
      const int vblock_idx = (vglobal_token_idx < seq_len)
                                 ? vglobal_token_idx / BLOCK_SIZE
                                 : last_seq_block;
      vphysical_block_number[vtoken_depth][vblock_depth] =
          block_table_seq[vblock_idx];
    }
  }

  _B16x16 Vlocal[VTLOOP][VHELOOP]
                [VTLANELOOP / 2];  // this can be interpreted as B8x16 too

  const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
  // v fetches are 16head elems across lanes x (16x2) tokens per lane
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
    const cache_t* v_ptr2 = v_ptr + vhead_elem * BLOCK_SIZE;

    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
      for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
        const int64_t vblock_number = static_cast<int64_t>(
            vphysical_block_number[vtoken_depth]
                                  [vfetch_depth / VBLOCKS_PER_LANE]);
        const cache_t* v_ptr3 = v_ptr2 + (vblock_number * kv_block_stride);

        const cache_t* v_fetch_ptr =
            v_ptr3 +
            (vfetch_depth % VBLOCKS_PER_LANE) * CONTIGUOUS_KV_ELEMS_16B_LOAD;
        const _B16x8* v_fetch_ptr_16B =
            reinterpret_cast<const _B16x8*>(v_fetch_ptr);
        Vlocal[vtoken_depth][vhe_depth][vfetch_depth / 2].xy[vfetch_depth % 2] =
            *v_fetch_ptr_16B;
      }
    }
  }

  floatx8 dout[TLOOP];
  // qk wmma
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    dout[token_depth] = {0};
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP / 2; qkhe_depth++) {
      dout[token_depth] = gcn_wmma16x16x16_instr<scalar_t, 0, 0, 0>(
          Klocal[token_depth][qkhe_depth].u16x16, Qlocal[qkhe_depth].u16x16,
          dout[token_depth]);
    }
    dout[token_depth] *= scale;
  }

  // calculate qk_max and exp_sum per warp and write to shared memory
  float qk_max = -FLT_MAX;
  float exp_sum = 0.0f;
  const int qkout_token_idx =
      partition_start_token_idx + TOKENS_PER_WARP * warpid + rowid;
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 8; i++) {
      const float tmp =
          (local_token_idx + 2 * i < seq_len) ? dout[token_depth][i] : -FLT_MAX;
      qk_max = fmaxf(qk_max, tmp);
    }
  }

  qk_max = fmaxf(qk_max, __shfl_xor(qk_max, 16));

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 8; i++) {
      const float tmp = (local_token_idx + 2 * i < seq_len)
                            ? __expf(dout[token_depth][i] - qk_max)
                            : 0.0f;
      dout[token_depth][i] = tmp;
      exp_sum += tmp;
    }
  }

  exp_sum += __shfl_xor(exp_sum, 16);

  __syncthreads();

  if (laneid < 16) {
    shared_qk_max[warpid][lane16id] = qk_max;
    shared_exp_sum[warpid][lane16id] = exp_sum;
  }

  __syncthreads();

  // calculate partition qk_max and exp_sum
  float partition_qk_max = -FLT_MAX;
  float warp_qk_max_exp[NWARPS];
  float partition_exp_sum = 0.0f;

  #pragma unroll
  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = shared_qk_max[w][lane16id];
    partition_qk_max = fmaxf(partition_qk_max, warp_qk_max_exp[w]);
  }

  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = __expf(warp_qk_max_exp[w] - partition_qk_max);
    partition_exp_sum += shared_exp_sum[w][lane16id] * warp_qk_max_exp[w];
  }

  const float inv_sum_scale =
      __fdividef(1.f, partition_exp_sum + 1e-6f) * warp_qk_max_exp[warpid];

  __syncthreads();

  // write logits to shared mem
  #pragma unroll
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    dout[token_depth] *= inv_sum_scale;
    shared_logits[warpid][token_depth][lane16id][0].xy[rowid] =
        from_floatx8<scalar_t>(dout[token_depth]);
  }
  __syncthreads();

  _B16x8 swp_buf[TLOOP][2];
  #pragma unroll
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    swp_buf[token_depth][0] =
        shared_logits[warpid][token_depth][lane16id][0].xy[0];
    swp_buf[token_depth][1] =
        shared_logits[warpid][token_depth][lane16id][0].xy[1];
  }

  #pragma unroll
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      shared_logits[warpid][token_depth][lane16id][0].xy[rowid].u16x8[i] =
          swp_buf[token_depth][i % 2].u16x8[4 * rowid + (i / 2)];
    }
  }

  // write out partition max_logits and exp_sum
  if (threadIdx.x < GQA_RATIO) {
    const int qhead_idx = lane16id;
    const int offset = seq_idx * total_num_heads * max_num_partitions +
                       (wg_start_head_idx + qhead_idx) * max_num_partitions +
                       partition_idx;
    max_logits[offset] = partition_qk_max;
    exp_sums[offset] = partition_exp_sum;
  }

  __syncthreads();

  _B16x8 outelems[VHELOOP];
  // Softmax V wmma
  // v layout: 16he across lanes x (16x2) tokens per lane
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    floatx8 tmp_out = {0};
    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
      for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP / 2;
           vfetch_depth++) {
        const int offset = vfetch_depth;
        // if output format is 16 qheads across 16 lanes, 16 head elems spread
        // across rows
        tmp_out = gcn_wmma16x16x16_instr<scalar_t, 0, 0, 0>(
            Vlocal[vtoken_depth][vhe_depth][vfetch_depth].u16x16,
            shared_logits[vtoken_depth][offset][lane16id][0].u16x16, tmp_out);
      }
    }
    outelems[vhe_depth] = from_floatx8<scalar_t>(tmp_out);
  }

  __syncthreads();

  #pragma unroll
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    shared_logits[warpid][vhe_depth][lane16id][0].xy[rowid] =
        outelems[vhe_depth];  // lane16 id head dimension; rowid head element
                              // dimension
  }

  __syncthreads();

  #pragma unroll
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    swp_buf[vhe_depth][0] = shared_logits[warpid][vhe_depth][lane16id][0].xy[0];
    swp_buf[vhe_depth][1] = shared_logits[warpid][vhe_depth][lane16id][0].xy[1];
  }

  #pragma unroll
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      shared_logits[warpid][vhe_depth][lane16id][0].xy[rowid].u16x8[i] =
          swp_buf[vhe_depth][i % 2].u16x8[4 * rowid + (i / 2)];
    }
  }

  __syncthreads();

  // write to tmp_out with coalesced writes after reading from shared mem
  if (warpid == 0) {
    _B16x8 vout[GQA_RATIO2];
    // each lane writes out 16Bytes of tmp_out along head elem dimension
    const int head_elem_idx = lane16id * 8;
    if (head_elem_idx < HEAD_SIZE) {
      for (int h = 0; h < GQA_RATIO2; h++) {
        const int local_head_idx = 2 * h + rowid;
        const int offset1 = (head_elem_idx / 16) % NWARPS;
        const int offset2 = head_elem_idx / 16 / NWARPS;
        const int offset3 = (head_elem_idx / 8) % 2;  // num_he % num_row
        vout[h] =
            shared_logits[offset1][offset2][local_head_idx][0].xy[offset3];
      }

      const int hsz_maxp_mult = HEAD_SIZE * max_num_partitions;
      scalar_t* out_ptr = out + seq_idx * total_num_heads * hsz_maxp_mult +
                          partition_idx * HEAD_SIZE;
      for (int h = 0; h < GQA_RATIO2; h++) {
        const int local_head_idx = 2 * h + rowid;
        if (local_head_idx < GQA_RATIO) {
          const int out_head_idx = wg_start_head_idx + local_head_idx;
          scalar_t* out_ptr2 = out_ptr + out_head_idx * hsz_maxp_mult;
          scalar_t* out_ptr3 = out_ptr2 + head_elem_idx;
          _B16x8* out_ptr_B16x8 = reinterpret_cast<_B16x8*>(out_ptr3);
          *out_ptr_B16x8 = vout[h];
        }
      }
    }
  }
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma4_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  const auto num_heads = gridDim.x;
  const auto head_idx = blockIdx.x;
  const auto seq_idx = blockIdx.y;

  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1)) {
    return;
  }

  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  const int warpid = threadIdx.x / WARP_SIZE;

  __shared__ float shared_global_exp_sum;
  // max num partitions supported is warp_size * NPAR_LOOPS
  __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];

  if (warpid == 0) {
    const float* max_logits_ptr = max_logits +
                                  seq_idx * num_heads * max_num_partitions +
                                  head_idx * max_num_partitions;

    // valid partition is the last valid partition in case threadid > num
    // partitions
    int valid_partition[NPAR_LOOPS];
    float reg_max_logit[NPAR_LOOPS];
    const int last_valid_partition = num_partitions - 1;

  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      valid_partition[i] =
          (partition_no < num_partitions) ? partition_no : last_valid_partition;
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      reg_max_logit[i] = max_logits_ptr[valid_partition[i]];
    }
    float max_logit = reg_max_logit[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      max_logit = fmaxf(max_logit, reg_max_logit[i]);
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
    }

    const float* exp_sums_ptr = exp_sums +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;

    float rescaled_exp_sum[NPAR_LOOPS];
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      rescaled_exp_sum[i] *= (partition_no < num_partitions)
                                 ? expf(reg_max_logit[i] - max_logit)
                                 : 0.0f;
    }
    float global_exp_sum = rescaled_exp_sum[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      global_exp_sum += rescaled_exp_sum[i];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      shared_exp_sums[partition_no] = rescaled_exp_sum[i];
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      global_exp_sum += __shfl_xor(global_exp_sum, mask);
    }
    if (threadIdx.x == 0) {
      shared_global_exp_sum = global_exp_sum;
    }
  }  // warpid == 0
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
  constexpr int MAX_NPAR = 32;
  scalar_t tmps[MAX_NPAR];
  const float dzero = 0.0f;
  #pragma unroll
  for (int j = 0; j < MAX_NPAR; j++) {
    tmps[j] = from_float<scalar_t>(dzero);
  }
  const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
  const int num_partition_offset = (num_partitions)*HEAD_SIZE;
  int idx = 0;

  constexpr int JCHUNK = 16;

  #pragma unroll
  for (int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE) {
    // lastj is last valid partition
    const int lastj_offset =
        (j < num_partition_offset) ? j : last_partition_offset;
    tmps[idx] = tmp_out_ptr[lastj_offset];
    idx++;
  }
  __syncthreads();

  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE;
         j += HEAD_SIZE) {
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }
    }
  }  // num_partitions > JCHUNK

  // Aggregate tmp_out to out.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < JCHUNK; j++) {
    acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
  }
  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK; j < 2 * JCHUNK; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
      }
    }
  }

  for (int p = 1; p < NPAR_LOOPS; p++) {
    if (num_partitions > p * MAX_NPAR) {
      idx = 0;
  #pragma unroll
      for (int j = p * MAX_NPAR * HEAD_SIZE; j < (p + 1) * MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        // lastj is last valid partition
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }

  #pragma unroll
      for (int j = 0; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + p * MAX_NPAR];
      }
    }
  }

  const float inv_global_exp_sum =
      __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
  acc *= inv_global_exp_sum;

  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
  OUTT* out_ptr = out + query_start_off * num_heads * HEAD_SIZE +
                  static_cast<int64_t>(head_idx) * HEAD_SIZE;
  out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
}

#elif defined(__HIP__GFX12__)

using floatx8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;

using bit16x8 = __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
union b16x8_u {
  bit16x8 u16x8;
  _B16x4 xy[2];
};
typedef b16x8_u _B16x8;

using _B8x8 = uint2;
using bit8_t = uint8_t;

typedef struct _B8x16 {
  _B8x8 xy[2];
} _B8x16;

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx8 gcn_wmma16x16x16_instr(const bit16x8& inpA,
                                                          const bit16x8& inpB,
                                                          const floatx8& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(inpA, inpB, inpC);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(inpA, inpB, inpC);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float_b16(const bit16_t& inp) {
  union tmpcvt {
    bit16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t16;
  t16.u = inp;
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)t16.f;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(t16.b);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (_Float16)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __float2bfloat16(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x8 from_floatx8(const floatx8& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      __half2 h2[4];
      _B16x8 b16x8;
    } u;
    u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
    u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
    u.h2[2] = __float22half2_rn(make_float2(inp[4], inp[5]));
    u.h2[3] = __float22half2_rn(make_float2(inp[6], inp[7]));
    return u.b16x8;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    union b2cvt {
      __hip_bfloat162 b2[4];
      _B16x8 b16x8;
    } u;

    u.b2[0] = __float22bfloat162_rn(make_float2(inp[0], inp[1]));
    u.b2[1] = __float22bfloat162_rn(make_float2(inp[2], inp[3]));
    u.b2[2] = __float22bfloat162_rn(make_float2(inp[4], inp[5]));
    u.b2[3] = __float22bfloat162_rn(make_float2(inp[6], inp[7]));

    return u.b16x8;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO,
          MFMAType MFMA_TYPE,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS, 3) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,  // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  // clang-format on
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;  // 8 warps on gfx11
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;
  const int lane2id = laneid % 2;
  const int lane16id = laneid % 16;
  const int rowid = laneid / 16;

  const int seq_idx = blockIdx.x;
  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1)) {
    return;
  }
  const int partition_idx = blockIdx.y;

  constexpr int T_PAR_SIZE = 256;  // token partition size set to 256

  const int max_num_partitions = gridDim.y;

  const int seq_len = seq_lens[seq_idx];  // length of a seq

  const int partition_start_token_idx = partition_idx * T_PAR_SIZE;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= seq_len) {
    return;
  }

  constexpr int GQA_RATIO2 = DIVIDE_ROUND_UP(GQA_RATIO, 2);

  __shared__ float shared_qk_max[NWARPS][16 + 1];
  __shared__ float shared_exp_sum[NWARPS][16 + 1];
  // shared_logits is used for multiple purposes
  __shared__ _B16x8 shared_logits[NWARPS][2][16][2];

  // for QK wmma16x16_gfx12, layout is QHead/Tokenx16 across every 16 lanes,
  // 16 Bytes HeadElements in each lane, 2x16B HeadElements across 2 rows of
  // warp
  constexpr int ROWS_PER_WARP =
      WARP_SIZE / 16;  // rows refers to 16 lanes; refer dpp terminology
  constexpr int CONTIGUOUS_KV_ELEMS_16B_LOAD =
      16 / sizeof(cache_t);  // 8 for 16 bit cache type, 16 for 8 bit types
  constexpr int QKHE_PER_FETCH =
      CONTIGUOUS_KV_ELEMS_16B_LOAD *
      ROWS_PER_WARP;  // each fetch across a warp fetches these many elements
  constexpr int QKHELOOP = HEAD_SIZE / QKHE_PER_FETCH;  // 2xQKHE_16B across
                                                        // warp

  _B16x8 Qlocal[QKHELOOP];  // note that 16 contiguous elements of Q should
                            // be fetched per lane for 16 bit cache types

  constexpr int CONTIGUOUS_SCALAR_ELEMS_16B = 16 / sizeof(scalar_t);

  constexpr int TOKENS_PER_WARP =
      T_PAR_SIZE /
      NWARPS;  // sub partition of tokens per warp for qk calculation
  constexpr int TLOOP =
      TOKENS_PER_WARP /
      16;  // each wmma16x16x16 instruction processes 16 tokens

  _B16x8 Klocal[TLOOP]
               [QKHELOOP];  // can be interpreted as B8x16 for 8 bit types

  const int wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const int wg_start_kv_head_idx = blockIdx.z;
  const int total_num_heads = gridDim.z * GQA_RATIO;

  // for QK wmma, tokens in multiples of TOKENS_PER_WARP are spread across warps
  // each wmma takes QH16xT16x16HE across warp
  // repeat wmma across QKHELOOP dimension
  // output layout from QKwmma : QH16xT8x2 16 qheads across 16 lanes, 16 tokens
  // across 2 rows x 8 tokens per lane

  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);

  if (GQA_RATIO == 1) {
    const int local_qhead_idx = lane16id % GQA_RATIO;
    const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
    const scalar_t* q_ptr = q + query_start_off * q_stride +
                            global_qhead_idx * HEAD_SIZE +
                            rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;
    if (lane16id < GQA_RATIO) {
  #pragma unroll
      for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
        const scalar_t* q_fetch_ptr = q_ptr + qkhe_depth * QKHE_PER_FETCH;
        const _B16x8* q_fetch_ptr_16B =
            reinterpret_cast<const _B16x8*>(q_fetch_ptr);
        Qlocal[qkhe_depth] = *q_fetch_ptr_16B;
      }
    }
  } else {
    // fetch Q in shared across warps and then write to registers
    const int local_qhead_idx = 2 * warpid + rowid;
    const int global_qhead_idx = wg_start_head_idx + local_qhead_idx;
    const scalar_t* q_ptr =
        q + query_start_off * q_stride + global_qhead_idx * HEAD_SIZE;

    const int qhead_element = lane16id * CONTIGUOUS_SCALAR_ELEMS_16B;
    if ((local_qhead_idx < GQA_RATIO) && (qhead_element < HEAD_SIZE)) {
      const scalar_t* q_fetch_ptr = q_ptr + qhead_element;
      const _B16x8* q_fetch_ptr_16B =
          reinterpret_cast<const _B16x8*>(q_fetch_ptr);
      _B16x8 tmp = *q_fetch_ptr_16B;

      const int offset1 =
          lane16id /
          2;  // 16 contiguous chunks of head elems are spread across 8x2lanes
      shared_logits[offset1][lane2id][local_qhead_idx][0] = tmp;
    }

    __syncthreads();

  #pragma unroll
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
      Qlocal[qkhe_depth] =
          shared_logits[qkhe_depth][rowid][lane16id % GQA_RATIO][0];
    }
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int last_seq_block = num_seq_blocks - 1;

  const int* block_table_seq = block_tables + seq_idx * max_num_blocks_per_seq;

  int kphysical_block_number[TLOOP];

  // fetch k physical block numbers
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int klocal_token_idx =
        TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
    const int kglobal_token_idx = partition_start_token_idx + klocal_token_idx;
    const int kblock_idx = (kglobal_token_idx < seq_len)
                               ? kglobal_token_idx / BLOCK_SIZE
                               : last_seq_block;
    kphysical_block_number[token_depth] = block_table_seq[kblock_idx];
  }

  constexpr int KX = 16 / sizeof(cache_t);
  const cache_t* k_ptr = k_cache + wg_start_kv_head_idx * kv_head_stride;

  const int row_head_elem = rowid * CONTIGUOUS_KV_ELEMS_16B_LOAD;

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int64_t kblock_number =
        static_cast<int64_t>(kphysical_block_number[token_depth]);
    const cache_t* k_ptr2 = k_ptr + kblock_number * kv_block_stride;
    const int klocal_token_idx =
        TOKENS_PER_WARP * warpid + token_depth * 16 + lane16id;
    const int kphysical_block_offset = klocal_token_idx % BLOCK_SIZE;
    const cache_t* k_ptr3 = k_ptr2 + kphysical_block_offset * KX;

    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
      const int head_elem = row_head_elem + qkhe_depth * QKHE_PER_FETCH;
      const int offset1 = head_elem / KX;
      const int offset2 = head_elem % KX;
      const cache_t* k_fetch_ptr = k_ptr3 + offset1 * BLOCK_SIZE * KX + offset2;
      const _B16x8* k_fetch_ptr_16B =
          reinterpret_cast<const _B16x8*>(k_fetch_ptr);
      Klocal[token_depth][qkhe_depth] = *k_fetch_ptr_16B;
    }
  }

  constexpr int VTOKENS_PER_LANE =
      TOKENS_PER_WARP / ROWS_PER_WARP;  // 32/2 = 16 vtokens per lane
  constexpr int VBLOCKS_PER_LANE = 1;   // assumes block size >=16
  constexpr int VTLOOP = NWARPS;        // corresponds to tokens across warps
  constexpr int VTLANELOOP = DIVIDE_ROUND_UP(
      VTOKENS_PER_LANE,
      CONTIGUOUS_KV_ELEMS_16B_LOAD);  // optimized for 16B fetches; assumes
                                      // minimum block size is 16
  constexpr int VHELOOP = DIVIDE_ROUND_UP(
      (HEAD_SIZE / 16), NWARPS);  // head_size distributed across warps; each
                                  // wmma instr works on 16 head elements

  int vphysical_block_number[VTLOOP][VBLOCKS_PER_LANE];

  // fetch v physical block numbers
  for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
    for (int vblock_depth = 0; vblock_depth < VBLOCKS_PER_LANE;
         vblock_depth++) {
      const int vlocal_token_idx =
          vtoken_depth * VTOKENS_PER_LANE * ROWS_PER_WARP +
          rowid * VTOKENS_PER_LANE + vblock_depth * BLOCK_SIZE;
      const int vglobal_token_idx =
          partition_start_token_idx + vlocal_token_idx;
      const int vblock_idx = (vglobal_token_idx < seq_len)
                                 ? vglobal_token_idx / BLOCK_SIZE
                                 : last_seq_block;
      vphysical_block_number[vtoken_depth][vblock_depth] =
          block_table_seq[vblock_idx];
    }
  }

  _B16x8 Vlocal[VTLOOP][VHELOOP]
               [VTLANELOOP];  // this can be interpreted as B8x16 too

  const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride +
                         ((rowid * VTOKENS_PER_LANE) % BLOCK_SIZE);

  // v fetches are 16head elems across lanes x 16 tokens per lane
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    const int vhead_elem = vhe_depth * NWARPS * 16 + warpid * 16 + lane16id;
    const cache_t* v_ptr2 = v_ptr + vhead_elem * BLOCK_SIZE;

    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
      for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
        const int vblock_depth = 0;
        const int64_t vblock_number = static_cast<int64_t>(
            vphysical_block_number[vtoken_depth][vblock_depth]);
        const cache_t* v_ptr3 = v_ptr2 + (vblock_number * kv_block_stride);

        const cache_t* v_fetch_ptr =
            v_ptr3 + vfetch_depth * CONTIGUOUS_KV_ELEMS_16B_LOAD;
        const _B16x8* v_fetch_ptr_16B =
            reinterpret_cast<const _B16x8*>(v_fetch_ptr);
        Vlocal[vtoken_depth][vhe_depth][vfetch_depth] = *v_fetch_ptr_16B;
      }
    }
  }

  floatx8 dout[TLOOP];
  // qk wmma
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    dout[token_depth] = {0};
    for (int qkhe_depth = 0; qkhe_depth < QKHELOOP; qkhe_depth++) {
      dout[token_depth] = gcn_wmma16x16x16_instr<scalar_t, 0, 0, 0>(
          Klocal[token_depth][qkhe_depth].u16x8, Qlocal[qkhe_depth].u16x8,
          dout[token_depth]);
    }
    dout[token_depth] *= scale;
  }

  // calculate qk_max and exp_sum per warp and write to shared memory
  float qk_max = -FLT_MAX;
  float exp_sum = 0.0f;
  const int qkout_token_idx =
      partition_start_token_idx + TOKENS_PER_WARP * warpid + rowid * 8;
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 8; i++) {
      const float tmp =
          (local_token_idx + i < seq_len) ? dout[token_depth][i] : -FLT_MAX;
      qk_max = fmaxf(qk_max, tmp);
    }
  }

  qk_max = fmaxf(qk_max, __shfl_xor(qk_max, 16));

  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    const int local_token_idx = qkout_token_idx + token_depth * 16;
    for (int i = 0; i < 8; i++) {
      const float tmp = (local_token_idx + i < seq_len)
                            ? __expf(dout[token_depth][i] - qk_max)
                            : 0.0f;
      dout[token_depth][i] = tmp;
      exp_sum += tmp;
    }
  }

  exp_sum += __shfl_xor(exp_sum, 16);

  __syncthreads();

  if (laneid < 16) {
    shared_qk_max[warpid][lane16id] = qk_max;
    shared_exp_sum[warpid][lane16id] = exp_sum;
  }

  __syncthreads();

  // calculate partition qk_max and exp_sum
  float partition_qk_max = -FLT_MAX;
  float warp_qk_max_exp[NWARPS];
  float partition_exp_sum = 0.0f;

  #pragma unroll
  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = shared_qk_max[w][lane16id];
    partition_qk_max = fmaxf(partition_qk_max, warp_qk_max_exp[w]);
  }

  for (int w = 0; w < NWARPS; w++) {
    warp_qk_max_exp[w] = __expf(warp_qk_max_exp[w] - partition_qk_max);
    partition_exp_sum += shared_exp_sum[w][lane16id] * warp_qk_max_exp[w];
  }

  const float inv_sum_scale =
      __fdividef(1.f, partition_exp_sum + 1e-6f) * warp_qk_max_exp[warpid];

  __syncthreads();

  // write logits to shared mem
  #pragma unroll
  for (int token_depth = 0; token_depth < TLOOP; token_depth++) {
    dout[token_depth] *= inv_sum_scale;
    shared_logits[warpid][token_depth][lane16id][rowid] =
        from_floatx8<scalar_t>(dout[token_depth]);
  }

  // write out partition max_logits and exp_sum
  if (threadIdx.x < GQA_RATIO) {
    const int qhead_idx = lane16id;
    const int offset = seq_idx * total_num_heads * max_num_partitions +
                       (wg_start_head_idx + qhead_idx) * max_num_partitions +
                       partition_idx;
    max_logits[offset] = partition_qk_max;
    exp_sums[offset] = partition_exp_sum;
  }

  __syncthreads();

  _B16x8 outelems[VHELOOP];
  // Softmax V wmma
  // v layout: 16he across lanes x 16 tokens per lane
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    floatx8 tmp_out = {0};

    for (int vtoken_depth = 0; vtoken_depth < VTLOOP; vtoken_depth++) {
      for (int vfetch_depth = 0; vfetch_depth < VTLANELOOP; vfetch_depth++) {
        const int offset = rowid * VTLANELOOP + vfetch_depth;
        const int offset1 = offset % ROWS_PER_WARP;
        const int offset2 = offset / ROWS_PER_WARP;
        // if output format is 16 qheads across 16 lanes, 16 head elems spread
        // across rows
        tmp_out = gcn_wmma16x16x16_instr<scalar_t, 0, 0, 0>(
            Vlocal[vtoken_depth][vhe_depth][vfetch_depth].u16x8,
            shared_logits[vtoken_depth][offset2][lane16id][offset1].u16x8,
            tmp_out);
      }
    }
    outelems[vhe_depth] = from_floatx8<scalar_t>(tmp_out);
  }

  __syncthreads();

  #pragma unroll
  for (int vhe_depth = 0; vhe_depth < VHELOOP; vhe_depth++) {
    shared_logits[warpid][vhe_depth][lane16id][rowid] =
        outelems[vhe_depth];  // lane16 id head dimension; rowid head element
                              // dimension
  }

  __syncthreads();

  // write to tmp_out with coalesced writes after reading from shared mem
  if (warpid == 0) {
    _B16x8 vout[GQA_RATIO2];
    // each lane writes out 16Bytes of tmp_out along head elem dimension
    const int head_elem_idx = lane16id * 8;
    if (head_elem_idx < HEAD_SIZE) {
      for (int h = 0; h < GQA_RATIO2; h++) {
        const int local_head_idx = 2 * h + rowid;
        const int offset1 = (head_elem_idx / 16) % NWARPS;
        const int offset2 = head_elem_idx / 16 / NWARPS;
        const int offset3 = (head_elem_idx / 8) % 2;  // num_he % num_row
        vout[h] = shared_logits[offset1][offset2][local_head_idx][offset3];
      }

      const int hsz_maxp_mult = HEAD_SIZE * max_num_partitions;
      scalar_t* out_ptr = out + seq_idx * total_num_heads * hsz_maxp_mult +
                          partition_idx * HEAD_SIZE;
      for (int h = 0; h < GQA_RATIO2; h++) {
        const int local_head_idx = 2 * h + rowid;
        if (local_head_idx < GQA_RATIO) {
          const int out_head_idx = wg_start_head_idx + local_head_idx;
          scalar_t* out_ptr2 = out_ptr + out_head_idx * hsz_maxp_mult;
          scalar_t* out_ptr3 = out_ptr2 + head_elem_idx;
          _B16x8* out_ptr_B16x8 = reinterpret_cast<_B16x8*>(out_ptr3);
          *out_ptr_B16x8 = vout[h];
        }
      }
    }
  }
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma4_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,    // [num_seqs, num_heads, max_num_partitions,
                                   // head_size]
    OUTT* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  const auto num_heads = gridDim.x;
  const auto head_idx = blockIdx.x;
  const auto seq_idx = blockIdx.y;

  // NOTE queries with sequence len > 1 are prefills and taken care by another
  // kernel.
  if (query_start_loc_ptr != nullptr &&
      (query_start_loc_ptr[seq_idx + 1] - query_start_loc_ptr[seq_idx] != 1)) {
    return;
  }

  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  const int warpid = threadIdx.x / WARP_SIZE;

  __shared__ float shared_global_exp_sum;
  // max num partitions supported is warp_size * NPAR_LOOPS
  __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];

  if (warpid == 0) {
    const float* max_logits_ptr = max_logits +
                                  seq_idx * num_heads * max_num_partitions +
                                  head_idx * max_num_partitions;

    // valid partition is the last valid partition in case threadid > num
    // partitions
    int valid_partition[NPAR_LOOPS];
    float reg_max_logit[NPAR_LOOPS];
    const int last_valid_partition = num_partitions - 1;

  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      valid_partition[i] =
          (partition_no < num_partitions) ? partition_no : last_valid_partition;
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      reg_max_logit[i] = max_logits_ptr[valid_partition[i]];
    }
    float max_logit = reg_max_logit[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      max_logit = fmaxf(max_logit, reg_max_logit[i]);
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
    }

    const float* exp_sums_ptr = exp_sums +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;

    float rescaled_exp_sum[NPAR_LOOPS];
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      rescaled_exp_sum[i] *= (partition_no < num_partitions)
                                 ? expf(reg_max_logit[i] - max_logit)
                                 : 0.0f;
    }
    float global_exp_sum = rescaled_exp_sum[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      global_exp_sum += rescaled_exp_sum[i];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      shared_exp_sums[partition_no] = rescaled_exp_sum[i];
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      global_exp_sum += __shfl_xor(global_exp_sum, mask);
    }
    if (threadIdx.x == 0) {
      shared_global_exp_sum = global_exp_sum;
    }
  }  // warpid == 0
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
  constexpr int MAX_NPAR = 32;
  scalar_t tmps[MAX_NPAR];
  const float dzero = 0.0f;
  #pragma unroll
  for (int j = 0; j < MAX_NPAR; j++) {
    tmps[j] = from_float<scalar_t>(dzero);
  }
  const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
  const int num_partition_offset = (num_partitions)*HEAD_SIZE;
  int idx = 0;

  constexpr int JCHUNK = 16;

  #pragma unroll
  for (int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE) {
    // lastj is last valid partition
    const int lastj_offset =
        (j < num_partition_offset) ? j : last_partition_offset;
    tmps[idx] = tmp_out_ptr[lastj_offset];
    idx++;
  }
  __syncthreads();

  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE;
         j += HEAD_SIZE) {
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }
    }
  }  // num_partitions > JCHUNK

  // Aggregate tmp_out to out.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < JCHUNK; j++) {
    acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
  }
  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK; j < 2 * JCHUNK; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
      }
    }
  }

  for (int p = 1; p < NPAR_LOOPS; p++) {
    if (num_partitions > p * MAX_NPAR) {
      idx = 0;
  #pragma unroll
      for (int j = p * MAX_NPAR * HEAD_SIZE; j < (p + 1) * MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        // lastj is last valid partition
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }

  #pragma unroll
      for (int j = 0; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + p * MAX_NPAR];
      }
    }
  }

  const float inv_global_exp_sum =
      __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
  acc *= inv_global_exp_sum;

  const int64_t query_start_off = static_cast<int64_t>(
      query_start_loc_ptr ? query_start_loc_ptr[seq_idx] : seq_idx);
  OUTT* out_ptr = out + query_start_off * num_heads * HEAD_SIZE +
                  static_cast<int64_t>(head_idx) * HEAD_SIZE;
  out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
}

#else

// clang-format off
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO, MFMAType MFMA_TYPE,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,    // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,    // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,    // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,             // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,           // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,               // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,             // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  UNREACHABLE_CODE
}

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO,
          int KV_CACHE_LAYOUT>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma4_kernel(
    const scalar_t* __restrict__ q,          // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,     // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,    // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    float* __restrict__ exp_sums,            // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,          // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,              // [num_seqs, num_heads, max_num_partitions, head_size]
    OUTT* __restrict__ final_out,            // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const int* __restrict__ seq_lens,  // [num_seqs]
    const int* __restrict__ query_start_loc_ptr,  // [num_seqs]
    const int max_num_partitions, const float* __restrict__ fp8_out_scale_ptr) {
  UNREACHABLE_CODE
}
// clang-format on

#endif

#define LAUNCH_CUSTOM_ATTENTION_MFMA16(GQA_RATIO)                              \
  paged_attention_ll4mi_QKV_mfma16_kernel<                                     \
      T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE, HEAD_SIZE, NTHR, ALIBI_ENABLED,      \
      GQA_RATIO, MFMA_TYPE, KV_CACHE_LAYOUT>                                   \
      <<<grid, block, 0, stream>>>(                                            \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,      \
          block_tables_ptr, seq_lens_ptr, query_start_loc_ptr,                 \
          max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, \
          kv_head_stride, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr,  \
          max_ctx_blocks, k_scale_ptr, v_scale_ptr);

#define LAUNCH_CUSTOM_ATTENTION_MFMA4(GQA_RATIO)                               \
  paged_attention_ll4mi_QKV_mfma4_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,   \
                                         HEAD_SIZE, NTHR, ALIBI_ENABLED,       \
                                         GQA_RATIO, KV_CACHE_LAYOUT>           \
      <<<grid, block, 0, stream>>>(                                            \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,      \
          block_tables_ptr, seq_lens_ptr, query_start_loc_ptr,                 \
          max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, \
          kv_head_stride, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr,  \
          max_ctx_blocks, k_scale_ptr, v_scale_ptr);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                                 \
  paged_attention_ll4mi_reduce_kernel<T, OUTT, HEAD_SIZE, HEAD_SIZE,        \
                                      PARTITION_SIZE, NPAR_LOOPS>           \
      <<<reduce_grid, reduce_block, 0, stream>>>(                           \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr, \
          query_start_loc_ptr, max_num_partitions, fp8_out_scale_ptr);

template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE_OLD,
          bool ALIBI_ENABLED, MFMAType MFMA_TYPE,
          int KV_CACHE_LAYOUT>
void paged_attention_custom_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale) {
  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: query start location is optional for V0 decode should not be used.
  // If batch contains mix of prefills and decode, prefills should be skipped.
  const int* query_start_loc_ptr =
      query_start_loc
          ? reinterpret_cast<const int*>(query_start_loc.value().data_ptr())
          : nullptr;

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());
  // NOTE: fp8_out_scale is optional.
  const auto fp8_out_scale_ptr =
      fp8_out_scale
          ? static_cast<const float*>(fp8_out_scale.value().data_ptr())
          : nullptr;
  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE);

  // partition size is fixed at 256 since both mfma4 and mfma16 kernels support
  // it mfma4 kernel also supports partition size 512
  constexpr int PARTITION_SIZE = 256;
  const int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = 256;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // mfma4 kernel is faster than mfma16 for gqa_ratio <= 4
  switch (gqa_ratio) {
    case 1:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(1);
      break;
    case 2:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(2);
      break;
    case 3:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(3);
      break;
    case 4:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(4);
      break;
    case 5:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(5);
      break;
    case 6:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(6);
      break;
    case 7:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(7);
      break;
    case 8:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(8);
      break;
    case 9:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(9);
      break;
    case 10:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(10);
      break;
    case 11:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(11);
      break;
    case 12:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(12);
      break;
    case 13:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(13);
      break;
    case 14:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(14);
      break;
    case 15:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(15);
      break;
    case 16:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  dim3 reduce_grid(num_heads, num_seqs);
  dim3 reduce_block(head_size);
  const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
  // reduction kernel supports upto 8 NPAR_loops * 64 (warp_size) * 256
  // (partition size) = 128K context length
  switch (npar_loops) {
    case 1:
      LAUNCH_CUSTOM_REDUCTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_REDUCTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_REDUCTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_REDUCTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_REDUCTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_REDUCTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_REDUCTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_REDUCTION(8);
      break;
    default:
      TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
      break;
  }
}

template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE_OLD,
          bool ALIBI_ENABLED, MFMAType MFMA_TYPE,
          int KV_CACHE_LAYOUT>
void paged_attention_custom_launcher_navi(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: query start location is optional for V0 decode should not be used.
  // If batch contains mix of prefills and decode, prefills should be skipped.
  const int* query_start_loc_ptr =
      query_start_loc
          ? reinterpret_cast<const int*>(query_start_loc.value().data_ptr())
          : nullptr;

  // NOTE: Navi does not support alibi_slopes.
  const float* alibi_slopes_ptr = nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());
  // NOTE: Navi does not support fp8.
  const auto fp8_out_scale_ptr = nullptr;
  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE);

  constexpr int PARTITION_SIZE = 256;
  const int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = 256;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (gqa_ratio) {
    case 1:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(1);
      break;
    case 2:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(2);
      break;
    case 3:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(3);
      break;
    case 4:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(4);
      break;
    case 5:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(5);
      break;
    case 6:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(6);
      break;
    case 7:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(7);
      break;
    case 8:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(8);
      break;
    case 9:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(9);
      break;
    case 10:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(10);
      break;
    case 11:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(11);
      break;
    case 12:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(12);
      break;
    case 13:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(13);
      break;
    case 14:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(14);
      break;
    case 15:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(15);
      break;
    case 16:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  dim3 reduce_grid(num_heads, num_seqs);
  dim3 reduce_block(head_size);
  const int warp_size = 32;
  const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, warp_size);
  // reduction kernel supports upto 16 NPAR_loops * 32 (warp_size) * 256
  // (partition size) = 128K context length
  switch (npar_loops) {
    case 1:
      LAUNCH_CUSTOM_REDUCTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_REDUCTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_REDUCTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_REDUCTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_REDUCTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_REDUCTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_REDUCTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_REDUCTION(8);
      break;
    case 9:
      LAUNCH_CUSTOM_REDUCTION(9);
      break;
    case 10:
      LAUNCH_CUSTOM_REDUCTION(10);
      break;
    case 11:
      LAUNCH_CUSTOM_REDUCTION(11);
      break;
    case 12:
      LAUNCH_CUSTOM_REDUCTION(12);
      break;
    case 13:
      LAUNCH_CUSTOM_REDUCTION(13);
      break;
    case 14:
      LAUNCH_CUSTOM_REDUCTION(14);
      break;
    case 15:
      LAUNCH_CUSTOM_REDUCTION(15);
      break;
    case 16:
      LAUNCH_CUSTOM_REDUCTION(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
      break;
  }
}

// kv_cache_layout selects the KV cache read path at compile time.
//   0 = PAGED:   K=[blocks,heads,hd/x,bs,x]  V=[blocks,heads,hd,bs]
//   1 = FLASH:   K/V=[blocks,bs,heads,hd]  (flat NHD per page)
//   2 = SHUFFLE: K=same as PAGED, V=[blocks,heads,bs/x,hd,x] (interleaved)
// GFX9 kernels support all three layouts.
// NAVI (GFX11/GFX12) kernels only support layout 0 (PAGED).
#define CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT,     \
                             PSIZE, ALIBI_ENABLED, MFMA_TYPE)                 \
  if (kv_cache_layout == 2) {                                                  \
    TORCH_CHECK(!is_navi,                                                     \
                "Shuffle/interleaved KV cache layout (2) is not supported "   \
                "on NAVI GPUs");                                               \
    paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,    \
                                    OUTT, PSIZE, ALIBI_ENABLED, MFMA_TYPE,    \
                                    2>(                                        \
        out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,    \
        num_kv_heads, scale, block_tables, seq_lens, query_start_loc,         \
        max_seq_len, alibi_slopes, k_scale, v_scale, fp8_out_scale);          \
  } else if (kv_cache_layout == 1) {                                           \
    TORCH_CHECK(!is_navi,                                                     \
                "Flash KV cache layout (1) is not supported on NAVI GPUs");   \
    paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,    \
                                    OUTT, PSIZE, ALIBI_ENABLED, MFMA_TYPE,    \
                                    1>(                                        \
        out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,    \
        num_kv_heads, scale, block_tables, seq_lens, query_start_loc,         \
        max_seq_len, alibi_slopes, k_scale, v_scale, fp8_out_scale);          \
  } else {                                                                    \
    if (!is_navi) {                                                           \
      paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,  \
                                      OUTT, PSIZE, ALIBI_ENABLED, MFMA_TYPE,  \
                                      0>(                                      \
          out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,  \
          num_kv_heads, scale, block_tables, seq_lens, query_start_loc,       \
          max_seq_len, alibi_slopes, k_scale, v_scale, fp8_out_scale);        \
    } else {                                                                  \
      paged_attention_custom_launcher_navi<T, KVT, KV_DTYPE, BLK_SIZE,        \
                                           HEAD_SIZE, OUTT, PSIZE,            \
                                           ALIBI_ENABLED, MFMA_TYPE,           \
                                           0>(                                 \
          out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,  \
          num_kv_heads, scale, block_tables, seq_lens, query_start_loc,       \
          max_seq_len, alibi_slopes, k_scale, v_scale);                       \
    }                                                                         \
  }

#define CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,    \
                                   OUTT, PSIZE, MFMA_TYPE)                   \
  if (alibi_slopes) {                                                        \
    CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, \
                         true, MFMA_TYPE);                                   \
  } else {                                                                   \
    CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, OUTT, PSIZE, \
                         false, MFMA_TYPE);                                  \
  }

#if defined(__HIPCC__) && defined(__gfx90a__)
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,  \
                                   MFMA_TYPE)                              \
    if (fp8_out_scale) {                                                   \
      TORCH_CHECK(false, "fp8 out scale unsupported for gfx90a");          \
    } else {                                                               \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T, \
                                 256, MFMA_TYPE);                          \
    }
#else
  #define CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,  \
                                   MFMA_TYPE)                              \
    if (fp8_out_scale) {                                                   \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE,    \
                                 uint8_t, 256, MFMA_TYPE);                 \
    } else {                                                               \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T, \
                                 256, MFMA_TYPE);                          \
    }
#endif

#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE, MFMA_TYPE)    \
  switch (block_size) {                                                     \
    case 16:                                                                \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 16, HEAD_SIZE, MFMA_TYPE); \
      break;                                                                \
    case 32:                                                                \
      CALL_CUSTOM_LAUNCHER_OUT(T, KVT, KV_DTYPE, 32, HEAD_SIZE, MFMA_TYPE); \
      break;                                                                \
    default:                                                                \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);           \
      break;                                                                \
  }

#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE, MFMA_TYPE) \
  switch (head_size) {                                             \
    case 64:                                                       \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64, MFMA_TYPE);   \
      break;                                                       \
    case 128:                                                      \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128, MFMA_TYPE);  \
      break;                                                       \
    default:                                                       \
      TORCH_CHECK(false, "Unsupported head size: ", head_size);    \
      break;                                                       \
  }

bool is_navi_gpu() {
  static bool is_cached = false;
  static bool result;

  if (!is_cached) {
    int device_id;
    hipDeviceProp_t deviceProp;
    hipGetDevice(&device_id);
    hipGetDeviceProperties(&deviceProp, device_id);

    std::string arch = deviceProp.gcnArchName;
    result = arch.find("gfx11") == 0 || arch.find("gfx12") == 0;
    is_cached = true;
  }

  return result;
}

// clang-format off
void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,     // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // layout depends on kv_cache_layout
    torch::Tensor& value_cache, // layout depends on kv_cache_layout
    int64_t num_kv_heads, 
    double scale,
    torch::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens, // [num_seqs]
    const std::optional<torch::Tensor>& query_start_loc, // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type,
    int64_t kv_cache_layout) {  // 0=PAGED, 1=FLASH, 2=SHUFFLE
  // clang-format on
  bool is_navi = is_navi_gpu();
  const int head_size = query.size(2);
  if (kv_cache_dtype == "auto") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(
          _Float16, _Float16, vllm::Fp8KVCacheDataType::kAuto, MFMAType::F16);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, __hip_bfloat16,
                                    vllm::Fp8KVCacheDataType::kAuto,
                                    MFMAType::F16);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (query.dtype() == at::ScalarType::Half) {
      if (mfma_type == "fp8") {
        CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t,
                                      vllm::Fp8KVCacheDataType::kFp8E4M3,
                                      MFMAType::Fp8);
      } else {
        CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t,
                                      vllm::Fp8KVCacheDataType::kFp8E4M3,
                                      MFMAType::F16);
      }
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      if (mfma_type == "fp8") {
        CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                      vllm::Fp8KVCacheDataType::kFp8E4M3,
                                      MFMAType::Fp8);
      } else {
        CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                      vllm::Fp8KVCacheDataType::kFp8E4M3,
                                      MFMAType::F16);
      }
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else {
    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
