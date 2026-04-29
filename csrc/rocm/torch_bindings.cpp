#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type,"
      "                int kv_cache_layout) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);

  // Fused QK-norm + RoPE + per-tensor-quant + KV-cache update.
  // vLLM-native port of AITER's fused_qk_norm_rope_cache_pts_quant_shuffle,
  // adding the dim-major LDS staging fast path on the SHUFFLE write layout.
  // Uses slot_mapping directly to derive group-to-cache addressing — no
  // per-sequence workspace tensors required (matches AITER's flat-grid host
  // path; eliminates a value-dependent CPU<->GPU sync per forward pass).
  rocm_ops.def(
      "fused_qk_norm_rope_cache(Tensor! qkv, Tensor q_weight, Tensor k_weight,"
      "                         Tensor cos_sin_cache, Tensor positions,"
      "                         int num_heads_q, int num_heads_k,"
      "                         int num_heads_v, int head_dim,"
      "                         bool is_neox, float eps,"
      "                         Tensor! q_out,"
      "                         Tensor! k_cache, Tensor! v_cache,"
      "                         Tensor slot_mapping,"
      "                         Tensor per_tensor_k_scale,"
      "                         Tensor per_tensor_v_scale,"
      "                         str kv_cache_dtype,"
      "                         Tensor!? k_out, Tensor!? v_out,"
      "                         bool return_kv, bool use_shuffle_layout,"
      "                         int block_size, int x) -> ()");
  rocm_ops.impl("fused_qk_norm_rope_cache", torch::kCUDA,
                &fused_qk_norm_rope_cache);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
