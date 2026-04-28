#pragma once

#include <torch/all.h>
#include <optional>
#include <string>

void fused_qk_norm_rope_cache(
    torch::Tensor& qkv, torch::Tensor& q_weight, torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache, torch::Tensor& positions,
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v,
    int64_t head_dim, bool is_neox, double eps, torch::Tensor& q_out,
    torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& slot_mapping, torch::Tensor& query_start_loc,
    torch::Tensor& block_to_seq, torch::Tensor& block_to_group_in_seq,
    torch::Tensor& per_tensor_k_scale, torch::Tensor& per_tensor_v_scale,
    const std::string& kv_cache_dtype,
    const std::optional<torch::Tensor>& k_out,
    const std::optional<torch::Tensor>& v_out, bool return_kv,
    bool use_shuffle_layout, int64_t block_size, int64_t x);
