#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                     std::int32_t last);
void attention_projection(const Tensor& hidden, const AttentionParameters& parameters,
                          Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                          WorkspaceArena& workspace, cudaStream_t stream);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream);
void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query, Tensor& key,
               cudaStream_t stream);

void text_norm_rope(const Tensor& positions, const RopeConfig& config, const Tensor& query_weight,
                    const Tensor& key_weight, float epsilon, Tensor& query, Tensor& key,
                    Tensor& normalized_query, Tensor& normalized_key, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
