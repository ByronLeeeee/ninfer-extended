#pragma once

#include <cstdint>
#include <vector>

namespace ninfer {
enum class EmbeddingLinearBackend : std::uint8_t { Cublas, Native };
enum class EmbeddingAttention : std::uint8_t { Automatic, Float32, TensorCore };

struct EmbeddingRunOptions {
    EmbeddingLinearBackend linear = EmbeddingLinearBackend::Native;
    std::uint32_t dimensions = 0; // Zero selects the full hidden width.
    bool normalize = true;
    bool cuda_graph = true;
    EmbeddingAttention attention = EmbeddingAttention::Automatic;
};

struct EmbeddingResult {
    std::vector<std::vector<float>> vectors;
    std::uint64_t input_tokens = 0;
    double gpu_ms = 0;
    double wall_seconds = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t runtime_bytes = 0;
    bool graph_used = false;
    bool tensorcore_attention_used = false;
};
} // namespace ninfer
