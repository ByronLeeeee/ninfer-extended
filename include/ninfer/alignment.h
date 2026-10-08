#pragma once

#include "ninfer/asr.h"

namespace ninfer {

// One non-autoregressive forward pass. Timestamp token positions in each
// SpeechFeatures prompt are selected in order; no decode or KV cache is used.
struct AlignmentRunOptions {
    SpeechLinearBackend linear = SpeechLinearBackend::Native;
    bool audio_graph = true;
    bool prefill_graph = true;
    bool causal_tensorcore_prefill = true;
};

struct AlignmentResult {
    std::vector<std::vector<std::int32_t>> timestamp_classes;
    std::int32_t timestamp_segment_ms = 0;
    double audio_ms = 0;
    double language_ms = 0;
    double wall_seconds = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t runtime_bytes = 0;
    bool audio_graph_used = false;
    bool prefill_graph_used = false;
};

} // namespace ninfer
