#pragma once

#include <cstdint>
#include <vector>

namespace ninfer {

// CPU frontend output. Mel values are [mel_bin,frame] in row-major order;
// frame_mask preserves the checkpoint's padded-frame semantics.
struct SpeechFeatures {
    std::vector<float> mel;
    std::vector<std::int32_t> frame_mask;
    std::vector<std::int32_t> prompt_tokens;
    std::int32_t mel_bins = 128;
    std::int32_t frames = 0;
    std::int32_t audio_token_id = 151676;
};

enum class SpeechLinearBackend : std::uint8_t { Cublas, Native };
enum class SpeechDraftSource : std::uint8_t { ExternalModel, NGram };

struct SpeechRunOptions {
    std::uint32_t max_new_tokens = 1024;
    // Greedy block verification, single lane, K=1..5; zero disables it.
    std::uint32_t draft_tokens = 0;
    SpeechDraftSource draft_source = SpeechDraftSource::ExternalModel;
    SpeechLinearBackend linear = SpeechLinearBackend::Native;
    bool decode_graph = true;
    bool audio_graph = true;
    bool prefill_graph = true;
    bool audio_flash_attention = true;
    bool fused_qk_norm_rope = true;
    bool fused_projection_residual = true;
    bool causal_tensorcore_prefill = false;
};

struct SpeechResult {
    std::vector<std::vector<std::int32_t>> token_ids;
    double audio_ms = 0;
    double language_prefill_ms = 0;
    double decode_ms = 0;
    double wall_seconds = 0;
    double decode_wall_seconds = 0;
    double draft_audio_ms = 0;
    double draft_language_prefill_ms = 0;
    // Includes a paired catch-up/first-proposal forward when used.
    double draft_decode_ms = 0;
    double verify_ms = 0;
    // Standalone catch-up only; zero when catch-up is paired with drafting.
    double draft_repair_ms = 0;
    std::uint64_t proposed_tokens = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t verification_calls = 0;
    std::uint64_t fully_accepted_blocks = 0;
    std::uint64_t lookup_attempts = 0;
    std::uint64_t lookup_matches = 0;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t decode_tokens = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t runtime_bytes = 0;
    bool decode_graph_used = false;
    bool audio_graph_used = false;
    bool prefill_graph_used = false;
};

} // namespace ninfer
