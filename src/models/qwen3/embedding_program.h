#pragma once
#include "ninfer/types.h"
#include "ninfer/embedding.h"
#include "core/device.h"
#include <memory>

namespace ninfer::models::qwen3 {
class EmbeddingProgram {
public:
    EmbeddingProgram(const EngineOptions&, DeviceContext&);
    ~EmbeddingProgram();
    EmbeddingProgram(const EmbeddingProgram&) = delete;
    EmbeddingProgram& operator=(const EmbeddingProgram&) = delete;
    EmbeddingResult embed(const std::vector<std::vector<TokenId>>&, const EmbeddingRunOptions&);
    LoadSummary load_summary() const;
    MemorySummary memory_summary() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::models::qwen3
