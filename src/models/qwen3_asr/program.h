#pragma once

#include "ninfer/asr.h"
#include "ninfer/alignment.h"
#include "ninfer/types.h"
#include "core/device.h"

#include <memory>

namespace ninfer::models::qwen3_asr {

class Program {
public:
    Program(const EngineOptions&, DeviceContext&);
    ~Program();
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    SpeechResult transcribe(std::vector<SpeechFeatures>, const SpeechRunOptions&);
    AlignmentResult align(std::vector<SpeechFeatures>, const AlignmentRunOptions&);
    LoadSummary load_summary() const;
    MemorySummary memory_summary() const;
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen3_asr
