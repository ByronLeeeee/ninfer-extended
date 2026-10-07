---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-1.7B-hf
pipeline_tag: automatic-speech-recognition
library_name: ninfer
language:
- en
- zh
tags:
- ninfer
- bf16
- cuda
- qwen3-asr
- speech-recognition
---

# Qwen3-ASR-1.7B BF16 for NInfer

This BF16 conversion of [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) runs with the **main branch of [ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended)**. Audio encoding, language prefill and decode run in the native C++/CUDA engine. The artifact includes the tokenizer, processor and chat template, with all 707 source parameters preserved byte for byte.

## Build and transcribe

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-asr -j
pip install "transformers>=5.13.0" torch numpy
pip install huggingface_hub
hf download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer qwen3-asr-1.7b-bf16.ninfer --local-dir models
python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --audio recording.wav --out transcription.json
```

Build dependencies: 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl, PCRE2, cuDNN 9 and cuBLAS. Omit `CUBLAS_ROOT` if cuBLAS is installed with CUDA. Supply one to four mono PCM16 WAV files at 16 kHz. Defaults: BF16 weights/KV, 4K context, up to four lanes and 1,024 output tokens. Language prefill uses FP32 probability arithmetic by default.

[Build, conversion and API guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-asr.md)

## Transformers vs NInfer performance

Measured on 2026-10-07. Transformers uses BF16 GPU audio encoding, fullgraph language compilation and decode CUDA Graphs; CPU offload, compilation fallback and graph breaks were checked, with MATH SDPA disabled. NInfer uses CUDA Graphs for audio, language prefill and decode. Toolchains: CUDA 13.3/GCC 13.3 on 5070 Ti and CUDA 13.2/GCC 15.2 on 6000D; Transformers 5.17.0 and PyTorch 2.14.0+cu132 on both. Medians use three warm measurements for Transformers and six for NInfer.

Prefill includes audio encoding, language prefill and first-token GPU computation. Decode starts after the first token, includes EOS and sums tokens across lanes. All speed ratios compare this NInfer version with Transformers.

### RTX 5070 Ti

| Input | Lanes | TF prefill ms | NInfer prefill ms | Prefill speedup | TF decode tok/s | NInfer decode tok/s | Decode speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| English 15.05 s | 1 | 39.416 | 16.456 | 2.40× | 68.2 | 213.4 | 3.13× |
| Chinese 4.20 s | 1 | 41.861 | 8.639 | 4.85× | 66.1 | 214.8 | 3.25× |
| English 10 s | 1 | 44.510 | 12.537 | 3.55× | 67.4 | 213.3 | 3.16× |
| English 30 s | 1 | 56.840 | 30.695 | 1.85× | 68.3 | 211.2 | 3.09× |
| English 60 s | 1 | 74.651 | 59.059 | 1.26× | 66.8 | 207.8 | 3.11× |
| Chinese 60 s | 1 | 71.823 | 59.173 | 1.21× | 67.4 | 208.0 | 3.08× |
| English 15.05 s × 4 | 4 | 80.500 | 62.949 | 1.28× | 187.3 | 814.1 | 4.35× |
| English 30 s × 4 | 4 | 130.738 | 120.680 | 1.08× | 186.0 | 786.0 | 4.23× |

### RTX 6000D

| Input | Lanes | TF prefill ms | NInfer prefill ms | Prefill speedup | TF decode tok/s | NInfer decode tok/s | Decode speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| English 15.05 s | 1 | 13.336 | 10.519 | 1.27× | 93.5 | 294.6 | 3.15× |
| Chinese 4.20 s | 1 | 10.605 | 6.368 | 1.67× | 92.5 | 296.0 | 3.20× |
| English 10 s | 1 | 11.189 | 8.375 | 1.34× | 92.3 | 294.8 | 3.19× |
| English 30 s | 1 | 20.763 | 16.872 | 1.23× | 92.3 | 289.9 | 3.14× |
| English 60 s | 1 | 33.136 | 32.548 | 1.02× | 92.3 | 282.7 | 3.06× |
| Chinese 60 s | 1 | 32.940 | 32.509 | 1.01× | 92.2 | 283.5 | 3.08× |
| English 15.05 s × 4 | 4 | 34.548 | 40.310 | 0.86× | 279.8 | 1130.4 | 4.04× |
| English 30 s × 4 | 4 | 59.170 | 65.491 | 0.90× | 279.5 | 1095.8 | 3.92× |

## Inference latency and 60-second audio

Latency measures a complete warm model call, including audio encoding, prefill, generation and synchronization. Model loading, CPU feature extraction and initial Graph capture are excluded. Short-clip latency:

| GPU | Input | Transformers ms | NInfer ms | Inference speedup |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | English 15.05 s | 798.0 | 248.8 | 3.21× |
| RTX 5070 Ti | Chinese 4.20 s | 210.0 | 56.9 | 3.69× |
| RTX 6000D | English 15.05 s | 541.3 | 175.1 | 3.09× |
| RTX 6000D | Chinese 4.20 s | 124.5 | 40.5 | 3.07× |

The 60-second inputs concatenate fixed clips for benchmarking. Audio seconds per inference second equals 60 divided by measured warm inference time.

| GPU | 60 s input | TF inference s | NInfer inference s | TF audio s/s | NInfer audio s/s | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | English 60 s | 3.011 | 0.960 | 19.9 | 62.5 | 3.14× |
| RTX 5070 Ti | Chinese 60 s | 1.679 | 0.560 | 35.7 | 107.1 | 3.00× |
| RTX 6000D | English 60 s | 2.071 | 0.685 | 29.0 | 87.5 | 3.02× |
| RTX 6000D | Chinese 60 s | 1.115 | 0.392 | 53.8 | 153.1 | 2.85× |

## Recognition quality and output agreement

Quality is measured on 73 real labelled English recordings: 481.03 seconds and 1,150 gold words. WER normalization removes case, punctuation and extra whitespace while retaining apostrophes; CER excludes spaces.

| GPU | TF WER | NInfer WER | TF CER | NInfer CER | Raw text/token agreement | Normalized text agreement | Token edit rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 3.9130% | 3.8261% | 1.3553% | 1.3166% | 98.63% | 98.63% | 0.0573% |
| RTX 6000D | 3.5652% | 3.5652% | 1.2585% | 1.2585% | 97.26% | 100.00% | 0.1720% |

On 5070 Ti, Transformers has 45 word errors and NInfer 44; the differing `has/is` choice matches the gold transcript in NInfer. On 6000D both have 41 errors and all normalized texts agree; raw differences are capitalization and punctuation. Both GPUs pass independent FP64 operator checks, and the final iteration preserves all 73 previously qualified outputs.

## Artifact and license

`qwen3-asr-1.7b-bf16.ninfer` · 4,087,720,481 bytes · BF16 · Apache-2.0

SHA256: `50d9d8c2ed690b80f5acbf94c66b344ee7fd12cf79dfe670ad10e96b5b411eea`

[Measurements](gpu-results.json) · [Artifact manifest](artifact-manifest.json)
