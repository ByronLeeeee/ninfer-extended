---
license: apache-2.0
base_model: SeerRay-Lab/Xiaomi-OCR-0
pipeline_tag: image-text-to-text
library_name: ninfer
language:
- zh
- en
tags:
- ninfer
- bf16
- ocr
- document-parsing
- qwen3_5
- cuda
---

# Xiaomi-OCR-0 BF16 for NInfer

This BF16 conversion of [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) supports OCR and document parsing. The source model is based on Qwen3.5-0.8B-Base. The artifact includes language and vision weights, the tokenizer and image-processing configuration.

## Build and run

Build the **[ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended)** main branch to run this artifact.

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
pip install huggingface_hub
hf download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy
```

This starts an OCR service with 32K context per request, four concurrent requests and BF16 weights/KV. Build dependencies: 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl, PCRE2, cuDNN 9 and cuBLAS. Omit `CUBLAS_ROOT` when cuBLAS is installed with CUDA.

[Build and conversion guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md)

For RTX 6000D, set `--prefill-chunk 2048` for the measured OCR profile.

## Transformers vs NInfer performance

On RTX 5070 Ti, complete prefill throughput improves by **8.3–15.0%** and decode reaches **2.95–3.06×** compiled Transformers. On RTX 6000D, complete prefill throughput improves by **1.8–11.8%** and decode reaches **2.30–2.33×** compiled Transformers.

Both engines use BF16 weights/KV, 4K context, one request, identical images/prompts, greedy decoding and a 256-token output cap. NInfer uses 1,024-token chunks on RTX 5070 Ti and 2,048-token chunks on RTX 6000D. Text pages finish naturally; the formula page measures its first 256 output tokens. Results are medians.

Transformers runs compiled vision/language prefill and decode with verified fused causal-conv1d and Gated DeltaNet kernels. Prefill includes vision encoding and language prefill; Transformers uses synchronized stage timing and NInfer uses CUDA events. Decode starts after the first generated token.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | Prefill change | TF decode tok/s | NInfer decode tok/s | Decode speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 20,983 | 24,125 | +15.0% | 148.6 | 438.6 | 2.95× |
| RTX 5070 Ti | English contract | 21,402 | 23,830 | +11.3% | 143.6 | 439.5 | 3.06× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 20,971 | 22,710 | +8.3% | 147.7 | 435.8 | 2.95× |
| RTX 6000D | Chinese document | 36,263 | 40,404 | +11.4% | 232.2 | 540.5 | 2.33× |
| RTX 6000D | English contract | 36,247 | 40,541 | +11.8% | 231.7 | 540.6 | 2.33× |
| RTX 6000D | Dense formulas (256 tokens) | 36,174 | 36,842 | +1.8% | 234.0 | 537.4 | 2.30× |

Prefill change is `(NInfer / Transformers − 1) × 100%`; decode speedup is `NInfer / Transformers`.

### End-to-end request time

Request time includes preprocessing, inference and the local HTTP round trip.

| GPU | Input | TF request ms | NInfer request ms | Speedup |
|---|---|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 956.7 | 412.8 | 2.32× |
| RTX 5070 Ti | English contract | 980.5 | 432.7 | 2.27× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 1981.8 | 791.3 | 2.50× |
| RTX 6000D | Chinese document | 582.2 | 295.1 | 1.97× |
| RTX 6000D | English contract | 583.5 | 295.6 | 1.97× |
| RTX 6000D | Dense formulas (256 tokens) | 1208.1 | 581.1 | 2.08× |

## OCR accuracy and output agreement

The two complete text pages contain 478 annotated characters. Character accuracy is `1 − CER`, scored after character normalization and removal of whitespace and Markdown markers. Output agreement covers both complete text outputs and the 256-token formula output.

| GPU | Transformers character accuracy | NInfer character accuracy | Exact output agreement | Character difference |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |
| RTX 6000D | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |

On both GPUs, NInfer also scores **100% character accuracy (CER 0%)** on six annotated text images totaling 1,117 normalized characters. BF16 projection, fused bias/residual and bias/GELU, SwiGLU, Gated DeltaNet and D64 attention pass independent FP64 checks, including CUDA Graph replay and dispatch boundaries.

## Artifact and license

`xiaomi-ocr-0-bf16.ninfer` · 1,726,110,464 bytes · BF16 · Apache-2.0

SHA256: `1db6a88a5947ba06e88210feff0c44f8eeee6d7b506b4744cbf9f94a0e61cb9d`

[Measurements](gpu-results.json) · [Artifact manifest](artifact-manifest.json)
