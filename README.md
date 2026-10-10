# NInfer Extended

English · [简体中文](README_zh-CN.md)

A C++/CUDA inference engine extending [Neroued/ninfer](https://github.com/Neroued/ninfer)
with Qwen3.5-0.8B multimodal support, Qwen3 speech recognition, forced alignment
and text embedding. Qwen3.5-0.8B support is validated with **Xiaomi-OCR-0**, based
on **Qwen3.5-0.8B-Base**.

The engine runs models through the native v3 `.ninfer` artifact and Engine APIs.
It provides local CLIs and an OpenAI-/Anthropic-compatible HTTP server, with
shared CUDA operators for vision, audio and language processing.

## Features

- **OCR and document parsing:** native BF16 vision encoding, language prefill
  and decode for Xiaomi-OCR-0, with image processing and tokenizer resources
  included in the model artifact.
- **Speech recognition:** a native audio encoder and Qwen3 decoder, the
  `SpeechRecognition` Engine purpose, `transcribe_features()` API and
  `ninfer-asr` CLI. Audio encoding, language prefill and decode use CUDA Graphs.
- **Word alignment:** native Qwen3-ForcedAligner inference, independent batches
  and reusable CUDA Graphs. It produces word timestamps in one audio/language
  forward pass.
- **Text embedding:** packed causal attention, last-token pooling,
  dimensional truncation and FP32 L2 normalization on CUDA, exposed through
  `embed_tokens()` and the `ninfer-embed` CLI.
- **Shared BF16 operators:** compact projections, fused residual and split
  outputs, fused bias/residual and bias/GELU, register-fused SwiGLU,
  Gated DeltaNet controls and segmented attention.
  Implementations are selected by tensor geometry, dtype and device resources.
- **Model conversion:** BF16 recipes for OCR, ASR, forced alignment and embedding;
  embedded tokenizer/processor resources and Unicode pre-tokenization.
- **Verification tools:** independent numerical checks and complete-model
  comparisons with compiled Transformers.

## Models

| Model | Guide | Hugging Face | ModelScope |
|---|---|---|---|
| Xiaomi-OCR-0 BF16 | [OCR](docs/xiaomi-ocr.md) | [Model](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer) | [Model](https://modelscope.cn/models/ByronLeeee/Xiaomi-OCR-0-Ninfer) |
| Qwen3-ASR-1.7B-hf BF16 | [ASR](docs/qwen3-asr.md) | [Model](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) | [Model](https://modelscope.cn/models/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) |
| Qwen3-ForcedAligner-0.6B BF16 | [Alignment](docs/qwen3-forced-aligner.md) | [ASR and alignment](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) | [ASR and alignment](https://modelscope.cn/models/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) |
| Qwen3-Embedding-0.6B BF16 | [Embedding](docs/qwen3-embedding.md) | [Model](https://huggingface.co/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer) | [Model](https://modelscope.cn/models/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer) |

Existing upstream Qwen3.6-27B, Qwen3.8-27B and Qwen3.6-35B-A3B v3 artifacts
and their generation routes are also supported.

## Transformers comparison

Results use BF16 weights and matched inputs on each GPU. OCR and ASR references
use compiled Transformers; embedding uses the faster eager or compiled path;
alignment uses the official eager Transformers reference. Tables report medians.

### Xiaomi-OCR-0

Single request, 4K context, BF16 KV and a 256-token output cap. Prefill includes
vision encoding and language prefill; decode starts after the first token.
Transformers stages use synchronized timing and NInfer uses CUDA events. NInfer chunk size is 1,024 on 5070 Ti and 2,048 on 6000D.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | Prefill change | TF decode tok/s | NInfer decode tok/s | Decode speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 20,983 | 24,125 | +15.0% | 148.6 | 438.6 | 2.95× |
| RTX 5070 Ti | English contract | 21,402 | 23,830 | +11.3% | 143.6 | 439.5 | 3.06× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 20,971 | 22,710 | +8.3% | 147.7 | 435.8 | 2.95× |
| RTX 6000D | Chinese document | 36,263 | 40,404 | +11.4% | 232.2 | 540.5 | 2.33× |
| RTX 6000D | English contract | 36,247 | 40,541 | +11.8% | 231.7 | 540.6 | 2.33× |
| RTX 6000D | Dense formulas (256 tokens) | 36,174 | 36,842 | +1.8% | 234.0 | 537.4 | 2.30× |

Both engines score **100% character accuracy (CER 0%)** on the two annotated
text pages. Raw output agreement is **3/3 (100%)**, covering both complete text
outputs and the same 256-token formula prefix. [Full OCR results](docs/xiaomi-ocr-performance.md).

### Qwen3-ASR-1.7B

Complete transcription speed, measured as **audio seconds processed per second**.
This equals total audio duration divided by the complete warm model-call time;
four-lane results sum all four recordings. Model loading, CPU features and
first Graph capture are excluded.

| GPU | Input | TF audio seconds/s | NInfer audio seconds/s | Speedup |
|---|---|---:|---:|---:|
| RTX 5070 Ti | English 60 s | 19.9 | **62.5** | 3.14× |
| RTX 5070 Ti | English 30 s × 4 | 54.1 | **197.8** | 3.66× |
| RTX 6000D | English 60 s | 29.0 | **87.7** | 3.03× |
| RTX 6000D | English 30 s × 4 | 84.3 | **299.0** | 3.55× |

Prefill and decode for one 15.05-second English recording:

| GPU | TF prefill ms | NInfer prefill ms | Prefill speedup | TF decode tok/s | NInfer decode tok/s | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 39.416 | 16.456 | 2.40× | 68.2 | 213.4 | 3.13× |
| RTX 6000D | 13.487 | 10.544 | 1.28× | 93.3 | 294.8 | 3.16× |

Four-lane inference on RTX 6000D:

| Input | TF prefill ms | NInfer prefill ms | Prefill speedup | TF decode tok/s | NInfer decode tok/s | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|
| English 15.05 s × 4 | 34.437 | 29.374 | 1.17× | 278.9 | 1129.5 | 4.05× |
| Chinese 4.20 s × 4 | 18.098 | 12.534 | 1.44× | 278.6 | 1153.0 | 4.14× |
| English 30 s × 4 | 58.855 | 57.538 | 1.02× | 278.5 | 1096.5 | 3.94× |

For 60 seconds of concatenated English audio, complete warm inference takes:

| GPU | TF inference time | NInfer inference time | TF audio s/s | NInfer audio s/s | Speedup |
|---|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 3.011 s | 0.960 s | 19.9 | 62.5 | 3.14× |
| RTX 6000D | 2.071 s | 0.684 s | 29.0 | 87.7 | 3.03× |

Single-lane recognition quality on 73 labelled English recordings, totaling 1,150 gold words:

| GPU | TF WER | NInfer WER | Raw output agreement | Normalized text agreement |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 3.9130% | 3.8261% | 98.63% | 98.63% |
| RTX 6000D | 3.5652% | 3.5652% | 97.26% | 100.00% |

[ASR results and usage](model-cards/Qwen3-ASR-1.7B-hf-NInfer/README.md).

### Qwen3-Embedding-0.6B

Single text, normalized 1,024-dimensional vectors. GPU time includes complete
language prefill and pooling; embeddings have no decode phase.

| GPU | Tokens | TF GPU ms | NInfer GPU ms | Throughput change |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 128 | 3.266 | 2.578 | +26.69% |
| RTX 5070 Ti | 512 | 8.781 | 8.150 | +7.75% |
| RTX 5070 Ti | 2048 | 42.161 | 37.610 | +12.10% |
| RTX 6000D | 128 | 2.797 | 1.980 | +41.29% |
| RTX 6000D | 512 | 6.187 | 4.616 | +34.02% |
| RTX 6000D | 2048 | 21.251 | 21.298 | -0.22% |

Across 249 vectors, mean cosine similarity is approximately **0.9998**.
Both engines retrieve the correct Top1 document for all eight tested queries,
with **100% Top1/Top3 agreement**. [Embedding results](model-cards/Qwen3-Embedding-0.6B-NInfer/README.md).

### Qwen3-ForcedAligner-0.6B

One 15.05-second English recording; GPU time includes audio encoding,
language forward and timestamp classification.

| GPU | TF GPU ms | NInfer GPU ms | Speedup |
|---|---:|---:|---:|
| RTX 5070 Ti | 68.464 | 10.283 | 6.66× |
| RTX 6000D | 10.920 | 6.157 | 1.77× |

Across four single-sample cases, timestamp-boundary agreement is **100%** on
5070 Ti and **99.75%** on 6000D; maximum differences are **0 ms / 80 ms**.
[Alignment results and usage](docs/qwen3-forced-aligner.md).

## Build

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build -j
```

Requires 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`, C++20,
CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl ≥7.85, pkg-config,
PCRE2, cuDNN 9 and cuBLAS. Omit `CUBLAS_ROOT` when CUDA provides cuBLAS.
The extension is tested on RTX 5070 Ti and RTX 6000D.

## Documentation

- [CLI guide](docs/cli.md)
- [Serving guide](docs/serving.md)
- [Weight conversion](docs/weight-conversion.md)
- [Updates and optimization results](CHANGELOG.md)
- [OCR performance](docs/xiaomi-ocr-performance.md)
- [Build configuration](docs/maintainer/build-system.md)

---

Based on [Neroued/ninfer](https://github.com/Neroued/ninfer), starting from
`594930e7b609efa4bcea3ae4f24cd9d66b5f224f`.
See the [upstream project](https://github.com/Neroued/ninfer#readme) for its original introduction.
