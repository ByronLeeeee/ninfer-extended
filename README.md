# NInfer Extended

An extension of [Neroued/ninfer](https://github.com/Neroued/ninfer) with **Qwen3.5-0.8B multimodal support**, **native Qwen3-ASR-1.7B**, **native Qwen3-Embedding-0.6B**, and shared BF16 CUDA operators measured on **RTX 5070 Ti** and **RTX 6000D**. Qwen3.5-0.8B support has been validated with **Xiaomi-OCR-0**, based on **Qwen3.5-0.8B-Base**. The fork builds on upstream `594930e7b609efa4bcea3ae4f24cd9d66b5f224f` and keeps the v3 artifact/Engine execution path.

## Changes from upstream NInfer

- **Qwen3.5-0.8B:** BF16/A16 projection, Gated DeltaNet, linear-add and SwiGLU routes for a 1,024-wide decoder; D256 text attention with Q8/KV2; D64/H12 segmented vision attention; 6,144-channel causal convolution; corrected vision position ordering and two-axis RoPE.
- **Qwen3-ASR-1.7B:** a native audio encoder and Qwen3 language decoder, the `SpeechRecognition` Engine purpose, `transcribe_features()` API, and `ninfer-asr` CLI. BF16 cuDNN convolution and audio projections feed shared language Ops. Audio encoding, language prefill and decode all use CUDA Graphs.
- **Qwen3-Embedding-0.6B:** a BF16 Qwen3 decoder, the `TextEmbedding` Engine purpose, `embed_tokens()` API, and `ninfer-embed` CLI. Packed sequences use independent causal ranges and positions, with one Tensor Core attention invocation per layer for the batch. Last-token pooling, dimensional truncation and stable FP32 L2 normalization produce vectors directly on CUDA. The CPU frontend uses the original tokenizer and retrieval instruction format.
- **Small-batch BF16 operators:** compact GEMV/SIMT projections, fused residual and split-output projections, fused SwiGLU, and offset RMSNorm plus GDN controls. Matrix shape, dtype, strides, token extent and supplied physical SM count select the implementations.
- **Attention:** compact D64 segmented attention; fewer split-KV partitions and roughly half the temporary workspace on selected Q8/KV2 decode routes; D128 causal GQA prefill and partitioned Q16/KV8 decode. Long-input Tensor Core attention can stage wider K/V tiles while retaining 32-column softmax groups and probability compensation; workload geometry and physical SM count choose the schedule. The default ASR prefill reuses K/V across four queries in registers while preserving FP32 softmax/probability arithmetic. An optional compensated Tensor Core prefill is also available.
- **Prefill projections:** wider MMA tiles for selected matrix/column extents, narrower tiles for short input rows, register-fused gate/up and SwiGLU for `[6144,1024]`, and FP32 gate/up staging through SiLU and multiplication for `[12288,2048]`. The register-fused profile writes the BF16 activation directly and needs no projection scratch.
- **Conversion and tokenization:** BF16 v3 recipes for OCR, ASR and embedding, embedded tokenizer/processor/chat-template resources, original added-token metadata, and PCRE2 Unicode pre-tokenization. Row concatenation preserves all 707 ASR and 310 embedding source parameters byte for byte.
- **Qualification and reference tools:** independent FP64 and exact-transform checks, complete-model benchmarks, and audited compiled Transformers comparisons. Operator dispatch uses explicit tensor geometry rather than model/GPU names.

## Build and models

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build -j
```

Dependencies: 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl ≥7.85, pkg-config, PCRE2, cuDNN 9 and cuBLAS. Omit `CUBLAS_ROOT` when CUDA provides cuBLAS. Tested toolchains are CUDA 13.3/GCC 13.3 on RTX 5070 Ti 16 GB and CUDA 13.2/GCC 15.2 on RTX 6000D. Both tested GPUs are Blackwell compute capability 12.0; other architectures are outside this build target.

| Model | Guide | Hugging Face | ModelScope |
|---|---|---|---|
| Xiaomi-OCR-0 BF16 | [OCR setup/conversion](docs/xiaomi-ocr.md) | [Model](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer) | [Model](https://modelscope.cn/models/ByronLeeee/Xiaomi-OCR-0-Ninfer) |
| Qwen3-ASR-1.7B-hf BF16 | [ASR setup/conversion](docs/qwen3-asr.md) | [Model](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) | [Model](https://modelscope.cn/models/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) |
| Qwen3-Embedding-0.6B BF16 | [Embedding setup/conversion](docs/qwen3-embedding.md) | [Model](https://huggingface.co/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer) | [Model](https://modelscope.cn/models/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer) |

## Upstream compatibility and performance

| Workload | Unmodified upstream NInfer | NInfer Extended |
|---|---|---|
| Xiaomi-OCR-0 / Qwen3.5-0.8B | Missing the required small-model BF16/vision routes | Native vision, language prefill and decode on both tested GPUs |
| Qwen3-ASR-1.7B | No ASR architecture/frontend | Native audio encoding, language prefill and decode on both tested GPUs |
| Qwen3-Embedding-0.6B | No Qwen3 embedding architecture/frontend | Native BF16 vector inference on RTX 5070 Ti and RTX 6000D |
| Existing Qwen3.8-27B artifact | Supported | Existing large-model routes retained; 6000D control below |

### Xiaomi OCR: upstream-derived compatibility baseline vs optimized operators

Unmodified upstream cannot execute the added OCR, ASR or embedding architectures. This table uses the **initial working BF16 OCR adaptation before operator optimization** as the baseline, with the same artifact and matched toolchain; it is not an unmodified-upstream speed claim. Measurements use the complete Chinese document, 4K context, BF16 KV, 1,024-token prefill chunks and greedy decoding. Each version has two warmups and ten formal bursts in adjacent A-B-B-A cycles. Prefill includes vision and language GPU time; decode is tok/s per request.

| GPU | Requests | Baseline prefill tok/s | Extended prefill tok/s | Prefill change | Baseline decode tok/s | Extended decode tok/s | Decode change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 1 | 20,016 | 22,082 | +10.3% | 406.4 | 418.6 | +3.0% |
| RTX 5070 Ti | 2 | 19,573 | 21,602 | +10.4% | 282.6 | 403.2 | +42.7% |
| RTX 5070 Ti | 4 | 19,480 | 21,538 | +10.6% | 268.9 | 370.8 | +37.9% |
| RTX 6000D | 1 | 35,238 | 37,911 | +7.6% | 514.2 | 541.9 | +5.4% |
| RTX 6000D | 2 | 35,254 | 37,896 | +7.5% | 356.3 | 527.3 | +48.0% |
| RTX 6000D | 4 | 35,349 | 37,876 | +7.1% | 341.2 | 487.1 | +42.7% |

The measured complete-model gains retain exact outputs. Both GPUs pass 225 independent FP64 OCR checks; the subsequent ASR operator iterations also preserve OCR outputs in the 4K/32K, one-/two-/four-request regression tests. [Detailed OCR measurements](docs/xiaomi-ocr-performance.md).

### Qwen3.8-27B: existing NInfer server vs extension on RTX 6000D

The existing pre-rollout NInfer executable and extended executable read the same mixed NVFP4/FP8/BF16/integer artifact, FP8 KV, 16K context and DFlash2 K7. A-B-B-A uses two warmups and six formal bursts per version. The copy workload produces 128 tokens; its draft acceptance is 100%.

| Requests | Prompt tokens | Existing prefill tok/s | Extended prefill tok/s | Existing decode tok/s | Extended decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 2504 | 6,480 | 6,493 | 352.5 | 352.6 |
| 1 | 10256 | 6,415 | 6,416 | 347.2 | 346.9 |
| 4 | 2504 | 6,498 | 6,505 | 295.5 | 295.5 |
| 4 | 10256 | 6,440 | 6,437 | 283.0 | 282.8 |

Changes stay within ±0.21%, with all 120 formal outputs matching. Open-ended single-request writing reaches 116–131 decode tok/s at 26.1% draft acceptance. These operator extensions do not establish a 27B speed gain. This 27B control was measured on 6000D; no 5070 Ti 27B result is claimed.

### Qwen3 Embedding: optimizations after native BF16 support

The native embedding route uses packed causal attention and register-fused SwiGLU. A 4×128 forward needs 228 kernel launches, including 28 attention invocations and no separate SwiGLU activation; the initial supported route needed 340 launches and 112 attention invocations. These are measurements of the supported extension, since upstream has no embedding frontend.

| GPU | Input | Earlier supported NInfer ms | Optimized NInfer ms | Whole-model throughput change |
|---|---|---:|---:|---:|
| RTX 5070 Ti | 4×33 tokens | 3.563 | 3.316 | +7.45% |
| RTX 5070 Ti | 8×17 tokens | 3.489 | 3.236 | +7.82% |
| RTX 6000D | 1×2048 tokens | 23.000 | 21.298 | +7.99% |

Short packed batches reduce unused projection tile work. Long-input attention can stage 64 K/V columns while retaining the original 32-column softmax and compensated value accumulation. Resource-based dispatch selects the wider schedule on 6000D for the measured 2K input; 5070 Ti retains the original narrow GPU instructions. The wider kernel is compiled in a separate non-RDC CUDA module. It adds no global scratch allocation.

Measurements use separate A-B-B-A process windows, 25 warmups and 15 measured forwards per process, for 30 measurements per version. Both GPUs pass 164 embedding and 117 ASR FP64 checks; all 249 tested embedding vectors match the preceding version exactly, and the tested ASR outputs remain token-identical. [Embedding guide](docs/qwen3-embedding.md).

The model repositories report **Transformers vs NInfer Extended** speed and result agreement. OCR and ASR include complete prefill/decode; the ASR cards also report warm latency and inference time/throughput for 60-second audio. Embedding reports complete forward/pooling throughput, vector similarity, retrieval ranking agreement and semantic scores.

---

# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for Qwen3.5 Dense and MoE architectures on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs. The runtime is deliberately specialized: one GPU, one
resident model, and a startup-fixed capacity of one to eight active requests.

Five official artifacts are available. The quick-start commands use Qwen3.8-27B NVFP4.

| Model | Weights | Artifact | Download and model card |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

Each v3 `.ninfer` artifact carries model configuration, encoded weights, logical bindings and
frontend resources. Runtime execution uses those facts with the implemented model and Op
capabilities. You can also [convert your own weights](docs/weight-conversion.md), reuse an official
recipe or choose another supported mixture of formats.

The current engine requires v3 artifacts. Existing official v2 downloads can be
[upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact) without downloading
the weights again.

## Quick start

NInfer requires 64-bit Linux, an NVIDIA GeForce RTX 5090, a CUDA toolkit supporting `sm_120a`,
CMake 3.28 or newer, a C++20 host compiler, Ninja, `pkg-config`, FFmpeg development libraries
(`libavformat`, `libavcodec`, `libavutil`, and `libswscale`), and `libcurl >= 7.85`.
CUDA 13.1 is the validated development toolkit; CMake does not impose a CUDA version floor.
The build rejects CUDA architectures other than `sm_120a`.

Build the product binaries:

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Tests and benchmarks are excluded from the default build. `cmake --preset release` configures
the same product build; `cmake --preset dev` also enables tests and benchmarks and finds a
Python 3 interpreter. Both presets use `build/` and explicitly reset the build options.
Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`.
See [build organization and configuration](docs/maintainer/build-system.md) for details.

There is no install target or packaged binary distribution; run NInfer from its source build tree.
Python tools run independently of CMake; the standalone HBM probe has its own
[build command](tools/README.md#standalone-hbm-probe).

Download the artifact used by this example with the Hugging Face CLI:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Start a long-running text/agent server with two active-request lanes and explicit Device/Host
checkpoint capacity:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

Each request has a 240,000-token logical ceiling. A shared 240,000-token Device KV pool serves
admitted requests; two requests run concurrently when their combined reservations fit. The cache
tiers provide two Device checkpoint slots, eight pinned Host State slots, and 8 GiB of pinned Host
KV beyond the two active StateImages.

Send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

Run a one-shot CLI request with a 32,768-token allocation:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Answer content is written to stdout. Human-readable startup/runtime diagnostics and the CLI-owned
reasoning, timing, throughput, memory, and speculative-decoding report are written to stderr;
reasoning and the result report remain unprefixed product output. On a terminal, weight
materialization uses one transient progress line followed by a compact Engine-ready summary.
Redirected stderr receives persistent readable progress without terminal control sequences. Use
`--log-level debug` for complete startup detail. Option and local input errors remain direct command
diagnostics. Use `--messages FILE` and `--vision` for structured image/video input; see the
[CLI guide](docs/cli.md) and [committed examples](examples/cli/).

## Resource-aware long-context reuse

A reusable prefix checkpoint contains KV and the complete continuation state for its exact prompt
frontier. A Device-resident checkpoint resumes directly. Under pressure, the planner weighs Device
retention, pinned Host State/KV, and eviction by immediate restore work and later reuse cost. Active
requests retain their completion reservations.

See [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
for the algorithm and [Serve TTFT benchmark](tools/bench/ttft/) for public-HTTP coverage of hot
reuse, Host resume, eviction, shared prefixes, scheduling boundaries, and multimodal load.

## Performance

Published measurements use an RTX 5090. The [performance index](docs/performance.md) links to
per-model run records and the [measurement rules](docs/performance/methodology.md). The tables
below are excerpts from those detailed results.

### Concurrent MTP3 decode

Saturated decode used INT8 group-64 KV, CUDA Graphs, MTP3, and one 8,192-token generation per active
request. Throughput uses aggregate committed decode tokens from complete intervals whose actual
decode batch equaled the configured concurrency. Acceptance covers the complete request wave;
these rates are steady decode (tok/s).

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

### Single-request serving

The serial serving corpus used INT8 group-64 KV, CUDA Graphs, a 1,024-token prefill chunk, and five
fixed seeds after warm-up. The table keeps one short-prefill, one extreme-prefill, and one
structured-output MTP3 point for each published profile; the full context and scenario matrices are
linked from each model below.

| Model profile | 7,680-token prefill | 260,096-token prefill | Structured MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,274.7 tok/s | 1,609.7 tok/s | 224.4 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 8,340.4 tok/s | 2,203.1 tok/s | 219.8 tok/s |

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8 rows used temperature
1.0 and presence penalty 0.0. Multimodal evaluation used `--vision` and an 81,920-token context
limit. Text evaluation used 262,144 tokens except Qwen3.8-27B NVFP4, which used 252,928 tokens to
fit the RTX 5090 after weights. Each score is one sample per problem; model cards contain the
correct/total counts and evaluation notes.

## Startup notes

GPU residency is fixed at process startup. `--spec` selects speculative decoding residency, and
`--vision` independently selects Vision residency. Qwen3.6-35B-A3B DFlash can be combined with
Vision; it accelerates generated-text decode after multimodal prefill, not Vision encode itself.

## Docker

Build the runtime image on a host with the NVIDIA Container Toolkit:

```bash
docker build --tag ninfer:local .
```

Mount the downloaded model and run the same example server profile:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## Capabilities and limits

The official artifacts provide the following capabilities, with optional components enabled at startup:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill, exact-batch CUDA Graph decode, and startup-bounded batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16, INT8, FP8, NVFP4, and K8V4 KV storage;
- offline causal-perplexity scoring;
- private and shared exact-prefix reuse with Device/Host State and KV retention;
- model-aware sampling defaults and explicit sampler overrides;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming,
  tools, local response state, token counting, and usage accounting.

The 35B-A3B target additionally supports DFlash with draft windows from one to fifteen for Text and
image/video Vision prompts. Qwen3.8-27B artifacts with the DFlash2 companion weights support
`--spec dflash2 --draft-tokens 7` for the same Text/Vision Engine path, with draft counts 1..15
and either full or optimized proposal heads.

The product boundary remains intentionally small:

- one RTX 5090 and one resident model per Engine;
- a startup-fixed capacity of one to eight active requests with bounded FIFO ingress;
- no request preemption, priority/QoS, active-request swapping, weight offload, multi-GPU, or
  distributed serving;
- one shared startup-fixed KV pool across active requests and retained prefixes;
- model architectures and format/shape combinations use explicitly implemented native paths;
- parsed tool calls are returned to the client; NInfer does not execute tools;
- the in-tree C++ headers are not distributed as an installed SDK.

`--max-context` is each sequence's logical limit. `--kv-capacity` sizes the shared Main Text KV pool
used by active requests and retained prefixes; `auto` resolves the largest legal capacity at
startup from the memory remaining after weights while keeping 1 GiB of sizing headroom. Explicit
capacities remain fixed for the process lifetime.

## Documentation

- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [Perplexity evaluation](docs/perplexity.md)
- [Weight conversion and custom recipes](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT benchmark](tools/bench/ttft/)
- [CLI examples](examples/cli/)
- [Contributing](CONTRIBUTING.md)

Run the relevant `--help` for the exact current option contract.

## Support

NInfer is a personal project that I develop out of interest. If you find it useful and would like
to support its continued development, you can [support the project on Ko-fi](https://ko-fi.com/neroued).

Support is entirely voluntary. It is not a purchase or investment and does not come with financial
returns, promised services or features, or a role in project decisions. The project's direction,
priorities, technical choices, and release schedule remain independently determined by the
maintainer.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
