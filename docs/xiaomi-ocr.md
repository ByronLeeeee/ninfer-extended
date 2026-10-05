# Xiaomi-OCR-0 BF16 adaptation

This fork adds Qwen3.5-0.8B architecture support to NInfer, validated with
Xiaomi-OCR-0, which is trained from Qwen3.5-0.8B-Base. Standalone Base weights
have not been separately benchmarked. The validated
source base is upstream `594930e7b609efa4bcea3ae4f24cd9d66b5f224f`.
The restored source archive matches all 1,127 archived files at that revision.
The original upstream README and Apache-2.0 license are retained.

## Implemented changes

- BF16/A16 projection routes for the checkpoint's 1,024-wide text model,
  including GDN input/gating projections, attention projections, linear-add and
  SwiGLU. Shared input embedding/output weights remain tied. Norms and mathematical
  scalars retain the converter's prescribed representations.
- Text causal attention for head dimension 256, eight query heads and two KV heads;
  BF16 and optional FP8 paged KV support for this geometry.
- Packed vision attention for head dimension 64 and twelve heads, with tiled
  online softmax and segment-aware routing. Explicit vision position-embedding
  ordering and two-axis RoPE preserve the checkpoint's spatial semantics.
- The 6,144-channel causal convolution path handles both prefill and cached decode.
- PCRE2 performs the checkpoint's Unicode pre-tokenization expression; added-token
  metadata is exported from the original effective tokenizer instead of inventing
  a replacement vocabulary. `libpcre2-dev` is an additional build dependency.
- A BF16 conversion recipe embeds tokenizer, image/video processor and generation
  resources in the v3 artifact. The artifact is distributed separately on Hugging Face.
- The optional Transformers reference tools compile language prefill and vision tensor
  work with `fullgraph=True`. Functional custom-op boundaries call the original
  causal-conv1d/FLA fused kernels. Grid metadata remains outside compilation; official
  Transformers compiled decode is retained. This is a Python reference optimization,
  separate from the native C++/CUDA engine and its decode CUDA Graphs.

## Build and run

Validated GPUs: RTX 5070 Ti 16 GB (WSL2) and RTX 6000D (Linux), both compute capability
12.0. The native build targets `sm_120a`. Other Blackwell devices are not automatically
qualified: B200/GB200 use a different architecture. This fork does not claim compatibility
with every Blackwell product or all upstream model paths.

Use Linux, CUDA supporting `sm_120a`, a C++20 compiler, CMake >=3.28, Ninja, FFmpeg
development libraries, libcurl >=7.85, pkg-config and PCRE2 development headers/library.
CUDA 13.2 and GCC 15.2 were used for the complete clean build; that Linux binary was
also executed on the 5070 Ti through a private compatible runtime. WSL's system libc
was not replaced. The Python baseline compiled on each GPU independently.

```bash
git clone https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b.git
cd ninfer-qwen3.5-0.8b
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Download xiaomi-ocr-0-bf16.ninfer from the model repository linked in README.
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy
```

32K context and four lanes are a serving example; the measured comparison uses 4K and
one active request on both engines. No 256K throughput claim is made. Each image contributes
resolution-dependent vision tokens; a 256K context is generally unnecessary for a single
OCR page. BF16 KV is the recommended example. FP8 KV can reduce cache memory for longer
contexts but should be evaluated separately for output differences.

## Convert original weights

```bash
hf download SeerRay-Lab/Xiaomi-OCR-0 --revision e4d1c4a6804bd9ef342b93d705a73af003e2ef4e \
  --local-dir models/Xiaomi-OCR-0
python tools/xiaomi_ocr/convert.py --model models/Xiaomi-OCR-0 \
  --out models/xiaomi-ocr-0-bf16.ninfer
```

The converter also requires upstream conversion dependencies described in
[weight conversion](weight-conversion.md). The output artifact contains 1,726,110,464
bytes in the validated export. The source checkpoint has not been retrained or newly
quantized to integer/FP4 weights.

## Compiled Transformers reference

Validated reference: Python 3.12, Torch 2.14.0+cu132, Transformers 5.17.0,
FLA/fla-core 0.5.2, causal-conv1d 1.7.0. The causal-conv1d extension must be built against
your exact Torch/CUDA runtime for `sm_120`. Missing fused kernels fail the benchmark;
there is no silent acceptance of the slow fallback as the optimized baseline.

```bash
export XIAOMI_OCR_MODEL_PATH="$PWD/models/Xiaomi-OCR-0"
export XIAOMI_REQUIRE_FUSED_LINEAR=1
export XIAOMI_COMPILE_PREFILL=1
export XIAOMI_COMPILE_DECODE=1
export XIAOMI_COMPILE_CACHE_FLOOR=4096
export XIAOMI_BENCH_TIMING_MODE=wall
export XIAOMI_TORCH_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
python -m uvicorn baseline_server:app --app-dir tools/xiaomi_ocr \
  --host 127.0.0.1 --port 8081
```

Compilation is lazy. The first request for a new shape includes compilation and must
be excluded from steady-state throughput. Vision and language regions specialize on
shape (`dynamic=False`); new shapes can require compilation. The compiled custom-op
boundaries support cached, equal-length inference, not training or packed variable-length
GDN batches. `/health` exposes selected fusion and compilation settings.
Cache storage is initialized outside Dynamo using the Transformers cache APIs, without
running an eager model prefill. This preserves static storage addresses for decode
CUDA Graphs. Each response records compiled language-call counts and actual cache length.

## Reproduce qualification and prefill comparison

```bash
cmake -S . -B build -DNINFER_BUILD_XIAOMI_QUALIFICATION=ON
cmake --build build --target xiaomi-ocr-qualify -j
./build/xiaomi-ocr-qualify

python tools/xiaomi_ocr/benchmark_prefill.py \
  --model-dir models/Xiaomi-OCR-0 --functional-ops --vision --abba \
  --fixtures-dir /path/to/local/ocr-fixtures --fixtures page
```

Qualification checks represented BF16 inputs against independent naive FP64 math,
including cache casts, causal attention prefill/cached decode, vision attention,
vision RoPE and convolution at state/chunk boundaries. All ten checks passed on the
complete fork build. The reference benchmark performs A-B-B-A phases, excluding one
warmup per fixture per phase, keeping compiled decode in both variants. The published measurement used
two synthetic pages and one official example; dense-equations is capped at 256 output
tokens and therefore is a prefix comparison. See [measured results](xiaomi-ocr-performance.md).

Images and transcriptions are retained locally and are not distributed in this fork.
Supply a local `manifest.json` such as
`[{"id":"page","path":"page.png","prompt":"Extract the text in the image."}]`
and the corresponding image. Different images produce different timings from the table.

## Related implementations checked

Search on 2026-10-06 found no equivalent Xiaomi-OCR-0 NInfer adaptation in GitHub
repository/code search. This is a bounded search result, not proof that no private or
unindexed implementation exists. [Upstream NInfer](https://github.com/Neroued/ninfer)
is the engine base. Hardware forks such as [ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090)
and [ninfer-3090](https://github.com/Don-Chad/ninfer-3090) target other GPU architectures.
The [official Xiaomi OCR project](https://github.com/SeerRay-Lab/Xiaomi-OCR-0)
provides the model and application pipeline, rather than this NInfer artifact/shape adaptation.
