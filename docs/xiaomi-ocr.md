# Xiaomi-OCR-0 BF16 adaptation

This fork adds Qwen3.5-0.8B support to NInfer and has been tested with
Xiaomi-OCR-0, which is based on Qwen3.5-0.8B-Base. It builds on upstream
`594930e7b609efa4bcea3ae4f24cd9d66b5f224f`.

## Implemented changes

- BF16/A16 projection routes for the checkpoint's 1,024-wide text model,
  including GDN input/gating projections, attention projections, linear-add and
  SwiGLU. Shared input embedding/output weights remain tied. Norms and mathematical
  scalars retain the converter's prescribed representations.
- Shared BF16 projection kernels select GEMV, small-batch SIMT, or MMA from the
  matrix shape and token count. Residual and split-output projections write their
  final destinations directly during both prefill and decode. Fused SwiGLU pairs
  gate/up rows without changing the stored weights; its qualified profiles are
  7,168×1,024 and 14,336×5,120. These paths use the public Ops and their workspace
  capacity queries.
- Text causal attention for head dimension 256, eight query heads and two KV heads;
  BF16 and optional FP8 paged KV support for this geometry.
- Packed vision attention for head dimension 64 and twelve heads, with tiled
  online softmax and segment-aware routing. Explicit vision position-embedding
  ordering and two-axis RoPE preserve the checkpoint's spatial semantics.
- The 6,144-channel causal convolution path handles both prefill and cached decode.
- PCRE2 performs the checkpoint's Unicode pre-tokenization expression; added-token
  metadata is exported from the original tokenizer. `libpcre2-dev` is an additional build dependency.
- A BF16 conversion recipe embeds tokenizer, image/video processor and generation
  resources in the v3 model file. Download it from
  [Hugging Face](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer) or
  [ModelScope](https://modelscope.cn/models/ByronLeeee/Xiaomi-OCR-0-Ninfer).
- The optional Transformers reference tools compile language prefill and vision tensor
  work with `fullgraph=True`. Functional custom-op boundaries call the original
  causal-conv1d/FLA fused kernels. Grid metadata remains outside compilation; official
  Transformers compiled decode and its CUDA Graphs are retained.

## Build and run

Tested GPUs: RTX 5070 Ti 16 GB (WSL2) and RTX 6000D (Linux), both compute capability
12.0. The native build targets `sm_120a`.

Use Linux, CUDA supporting `sm_120a`, a C++20 compiler, CMake >=3.28, Ninja, FFmpeg
development libraries, libcurl >=7.85, pkg-config and PCRE2 development headers/library.
The tested build used CUDA 13.2 and GCC 15.2. The same Linux binary ran on both
GPUs; the WSL run used a compatible runtime library bundle. The Transformers
baseline was compiled separately on each GPU.

```bash
git clone https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b.git
cd ninfer-qwen3.5-0.8b
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

hf download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy
```

The command starts a service with a 32K context and four concurrent requests.
The Transformers comparison uses a 4K context and one request on both engines.
The latest local operator comparison also measures two and four concurrent requests.
Image token counts depend on resolution. BF16 KV works well for this OCR setup;
FP8 KV is available to reduce cache memory at longer contexts.

## Convert original weights

```bash
hf download SeerRay-Lab/Xiaomi-OCR-0 --revision e4d1c4a6804bd9ef342b93d705a73af003e2ef4e \
  --local-dir models/Xiaomi-OCR-0
python tools/xiaomi_ocr/convert.py --model models/Xiaomi-OCR-0 \
  --out models/xiaomi-ocr-0-bf16.ninfer
```

The converter also requires upstream conversion dependencies described in
[weight conversion](weight-conversion.md). The output artifact contains 1,726,110,464
bytes and retains BF16 projection weights.

## Compiled Transformers reference

Validated reference: Python 3.12, Torch 2.14.0+cu132, Transformers 5.17.0,
FLA/fla-core 0.5.2, causal-conv1d 1.7.0. The causal-conv1d extension must be built against
your Torch/CUDA runtime for `sm_120`. The benchmark checks that the fused kernels
are available before running.

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

Compilation runs on the first request for each new shape (`dynamic=False`).
Warm up the input shapes before measuring throughput. The compiled custom ops
support cached inference with equal-length batches. `/health` exposes selected fusion and compilation settings.
Cache storage is initialized outside Dynamo using the Transformers cache APIs, without
running an eager model prefill. This preserves static storage addresses for decode
CUDA Graphs. Each response records compiled language-call counts and actual cache length.

## Reproduce qualification and prefill comparison

```bash
cmake -S . -B build -DNINFER_BUILD_XIAOMI_QUALIFICATION=ON
cmake --build build --target xiaomi-ocr-qualify xiaomi-ocr-qualify-small-ops -j
./build/xiaomi-ocr-qualify
./build/xiaomi-ocr-qualify-small-ops

python tools/xiaomi_ocr/benchmark_prefill.py \
  --model-dir models/Xiaomi-OCR-0 --functional-ops --vision --abba \
  --fixtures-dir /path/to/local/ocr-fixtures --fixtures page
```

Qualification checks represented BF16 inputs against independent naive FP64 math,
including cache casts, causal attention prefill/cached decode, vision attention,
vision RoPE and convolution at state/chunk boundaries. The second executable adds
135 checks of BF16 linear, residual, GDN/attention split, control and SwiGLU projections,
covering T=1, 2, 3, 4, 5, 7, 8, 9, 17, 64 and 65. It also checks D64 vision
attention at tile boundaries, 7,168/9,216 patches, packed segment boundaries and
padded token strides. All 145 checks passed on the local 5070 Ti build. SwiGLU
preserves the established BF16 projection and SiLU rounding boundaries and uses
the repository's A16 relative-L2 and maximum-absolute-error criteria.
The reference benchmark performs A-B-B-A phases, excluding one
warmup per fixture per phase, keeping compiled decode in both variants. The published measurement used
two synthetic pages and one official example; dense-equations is capped at 256 output
tokens and therefore is a prefix comparison. See [measured results](xiaomi-ocr-performance.md).

For the benchmark, supply your images and a local `manifest.json` such as
`[{"id":"page","path":"page.png","prompt":"Extract the text in the image."}]`
and the corresponding image.

## Related implementations checked

GitHub repository and code searches on 2026-10-06 found no equivalent
Xiaomi-OCR-0 NInfer adaptation. [Upstream NInfer](https://github.com/Neroued/ninfer)
is the engine base. Hardware forks such as [ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090)
and [ninfer-3090](https://github.com/Don-Chad/ninfer-3090) target other GPU architectures.
The [official Xiaomi OCR project](https://github.com/SeerRay-Lab/Xiaomi-OCR-0)
provides the original model and application pipeline.
