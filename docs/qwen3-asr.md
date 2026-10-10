# Qwen3-ASR-1.7B BF16

NInfer runs the audio encoder and Qwen3 language decoder from
[Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
through the native C++ `Engine`. The converted file contains the original BF16
weights, tokenizer, processor configuration, and chat template. Matrix and bias
row concatenation preserves all original weight bytes.

## Build

In addition to the normal NInfer dependencies, ASR needs cuDNN 9 and cuBLAS.
Use a CUDA toolkit supporting `sm_120a`.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-asr -j
```

`CUDNN_ROOT` contains `include/cudnn.h` and `lib/libcudnn.so.9`.
`CUBLAS_ROOT` contains `lib/libcublas.so.13`; omit it when cuBLAS is installed
with the CUDA toolkit. Python is used for conversion and the CPU audio/token
frontend. ASR and forced alignment share the native Transformers 5.17.0 CPU
frontend environment:

```bash
python -m pip install -r tools/qwen3_asr/requirements.txt
```

## Convert and transcribe

```bash
python tools/qwen3_asr/convert.py \
  --model /path/to/Qwen3-ASR-1.7B-hf --out qwen3-asr-1.7b-bf16.ninfer

python tools/qwen3_asr/transcribe.py \
  --artifact qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --audio recording.wav --out transcription.json
```

For a single production pass, use `--warmups 0`. The default remains one
unmeasured warmup for benchmark use; `--repeats` controls measured runs.
Official processor context and forced language can be supplied directly:

```bash
python tools/qwen3_asr/transcribe.py \
  --artifact qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --audio recording.wav --out transcription.json --warmups 0 \
  --language Chinese --prompt 'Financial news in Mandarin.' \
  --hotwords 交易 停滞
```

Omit `--language` for automatic language detection. Context and hotwords use
the official processor's system prompt. Parsed JSON retains a `text` list per
run and adds a corresponding `language` list for all supplied audio lanes.

Supply one to four mono PCM16 WAV files at 16 kHz. For other audio formats:

```bash
ffmpeg -i recording.mp3 -ac 1 -ar 16000 -c:a pcm_s16le recording.wav
```

Defaults are BF16 weights, BF16 KV, 4,096 context tokens, a 1,024-token output
budget, and audio/prefill/decode CUDA Graphs. Greedy decoding stops at the
model's EOS tokens. The JSON result contains transcription text, token IDs,
stage timings, and Engine-owned allocation sizes. CPU log-mel extraction and
token decoding are separate from native GPU inference.

## Native API

Create an `Engine` with `EnginePurpose::SpeechRecognition`. Pass one to four
`SpeechFeatures` records to `Engine::transcribe_features()`.
Set `EngineOptions::max_context = 4096`, `max_concurrency = 4`, and
`kv_cache = KvCacheStorage::BFloat16` for the configuration measured below.
Each record supplies CPU FP32 log-mel features `[128,frames]`, a binary I32
valid-frame mask, and the original processor's audio-token prompt. Frames are
padded to a multiple of 100. The API supports the qualified 1.7B configuration
with 28 text layers, a 2,048-wide text decoder, and a 24-layer audio encoder.

`SpeechRunOptions` selects the native BF16 projection Ops or a cuBLAS reference
route. Audio convolution uses cuDNN; biased audio projections use cuBLAS with
exact-erf GELU and the model's BF16 rounding. Native text Ops handle projection,
SwiGLU, residual updates, Q/K normalization and RoPE, and causal attention.
The public Ops select their implementations by geometry and column extent.

The optimized routes include narrower row tiles for short prefill matrices,
wider column tiles for selected QKV and gate/up extents, fused projection plus
residual updates, D128 Tensor Core causal prefill, and partitioned KV attention
for one to four decode lanes. The [12288,2048] prefill SwiGLU route retains
FP32 projection staging through SiLU/multiplication before its BF16 output cast.
Weights and KV precision are unchanged. FP32 probability accumulation is the
default language prefill route. `SpeechRunOptions::causal_tensorcore_prefill`
and the CLI/frontend flag `--tensorcore-prefill` select the optional Tensor Core
route, which uses a BF16 probability plus a BF16 residual. Both routes execute
on the GPU with CUDA Graphs; this option changes prefill arithmetic only.

For two to four ASR lanes, language prefill packs the actual prompt rows for
shared projection, normalization and FFN calls. Attention retains each sample's
FP32 probability path, local RoPE positions and separate KV storage. The
first-token head gathers each sample's last row and runs once for the batch.
Workspace is sized from the total prompt rows before Graph capture. Single-lane
ASR and forced alignment retain their existing execution paths.

Audio convolution executes at most 32 independent 100-frame chunks per call.
Each cuDNN plan retains its reported math type and is checked against the
128 MiB convolution workspace using the actual workspace query. This keeps
long recordings and multi-lane batches on the same bounded execution path.

## Operator qualification

```bash
cmake -S . -B build -G Ninja -DNINFER_BUILD_ASR_QUALIFICATION=ON \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target asr-qualify -j
build/asr-qualify
build/asr-qualify --transforms
build/asr-qualify --linear
```

Floating-point checks use independent FP64 attention, projection, SwiGLU,
normalization, and RoPE oracles. Layout changes, cache writes, and argmax use
exact oracles. The checks cover short and long extents, partial tiles,
one/two/three/four-lane caches, empty KV partitions, EOS tie handling, and the
existing D64/H12 attention route.

## Performance and accuracy

## Transcription speed: audio seconds processed per second

On 60-second English audio, NInfer processes **62.5 seconds of audio per second** on RTX 5070 Ti and **87.7 seconds per second** on RTX 6000D. Four 30-second English recordings reach **197.8** and **299.0 audio seconds per second**, respectively.

Audio seconds/s = total input audio duration ÷ complete warm inference time. For example, 60 seconds of audio transcribed in 0.684 seconds is 87.7 audio seconds/s. The measurement includes audio encoding, prefill, the full transcript generation and synchronization. Model loading, CPU feature extraction and first Graph capture are excluded. Four-lane results sum the duration of all four recordings. The 60-second inputs repeat fixed clips; recognition quality is evaluated separately below.

### RTX 5070 Ti

| Input | Lanes | TF audio seconds/s | NInfer audio seconds/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| English 15.05 s | 1 | 18.9 | **60.5** | 3.21× |
| Chinese 4.20 s | 1 | 20.0 | **73.8** | 3.69× |
| English 60 s | 1 | 19.9 | **62.5** | 3.14× |
| Chinese 60 s | 1 | 35.7 | **107.1** | 3.00× |
| English 15.05 s × 4 | 4 | 52.0 | **194.5** | 3.74× |
| Chinese 4.20 s × 4 | 4 | 56.8 | **206.3** | 3.63× |
| English 30 s × 4 | 4 | 54.1 | **197.8** | 3.66× |
| Mixed English/Chinese × 4 | 4 | 26.5 | **111.2** | 4.19× |

### RTX 6000D

| Input | Lanes | TF audio seconds/s | NInfer audio seconds/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| English 15.05 s | 1 | 27.7 | **86.1** | 3.11× |
| Chinese 4.20 s | 1 | 33.7 | **104.0** | 3.09× |
| English 60 s | 1 | 29.0 | **87.7** | 3.03× |
| Chinese 60 s | 1 | 53.8 | **153.1** | 2.85× |
| English 15.05 s × 4 | 4 | 81.4 | **299.5** | 3.68× |
| Chinese 4.20 s × 4 | 4 | 100.3 | **352.2** | 3.51× |
| English 30 s × 4 | 4 | 84.3 | **299.0** | 3.55× |
| Mixed English/Chinese × 4 | 4 | 41.3 | **165.2** | 4.00× |

### Prefill and decode performance


BF16 weights and KV, 4,096 context tokens per lane, greedy decoding. Transformers uses GPU audio encoding, fullgraph language compilation and decode CUDA Graphs. NInfer uses CUDA Graphs for audio, language prefill and decode. Results are warm medians. Toolchains: CUDA 13.3/GCC 13.3 on 5070 Ti and CUDA 13.2/GCC 15.2 on 6000D; Transformers 5.17.0 and PyTorch 2.14.0+cu132.

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
| English 15.05 s | 1 | 13.487 | 10.544 | 1.28× | 93.3 | 294.8 | 3.16× |
| Chinese 4.20 s | 1 | 10.779 | 6.368 | 1.69× | 92.3 | 296.7 | 3.22× |
| English 60 s | 1 | 32.995 | 32.588 | 1.01× | 92.2 | 283.1 | 3.07× |
| English 15.05 s × 4 | 4 | 34.437 | 29.374 | 1.17× | 278.9 | 1129.5 | 4.05× |
| Chinese 4.20 s × 4 | 4 | 18.098 | 12.534 | 1.44× | 278.6 | 1153.0 | 4.14× |
| English 30 s × 4 | 4 | 58.855 | 57.538 | 1.02× | 278.5 | 1096.5 | 3.94× |
| Mixed English/Chinese × 4 | 4 | 50.569 | 18.693 | 2.71× | 143.6 | 585.4 | 4.08× |

## Inference latency and 60-second audio

Latency measures a complete warm model call, including audio encoding, prefill, generation and synchronization. Model loading, CPU feature extraction and initial Graph capture are excluded. Short-clip latency:

| GPU | Input | Transformers ms | NInfer ms | Inference speedup |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | English 15.05 s | 798.0 | 248.8 | 3.21× |
| RTX 5070 Ti | Chinese 4.20 s | 210.0 | 56.9 | 3.69× |
| RTX 6000D | English 15.05 s | 543.1 | 174.9 | 3.11× |
| RTX 6000D | Chinese 4.20 s | 124.8 | 40.4 | 3.09× |

The 60-second inputs concatenate fixed clips for benchmarking. Audio seconds per inference second equals 60 divided by measured warm inference time.

| GPU | 60 s input | TF inference s | NInfer inference s | TF audio s/s | NInfer audio s/s | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | English 60 s | 3.011 | 0.960 | 19.9 | 62.5 | 3.14× |
| RTX 5070 Ti | Chinese 60 s | 1.679 | 0.560 | 35.7 | 107.1 | 3.00× |
| RTX 6000D | English 60 s | 2.071 | 0.684 | 29.0 | 87.7 | 3.03× |
| RTX 6000D | Chinese 60 s | 1.115 | 0.392 | 53.8 | 153.1 | 2.85× |

## Recognition quality and output agreement

Single-lane quality is measured on 73 real labelled English recordings: 481.03 seconds and 1,150 gold words. WER normalization removes case, punctuation and extra whitespace while retaining apostrophes; CER excludes spaces.

| GPU | TF WER | NInfer WER | TF CER | NInfer CER | Raw text/token agreement | Normalized text agreement | Token edit rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 3.9130% | 3.8261% | 1.3553% | 1.3166% | 98.63% | 98.63% | 0.0573% |
| RTX 6000D | 3.5652% | 3.5652% | 1.2585% | 1.2585% | 97.26% | 100.00% | 0.1720% |

On 5070 Ti, Transformers has 45 word errors and NInfer 44; the differing `has/is` choice matches the gold transcript in NInfer. On 6000D both have 41 errors and all normalized texts agree; raw differences are capitalization and punctuation. Both GPUs pass independent FP64 operator checks.

## Artifact and license

`qwen3-asr-1.7b-bf16.ninfer` · 4,087,720,481 bytes · BF16 · Apache-2.0

SHA256: `50d9d8c2ed690b80f5acbf94c66b344ee7fd12cf79dfe670ad10e96b5b411eea`

[Measurements](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer/blob/main/gpu-results.json) · [Artifact manifest](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer/blob/main/artifact-manifest.json)
