---
license: apache-2.0
base_model:
- Qwen/Qwen3-ASR-1.7B-hf
- Qwen/Qwen3-ForcedAligner-0.6B
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
- forced-alignment
- word-timestamps
---

# Qwen3-ASR-1.7B BF16 for NInfer

This BF16 conversion of [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) runs with the **main branch of [ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended)**. Audio encoding, language prefill and decode run in the native C++/CUDA engine. The artifact includes the tokenizer, processor and chat template, with all 707 source parameters preserved byte for byte.

Also included: a native BF16 Qwen3-ForcedAligner-0.6B artifact for word timestamps. See the combined ASR/alignment workflow and measurements below.

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

## Build and transcribe

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-asr ninfer-align -j
pip install -r tools/qwen3_asr/requirements.txt
pip install huggingface_hub
hf download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer qwen3-asr-1.7b-bf16.ninfer --local-dir models
python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --audio recording.wav --out transcription.json
```

Build dependencies: 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl, PCRE2, cuDNN 9 and cuBLAS. Omit `CUBLAS_ROOT` if cuBLAS is installed with CUDA. Supply one to four mono PCM16 WAV files at 16 kHz. Defaults: BF16 weights/KV, 4K context, up to four lanes and 1,024 output tokens. Language prefill uses FP32 probability arithmetic by default.

[Build, conversion and API guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-asr.md)

## Prefill and decode performance

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

## Word timestamps with Qwen3-ForcedAligner-0.6B

This repository includes `qwen3-forced-aligner-0.6b-bf16.ninfer` alongside the main ASR artifact. Both run with the `main` branch of [ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended).

The ASR model produces the transcript and detected language. ForcedAligner then takes the same audio and transcript and assigns start/end times to each word. Audio encoding, language forward and timestamp classification run in one native BF16 pass with CUDA Graphs and no KV cache. Conversion preserves all 708 source BF16 parameters byte for byte. The timestamp step is **80 ms**.

Supported languages: Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian and Spanish.

### Run ASR and alignment together

The commands below transcribe the recording, read its text and detected language, and generate word timestamps. Both CPU frontends share one Transformers 5.17.0 environment; GPU inference runs in NInfer. Configure the build dependencies above before a first installation.

```bash
# Run from the ninfer-extended repository root.
cmake --build build --target ninfer-asr ninfer-align -j
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/qwen3_asr/requirements.txt huggingface_hub

.venv/bin/hf download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer \
  qwen3-asr-1.7b-bf16.ninfer qwen3-forced-aligner-0.6b-bf16.ninfer --local-dir models
ffmpeg -i recording.wav -ac 1 -ar 16000 -c:a pcm_s16le recording-16k.wav

.venv/bin/python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --audio recording-16k.wav --warmups 0 --repeats 1 --out transcription.json

TRANSCRIPT="$(.venv/bin/python -c 'import json; print(json.load(open("transcription.json", encoding="utf-8"))["runs"][-1]["text"][0])')"
LANGUAGE="$(.venv/bin/python -c 'import json; print(json.load(open("transcription.json", encoding="utf-8"))["runs"][-1]["language"][0])')"
.venv/bin/python tools/qwen3_forced_aligner/align.py \
  --artifact models/qwen3-forced-aligner-0.6b-bf16.ninfer --engine build/apps/ninfer-align \
  --audio recording-16k.wav --text "$TRANSCRIPT" --language "$LANGUAGE" \
  --warmups 0 --repeats 1 --out timestamps.json

.venv/bin/python -c 'import json; r=json.load(open("timestamps.json", encoding="utf-8")); print(json.dumps(r["cases"][0]["runs"][-1]["words"][0], ensure_ascii=False, indent=2))'
```

`transcription.json` contains the recognition result. Word timestamps are in `timestamps.json` at `cases[0].runs[-1].words[0]`: each entry contains `text`, `start` and `end`, in seconds. You can also pass an edited transcript directly with `--text`. For multiple recordings, supply matching `--audio`, `--text` and `--language` lists; ASR accepts 1–4 samples and the aligner accepts 1–8.

### Alignment timing and timestamp agreement

Single-sample medians from seven measurements after warmup. Synchronized CUDA events cover the complete audio encoder, language forward and timestamp head, including scheduling gaps between stages; model loading, CPU preprocessing and initial Graph capture are excluded. The official reference uses `qwen-asr 0.0.6`, Transformers 4.57.6 and PyTorch 2.10.0+cu130 with BF16 GPU weights and fused SDPA, without CPU offload or MATH attention fallback. This alignment reference runs eagerly; the main ASR reference above uses compilation and decode Graphs.

| GPU | Audio | Transformers ms | NInfer ms | Speedup |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese 4.20 s | 71.336 | 4.449 | 16.03× |
| RTX 5070 Ti | English 15.05 s | 68.464 | 10.283 | 6.66× |
| RTX 5070 Ti | English 60.205 s | 59.146 | 33.970 | 1.74× |
| RTX 5070 Ti | Chinese 0.75 s | 74.674 | 3.530 | 21.15× |
| RTX 6000D | Chinese 4.20 s | 10.280 | 3.704 | 2.78× |
| RTX 6000D | English 15.05 s | 10.920 | 6.157 | 1.77× |
| RTX 6000D | English 60.205 s | 24.070 | 23.536 | 1.02× |
| RTX 6000D | Chinese 0.75 s | 9.118 | 3.276 | 2.78× |

The 60.205-second input repeats a fixed English recording. Alignment emits all timestamps in one forward pass.

| GPU | Timestamp boundaries exactly matching the official reference | Agreement | Maximum difference |
| --- | --- | --- | --- |
| RTX 5070 Ti | 400 / 400 | 100.00% | 0 ms |
| RTX 6000D | 399 / 400 | 99.75% | 80 ms |

Agreement covers single-sample boundaries from the four cases above. Repeated Graph execution and eager execution agree for a fixed input geometry. Changing batch geometry can move a classifier tie by one 80 ms step.

Artifact: `qwen3-forced-aligner-0.6b-bf16.ninfer` · 1,840,359,257 bytes · BF16 · Apache-2.0. SHA256: `45234e54ba66833e846e4d3d30eca06c9f95833a35350508c667432a90143473`.

[Alignment guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-forced-aligner.md) · [Alignment measurements](forced-alignment-results.json) · [Alignment artifact manifest](forced-aligner-manifest.json)

## Artifact and license

`qwen3-asr-1.7b-bf16.ninfer` · 4,087,720,481 bytes · BF16 · Apache-2.0

SHA256: `50d9d8c2ed690b80f5acbf94c66b344ee7fd12cf79dfe670ad10e96b5b411eea`

[Measurements](gpu-results.json) · [Artifact manifest](artifact-manifest.json)
