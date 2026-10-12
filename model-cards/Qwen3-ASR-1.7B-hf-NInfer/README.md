# Qwen3-ASR-1.7B · NInfer

[简体中文](README_zh-CN.md)

## Model overview

A BF16 NInfer version of [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), supporting automatic language identification and transcription in 30 languages and 22 Chinese dialects. It runs with **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)**, with native C++/CUDA execution for audio encoding, prefill and decode.

This repository also provides a 0.6B draft model for faster inference and Qwen3-ForcedAligner-0.6B for word timestamps. The main model accepts batches of up to four audio files, automatic or specified language, context prompts and hotwords.

## Performance

**Audio s/s** is the number of audio seconds processed per inference second: total input duration ÷ inference time. Results are warm-run BF16 medians, including audio encoding, prefill, complete transcription and synchronization, excluding model loading and CPU feature extraction. Transformers uses compiled prefill and CUDA Graph for decode.

### Single 1.7B model

| GPU | Audio | Transformers (ms) | NInfer (ms) | Transformers audio s/s | NInfer audio s/s | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | English 15.05 s | 798.0 | **248.8** | 18.9 | **60.5** | 3.21× |
| RTX 5070 Ti | Chinese 4.20 s | 210.0 | **56.9** | 20.0 | **73.8** | 3.69× |
| RTX 6000D | English 15.05 s | 543.1 | **174.9** | 27.7 | **86.1** | 3.11× |
| RTX 6000D | Chinese 4.20 s | 124.8 | **40.4** | 33.7 | **104.0** | 3.09× |

The 60-second samples repeat a fixed recording:

| GPU | Audio | Transformers time (s) | NInfer time (s) | Transformers audio s/s | NInfer audio s/s | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | English | 3.011 | **0.960** | 19.9 | **62.5** | 3.14× |
| RTX 5070 Ti | Chinese | 1.679 | **0.560** | 35.7 | **107.1** | 3.00× |
| RTX 6000D | English | 2.071 | **0.684** | 29.0 | **87.7** | 3.03× |
| RTX 6000D | Chinese | 1.115 | **0.392** | 53.8 | **153.1** | 2.85× |

### Four-audio batches

| GPU | Audio | Transformers audio s/s | NInfer audio s/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | English 30 s × 4 | 54.1 | **197.8** | 3.66× |
| RTX 5070 Ti | Chinese 4.20 s × 4 | 56.8 | **206.3** | 3.63× |
| RTX 6000D | English 30 s × 4 | 84.3 | **299.0** | 3.55× |
| RTX 6000D | Chinese 4.20 s × 4 | 100.3 | **352.2** | 3.51× |

### Prefill and decode

Prefill includes audio encoding, language processing and the first token. Decode starts at the second token and includes EOS; batch values are aggregate tok/s.

| GPU | Audio | Transformers prefill (ms) | NInfer prefill (ms) | Transformers decode (tok/s) | NInfer decode (tok/s) | Decode speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | English 15.05 s | 39.416 | 16.456 | 68.2 | **213.4** | 3.13× |
| RTX 5070 Ti | Chinese 4.20 s | 41.861 | 8.639 | 66.1 | **214.8** | 3.25× |
| RTX 5070 Ti | English 60 s | 74.651 | 59.059 | 66.8 | **207.8** | 3.11× |
| RTX 5070 Ti | English 30 s × 4 | 130.738 | 120.680 | 186.0 | **786.0** | 4.23× |
| RTX 6000D | English 15.05 s | 13.487 | 10.544 | 93.3 | **294.8** | 3.16× |
| RTX 6000D | Chinese 4.20 s | 10.779 | 6.368 | 92.3 | **296.7** | 3.22× |
| RTX 6000D | English 60 s | 32.995 | 32.588 | 92.2 | **283.1** | 3.07× |
| RTX 6000D | English 30 s × 4 | 58.855 | 57.538 | 278.5 | **1096.5** | 3.94× |

### 0.6B draft acceleration

The 1.7B target runs with a 0.6B draft: K=5 on RTX 5070 Ti and K=3 on RTX 6000D.

| GPU | Audio | K | Transformers (ms) | NInfer (ms) | Transformers audio s/s | NInfer audio s/s | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | English 15.05 s | 5 | 798.0 | **162.0** | 18.9 | **92.9** | 4.93× |
| RTX 5070 Ti | Chinese 4.20 s | 5 | 210.0 | **42.9** | 20.0 | **98.0** | 4.89× |
| RTX 6000D | English 15.05 s | 3 | 543.1 | **146.4** | 27.7 | **102.8** | 3.71× |
| RTX 6000D | Chinese 4.20 s | 3 | 124.8 | **38.6** | 33.7 | **108.9** | 3.23× |

Complete prefill and decode with drafting enabled:

| GPU | Audio | Transformers prefill (ms) | NInfer prefill (ms) | Transformers decode (tok/s) | NInfer decode (tok/s) | Decode speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | English 15.05 s | 39.416 | 26.193 | 68.2 | **331.0** | 4.86× |
| RTX 5070 Ti | Chinese 4.20 s | 41.861 | 13.187 | 66.1 | **318.7** | 4.82× |
| RTX 6000D | English 15.05 s | 13.487 | 16.646 | 93.3 | **375.4** | 4.02× |
| RTX 6000D | Chinese 4.20 s | 10.779 | 9.984 | 92.3 | **356.0** | 3.86× |

A 60-second mixed recording made from seven distinct clips and silence intervals:

| GPU | K | Time for 60 s audio (s) | Audio s/s | Prefill (ms) | Decode (tok/s) | Acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 5 | **0.5518** | **108.7** | 99.40 | 345.5 | 98.57% |
| RTX 6000D | 3 | **0.5958** | **100.7** | 84.55 | 327.8 | 99.21% |

### Four-lane MTP

RTX 5070 Ti, BF16 1.7B target + 0.6B draft, K=5. Native build verified on 2026-10-13; six warm-run medians, including both models' audio encoding, prefill and complete decode. Loading and CPU feature extraction are excluded.

| Input | Ordinary four-lane ms | MTP four-lane ms | Speedup |
|---|---:|---:|---:|
| Mixed English/Chinese × 4 | 276.9 | 208.4 | 1.33× |
| English 30 s × 4 | 548.4 | 452.1 | 1.21× |
| Chinese 4.2 s × 4 | 70.0 | 65.4 | 1.07× |

[Four-lane measurements](../../docs/results/qwen3-asr-draft-batch4.json).

### ForcedAligner latency

BF16, one request, including audio encoding, language processing and timestamp classification. The Transformers reference uses eager execution; NInfer uses CUDA Graph.

| GPU | Audio | Transformers eager (ms) | NInfer (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | Chinese 4.20 s | 71.336 | **4.449** | 16.03× |
| RTX 5070 Ti | English 15.05 s | 68.464 | **10.283** | 6.66× |
| RTX 6000D | Chinese 4.20 s | 10.280 | **3.704** | 2.78× |
| RTX 6000D | English 15.05 s | 10.920 | **6.157** | 1.77× |

## Accuracy

73 human-labelled English recordings, totalling 481.03 seconds and 1,150 reference words. WER and CER are word and character error rates; lower is better. WER uses normalized words; CER excludes spaces.

| GPU | Transformers WER | NInfer WER | Transformers CER | NInfer CER |
| --- | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 3.9130% | **3.8261%** | 1.3553% | **1.3166%** |
| RTX 6000D | 3.5652% | **3.5652%** | 1.2585% | **1.2585%** |

### Agreement with Transformers

| GPU | Exact text/token agreement | Normalized text agreement | Token edit difference |
| --- | ---: | ---: | ---: |
| RTX 5070 Ti | 98.63% | 98.63% | 0.0573% |
| RTX 6000D | 97.26% | 100.00% | 0.1720% |

### Draft decoding agreement

Complete output tokens compared with the single 1.7B model on the same 73 recordings.

| GPU | K | Exact tokens vs. single model | Draft acceptance |
| --- | ---: | ---: | ---: |
| RTX 5070 Ti | 5 | **100% (73/73)** | 85.53% |
| RTX 6000D | 3 | **100% (73/73)** | 88.58% |

### Timestamp agreement

400 time boundaries across four recordings, compared with the official ForcedAligner reference.

| GPU | Timestamp agreement | Maximum difference |
| --- | ---: | ---: |
| RTX 5070 Ti | 100.00% (400/400) | 0 ms |
| RTX 6000D | 99.75% (399/400) | 80 ms |

## Usage

### Build

Requirements: Linux or WSL2, Python 3.11, CUDA with `sm_120a` support, cuDNN 9 and cuBLAS. ASR and alignment share one Python environment. See the [build guide](https://github.com/ByronLeeeee/ninfer-extended#build) for dependencies.

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-asr ninfer-align -j
pip install -r tools/qwen3_asr/requirements.txt
```

Use `main` on RTX 5070 Ti. The RTX 6000D K=3 results use revision [`64395f47`](https://github.com/ByronLeeeee/ninfer-extended/commit/64395f477fa8e5134f11edc5eb97697c47656d82); to reproduce this configuration, run `git checkout 64395f477fa8e5134f11edc5eb97697c47656d82` before building.

### Download and transcribe

```bash
pip install huggingface_hub
hf download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer \
  qwen3-asr-1.7b-bf16.ninfer --local-dir models
ffmpeg -i recording.wav -ac 1 -ar 16000 -c:a pcm_s16le recording-16k.wav
python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer \
  --engine build/apps/ninfer-asr --audio recording-16k.wav \
  --warmups 0 --repeats 1 --out transcription.json
```

`transcription.json` contains text and detected language. Pass 1–4 files to `--audio`. Language detection is automatic; optional controls include `--language English`, `--prompt "This is a medical lecture"` and `--hotwords aspirin hemoglobin`.

### Enable draft acceleration

```bash
hf download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer \
  qwen3-asr-0.6b-bf16.ninfer --local-dir models
python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --draft-artifact models/qwen3-asr-0.6b-bf16.ninfer --draft-tokens 5 \
  --audio recording-16k.wav --warmups 0 --repeats 1 --out transcription.json
```

Set `--draft-tokens 3` on RTX 6000D. Draft acceleration handles one audio file; four-audio batches use the main model. Engine allocations for a 60-second input are approximately **4.77 GiB** for the single model and **7.17 GiB** for target plus draft.

### Add word timestamps

Transcribe first, then align the resulting text:

```bash
hf download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer \
  qwen3-forced-aligner-0.6b-bf16.ninfer --local-dir models
TRANSCRIPT="$(python -c 'import json; print(json.load(open("transcription.json", encoding="utf-8"))["runs"][-1]["text"][0])')"
LANGUAGE="$(python -c 'import json; print(json.load(open("transcription.json", encoding="utf-8"))["runs"][-1]["language"][0])')"
python tools/qwen3_forced_aligner/align.py \
  --artifact models/qwen3-forced-aligner-0.6b-bf16.ninfer \
  --engine build/apps/ninfer-align --audio recording-16k.wav \
  --text "$TRANSCRIPT" --language "$LANGUAGE" \
  --warmups 0 --repeats 1 --out timestamps.json
```

Word results are in `cases[0].runs[-1].words[0]` of `timestamps.json`, with `text`, `start` and `end` fields in seconds. Edited transcripts can be passed directly to `--text`. Alignment supports Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian and Spanish.

[ASR guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-asr.md) · [ForcedAligner guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-forced-aligner.md)

## Model files

| File | Purpose | Size |
| --- | --- | ---: |
| `qwen3-asr-1.7b-bf16.ninfer` | Main ASR model | 4.088 GB |
| `qwen3-asr-0.6b-bf16.ninfer` | Optional draft model | 1.576 GB |
| `qwen3-forced-aligner-0.6b-bf16.ninfer` | Word timestamps | 1.840 GB |

All models use BF16 and are licensed under Apache-2.0.

[Full results](gpu-results.json) · [Draft results](draft-results.json) · [Alignment results](forced-alignment-results.json) · [Artifact manifest](artifact-manifest.json)
