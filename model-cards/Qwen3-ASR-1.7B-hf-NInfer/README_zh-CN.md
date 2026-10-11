---
license: apache-2.0
base_model:
- Qwen/Qwen3-ASR-1.7B-hf
- Qwen/Qwen3-ASR-0.6B-hf
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

# Qwen3-ASR-1.7B BF16 NInfer 模型

本仓库提供 Qwen3-ASR-1.7B-hf 的 BF16 NInfer 模型，使用 [ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended) 的 `main` 分支运行。音频编码、语言 prefill 和 decode 均由 C++/CUDA 引擎执行。模型文件已包含 tokenizer、processor 和聊天模板，707 个原始参数的权重字节完整保留。

本仓库还提供 Qwen3-ForcedAligner-0.6B 的 BF16 模型，可为主 ASR 的识别结果添加逐词时间戳，使用方法和实测结果见下文。

简体中文 · [English](README.md)

## 单路草稿加速：最佳已验证配置

使用同一份 BF16 1.7B 主模型和 BF16 0.6B 草稿模型：5070 Ti 选 K=5，6000D 选 K=3。
两种配置在 73 条标注录音中均与普通解码的完整 token 序列一致。四路批处理使用单个 1.7B 模型。

| GPU | 输入 | K | TF 推理 ms | NInfer 推理 ms | TF 音频秒/秒 | NInfer 音频秒/秒 | 推理倍数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 英文 15.05 秒 | 5 | 798.0 | **162.0** | 18.9 | **92.9** | 4.93× |
| RTX 5070 Ti | 中文 4.20 秒 | 5 | 210.0 | **42.9** | 20.0 | **98.0** | 4.89× |
| RTX 6000D | 英文 15.05 秒 | 3 | 543.1 | **146.4** | 27.7 | **102.8** | 3.71× |
| RTX 6000D | 中文 4.20 秒 | 3 | 124.8 | **38.6** | 33.7 | **108.9** | 3.23× |

以下 prefill 已计入两份模型的音频编码和语言前向；decode 包含草拟、主模型验证和 KV 补齐。

| GPU | 输入 | TF prefill ms | NInfer prefill ms | TF decode tok/s | NInfer decode tok/s | Decode 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 英文 15.05 秒 | 39.416 | 26.193 | 68.2 | **331.0** | 4.86× |
| RTX 5070 Ti | 中文 4.20 秒 | 41.861 | 13.187 | 66.1 | **318.7** | 4.82× |
| RTX 6000D | 英文 15.05 秒 | 13.487 | 16.646 | 93.3 | **375.4** | 4.02× |
| RTX 6000D | 中文 4.20 秒 | 10.779 | 9.984 | 92.3 | **356.0** | 3.86× |

### 60 秒非循环音频

七段不同录音加静音间隔。每秒处理音频秒数 = 总音频时长 ÷ 完整推理耗时，表中数据均为正式运行中位数。

| GPU | K | 推理秒数 | 每秒处理音频秒数 | 完整 prefill ms | Decode tok/s | 草稿接受率 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 5 | 0.5518 | **108.7** | 99.40 | 345.5 | 98.57% |
| RTX 6000D | 3 | 0.5958 | **100.7** | 84.55 | 327.8 | 99.21% |

本组音频与下方循环录音的 Transformer 对照不同，单独列出。

### 正确率、接受率与显存

接受率统计被主模型接受的草稿 token，WER/CER 与人工标注比较。

| GPU | K | WER | CER | 与普通解码完整 token 一致 | 73 条录音草稿接受率 | Engine 显存 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 5 | 3.8261% | 1.3166% | 73/73 (100%) | 85.53% | 7.17 GiB |
| RTX 6000D | 3 | 3.5652% | 1.2585% | 73/73 (100%) | 88.58% | 7.17 GiB |

60 秒输入的双模型 Engine 分配约 7.17 GiB，单模型约 4.77 GiB。6000D 的 K=5 出现过转写差异，推荐 K=3。

### 下载并启用

```bash
ms-hub download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer \
  qwen3-asr-1.7b-bf16.ninfer qwen3-asr-0.6b-bf16.ninfer --local-dir models
python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --draft-artifact models/qwen3-asr-0.6b-bf16.ninfer --draft-tokens 5 \
  --audio recording.wav --out transcription.json
```

5070 Ti 使用当前主分支和 `--draft-tokens 5`；6000D 使用已验证版本 [`64395f47`](https://github.com/ByronLeeeee/ninfer-extended/commit/64395f477fa8e5134f11edc5eb97697c47656d82) 与 `--draft-tokens 3`，编译前执行 `git checkout 64395f477fa8e5134f11edc5eb97697c47656d82`。
省略两个草稿参数即可只载入 1.7B。CPU 前处理与时间戳工具共用同一 Python 环境。

[草稿测量数据](draft-results.json) · [草稿模型清单](draft-artifact-manifest.json)

## 单个 1.7B 模型：Transformer 对照

单条 60 秒英文音频，NInfer 在 RTX 5070 Ti 上每秒可处理 **62.5 秒音频**，在 RTX 6000D 上可处理 **87.7 秒音频**。四条 30 秒英文音频一起转写，两张卡分别达到每秒 **197.8 秒**和 **299.0 秒音频**。

每秒处理音频秒数 = 输入音频总时长 ÷ 完整热推理耗时。例如，60 秒音频用 0.684 秒转写完成，就是每秒处理 87.7 秒音频。耗时包含音频编码、prefill、完整文字生成和同步，不含模型加载、CPU 特征提取和首次 Graph 捕获。四路批处理按四段音频的总时长计算。60 秒输入由固定录音重复拼接，识别质量另用下方标注录音评估。

### RTX 5070 Ti

| 输入 | 路数 | TF 每秒处理音频秒数 | NInfer 每秒处理音频秒数 | 速度倍数 |
| --- | ---: | ---: | ---: | ---: |
| 英文 15.05 秒 | 1 | 18.9 | **60.5** | 3.21× |
| 中文 4.20 秒 | 1 | 20.0 | **73.8** | 3.69× |
| 英文 60 秒 | 1 | 19.9 | **62.5** | 3.14× |
| 中文 60 秒 | 1 | 35.7 | **107.1** | 3.00× |
| 英文 15.05 秒 × 4 | 4 | 52.0 | **194.5** | 3.74× |
| 中文 4.20 秒 × 4 | 4 | 56.8 | **206.3** | 3.63× |
| 英文 30 秒 × 4 | 4 | 54.1 | **197.8** | 3.66× |
| 中英文混合 × 4 | 4 | 26.5 | **111.2** | 4.19× |

### RTX 6000D

| 输入 | 路数 | TF 每秒处理音频秒数 | NInfer 每秒处理音频秒数 | 速度倍数 |
| --- | ---: | ---: | ---: | ---: |
| 英文 15.05 秒 | 1 | 27.7 | **86.1** | 3.11× |
| 中文 4.20 秒 | 1 | 33.7 | **104.0** | 3.09× |
| 英文 60 秒 | 1 | 29.0 | **87.7** | 3.03× |
| 中文 60 秒 | 1 | 53.8 | **153.1** | 2.85× |
| 英文 15.05 秒 × 4 | 4 | 81.4 | **299.5** | 3.68× |
| 中文 4.20 秒 × 4 | 4 | 100.3 | **352.2** | 3.51× |
| 英文 30 秒 × 4 | 4 | 84.3 | **299.0** | 3.55× |
| 中英文混合 × 4 | 4 | 41.3 | **165.2** | 4.00× |

## 安装与运行

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-asr ninfer-align -j
pip install -r tools/qwen3_asr/requirements.txt
pip install modelscope-hub
ms-hub download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer qwen3-asr-1.7b-bf16.ninfer --local-dir models
python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --audio recording.wav --out transcription.json
```

构建需要 64 位 Linux（Windows 使用 WSL2）、支持 `sm_120a` 的 CUDA、C++20、CMake ≥3.28、Ninja、FFmpeg 开发库、libcurl、PCRE2、cuDNN 9 和 cuBLAS。若 cuBLAS 已随 CUDA 安装，可省略 `CUBLAS_ROOT`。输入为 16 kHz、单声道 PCM16 WAV，可一次传入 1–4 个文件。默认 BF16 权重和 KV、4K 上下文、最多 4 路并发，输出上限 1,024 token。语言 prefill 默认采用 FP32 概率运算。

[完整使用说明](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-asr.md)

## Prefill 和 decode 速度对比

两边均使用 BF16 权重和 KV，每路上下文 4,096 token，greedy 解码。Transformers 的音频编码在 GPU 上运行，语言部分采用 fullgraph 编译，decode 使用 CUDA Graph；NInfer 的音频、语言 prefill 和 decode 均使用 CUDA Graph。表中为预热后的中位数。5070 Ti 使用 CUDA 13.3/GCC 13.3，6000D 使用 CUDA 13.2/GCC 15.2；Transformers 为 5.17.0，PyTorch 为 2.14.0+cu132。

Prefill 包含音频编码、语言 prefill 和首 token 的 GPU 计算；decode 从第二个 token 开始统计，计入 EOS，多路取合计 token/s。速度倍数均为本版 NInfer 相对 Transformers 的比值。

### RTX 5070 Ti

| 输入 | 并发 | TF prefill ms | NInfer prefill ms | Prefill 倍数 | TF decode tok/s | NInfer decode tok/s | Decode 倍数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 英文 15.05 秒 | 1 | 39.416 | 16.456 | 2.40× | 68.2 | 213.4 | 3.13× |
| 中文 4.20 秒 | 1 | 41.861 | 8.639 | 4.85× | 66.1 | 214.8 | 3.25× |
| 英文 10 秒 | 1 | 44.510 | 12.537 | 3.55× | 67.4 | 213.3 | 3.16× |
| 英文 30 秒 | 1 | 56.840 | 30.695 | 1.85× | 68.3 | 211.2 | 3.09× |
| 英文 60 秒 | 1 | 74.651 | 59.059 | 1.26× | 66.8 | 207.8 | 3.11× |
| 中文 60 秒 | 1 | 71.823 | 59.173 | 1.21× | 67.4 | 208.0 | 3.08× |
| 英文 15.05 秒 × 4 | 4 | 80.500 | 62.949 | 1.28× | 187.3 | 814.1 | 4.35× |
| 英文 30 秒 × 4 | 4 | 130.738 | 120.680 | 1.08× | 186.0 | 786.0 | 4.23× |

### RTX 6000D

| 输入 | 路数 | TF prefill ms | NInfer prefill ms | Prefill 倍数 | TF decode tok/s | NInfer decode tok/s | Decode 倍数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 英文 15.05 秒 | 1 | 13.487 | 10.544 | 1.28× | 93.3 | 294.8 | 3.16× |
| 中文 4.20 秒 | 1 | 10.779 | 6.368 | 1.69× | 92.3 | 296.7 | 3.22× |
| 英文 60 秒 | 1 | 32.995 | 32.588 | 1.01× | 92.2 | 283.1 | 3.07× |
| 英文 15.05 秒 × 4 | 4 | 34.437 | 29.374 | 1.17× | 278.9 | 1129.5 | 4.05× |
| 中文 4.20 秒 × 4 | 4 | 18.098 | 12.534 | 1.44× | 278.6 | 1153.0 | 4.14× |
| 英文 30 秒 × 4 | 4 | 58.855 | 57.538 | 1.02× | 278.5 | 1096.5 | 3.94× |
| 中英文混合 × 4 | 4 | 50.569 | 18.693 | 2.71× | 143.6 | 585.4 | 4.08× |

## 推理延时与 60 秒音频

延时统计模型热推理的一次完整调用，包括音频编码、prefill、生成和同步；不含模型加载、CPU 特征提取和首次 Graph 捕获。短音频延时如下。

| 显卡 | 输入 | Transformers ms | NInfer ms | 推理速度倍数 |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 英文 15.05 秒 | 798.0 | 248.8 | 3.21× |
| RTX 5070 Ti | 中文 4.20 秒 | 210.0 | 56.9 | 3.69× |
| RTX 6000D | 英文 15.05 秒 | 543.1 | 174.9 | 3.11× |
| RTX 6000D | 中文 4.20 秒 | 124.8 | 40.4 | 3.09× |

60 秒样例由固定录音重复拼接，用于测速；表中的“每秒处理音频秒数”为 60 ÷ 实际热推理耗时。

| 显卡 | 60 秒输入 | TF 推理秒 | NInfer 推理秒 | TF 每秒处理音频秒数 | NInfer 每秒处理音频秒数 | 速度倍数 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 英文 60 秒 | 3.011 | 0.960 | 19.9 | 62.5 | 3.14× |
| RTX 5070 Ti | 中文 60 秒 | 1.679 | 0.560 | 35.7 | 107.1 | 3.00× |
| RTX 6000D | 英文 60 秒 | 2.071 | 0.684 | 29.0 | 87.7 | 3.03× |
| RTX 6000D | 中文 60 秒 | 1.115 | 0.392 | 53.8 | 153.1 | 2.85× |

## 识别质量与结果一致率

单请求质量测试使用 73 条真实标注英文录音，共 481.03 秒、1,150 个标注词。WER 去掉大小写、标点和多余空白，保留词内撇号；CER 不计空格。

| 显卡 | TF WER | NInfer WER | TF CER | NInfer CER | 原始文本/token 一致率 | 归一化文本一致率 | Token 编辑差异率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 3.9130% | 3.8261% | 1.3553% | 1.3166% | 98.63% | 98.63% | 0.0573% |
| RTX 6000D | 3.5652% | 3.5652% | 1.2585% | 1.2585% | 97.26% | 100.00% | 0.1720% |

5070 Ti 上，Transformers 为 45 个错词，NInfer 为 44 个；差异是一处 `has/is`，NInfer 与标注一致。6000D 两边均为 41 个错词，归一化后全部一致；原始差异为大小写和标点。两张卡均完成独立 FP64 算子校验。

## 逐词时间戳：Qwen3-ForcedAligner-0.6B

本仓库提供 `qwen3-forced-aligner-0.6b-bf16.ninfer`，与主 ASR 模型一起放在本仓库。两者都使用 [ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended) 的 `main` 分支。

主 ASR 先识别文字和语言，ForcedAligner 再根据同一段音频和识别文本给每个词标出开始、结束时间。它执行一次音频编码、语言前向和时间戳分类，使用 BF16 权重及 CUDA Graph，不占用 KV 缓存。708 个原始 BF16 参数的权重字节完整保留，时间戳步长为 **80 ms**。

支持中文、英文、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语和西班牙语。

### 和主 ASR 一起运行

以下命令先转写，再自动读取识别文本和语言生成时间戳。ASR 和对齐工具共用一个 Transformers 5.17.0 环境，GPU 推理由 NInfer 执行。首次安装请先按上面的步骤配置构建依赖。

```bash
# Run from the ninfer-extended repository root.
cmake --build build --target ninfer-asr ninfer-align -j
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/qwen3_asr/requirements.txt modelscope-hub

.venv/bin/ms-hub download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer \
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

`transcription.json` 保存识别文本，`timestamps.json` 的 `cases[0].runs[-1].words[0]` 保存逐词结果，每项包含 `text`、`start`、`end`，时间单位为秒。修改转写文本后，也可以直接把校正后的内容传给 `--text`。多文件时，`--audio`、`--text`、`--language` 各传入相同数量的值；主 ASR 支持 1–4 路，对齐工具支持 1–8 路。

### 对齐耗时和时间戳一致率

单请求，预热后测量 7 次，取中位数。下表统计完整音频编码、语言前向和时间戳分类的 CUDA 事件耗时，包含两阶段之间的调度等待；不含模型加载、CPU 前处理和首次 Graph 捕获。对齐参考为官方 `qwen-asr 0.0.6`、Transformers 4.57.6、PyTorch 2.10.0+cu130，使用 BF16 GPU 和融合 SDPA，无 CPU offload 或 MATH attention 回退。该参考采用 eager 执行；上方主 ASR 的参考使用编译和 decode Graph。

| 显卡 | 音频 | Transformers ms | NInfer ms | 速度倍数 |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文 4.20 秒 | 71.336 | 4.449 | 16.03× |
| RTX 5070 Ti | 英文 15.05 秒 | 68.464 | 10.283 | 6.66× |
| RTX 5070 Ti | 英文 60.205 秒 | 59.146 | 33.970 | 1.74× |
| RTX 5070 Ti | 中文 0.75 秒 | 74.674 | 3.530 | 21.15× |
| RTX 6000D | 中文 4.20 秒 | 10.280 | 3.704 | 2.78× |
| RTX 6000D | 英文 15.05 秒 | 10.920 | 6.157 | 1.77× |
| RTX 6000D | 英文 60.205 秒 | 24.070 | 23.536 | 1.02× |
| RTX 6000D | 中文 0.75 秒 | 9.118 | 3.276 | 2.78× |

60.205 秒样例由同一段英文录音重复拼接。对齐模型一次前向输出所有时间戳。

| 显卡 | 与官方参考完全一致的时间边界 | 一致率 | 最大时间差 |
| --- | --- | --- | --- |
| RTX 5070 Ti | 400 / 400 | 100.00% | 0 ms |
| RTX 6000D | 399 / 400 | 99.75% | 80 ms |

一致率统计上述四个样例的单请求时间边界。固定输入形状下，重复 Graph 和 eager 结果一致；改变批次形状时，接近分类边界的结果可能相差一个 80 ms 档位。

模型文件：`qwen3-forced-aligner-0.6b-bf16.ninfer` · 1,840,359,257 bytes · BF16 · Apache-2.0。SHA256：`45234e54ba66833e846e4d3d30eca06c9f95833a35350508c667432a90143473`。

[对齐使用说明](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-forced-aligner.md) · [对齐测量数据](forced-alignment-results.json) · [对齐模型清单](forced-aligner-manifest.json)

## 模型文件与许可证

`qwen3-asr-1.7b-bf16.ninfer` · 4,087,720,481 bytes · BF16 · Apache-2.0

SHA256: `50d9d8c2ed690b80f5acbf94c66b344ee7fd12cf79dfe670ad10e96b5b411eea`

[测量数据](gpu-results.json) · [模型清单](artifact-manifest.json)
