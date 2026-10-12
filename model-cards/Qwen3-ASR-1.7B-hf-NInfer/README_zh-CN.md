# Qwen3-ASR-1.7B · NInfer

[English](README.md)

## 模型介绍

这是 [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 的 BF16 NInfer 版本，支持自动语言识别、30 种语言和 22 种中文方言的语音转写。使用 **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)** 运行，音频编码、prefill 和 decode 均由原生 C++/CUDA 引擎执行。

本仓库同时提供 0.6B 草稿模型和 Qwen3-ForcedAligner-0.6B：草稿模型用于转写加速，ForcedAligner 为转写文本生成逐词时间戳。主模型支持最多四条音频批处理，默认自动识别语言，也可指定语言、提示词和热词。

## 速度对比

**音频秒/秒**表示每秒能处理的音频时长，按输入音频总时长 ÷ 推理耗时计算。数据为 BF16 预热后的中位数，统计音频编码、prefill、完整转写和同步，不含模型加载及 CPU 特征提取。Transformers 使用编译后的 prefill 和 decode CUDA Graph。

### 单个 1.7B 模型

| GPU | 音频 | Transformers (ms) | NInfer (ms) | Transformers 音频秒/秒 | NInfer 音频秒/秒 | 速度倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 英文 15.05 秒 | 798.0 | **248.8** | 18.9 | **60.5** | 3.21× |
| RTX 5070 Ti | 中文 4.20 秒 | 210.0 | **56.9** | 20.0 | **73.8** | 3.69× |
| RTX 6000D | 英文 15.05 秒 | 543.1 | **174.9** | 27.7 | **86.1** | 3.11× |
| RTX 6000D | 中文 4.20 秒 | 124.8 | **40.4** | 33.7 | **104.0** | 3.09× |

60 秒音频由固定录音重复拼接：

| GPU | 音频 | Transformers 耗时 (s) | NInfer 耗时 (s) | Transformers 音频秒/秒 | NInfer 音频秒/秒 | 速度倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 英文 | 3.011 | **0.960** | 19.9 | **62.5** | 3.14× |
| RTX 5070 Ti | 中文 | 1.679 | **0.560** | 35.7 | **107.1** | 3.00× |
| RTX 6000D | 英文 | 2.071 | **0.684** | 29.0 | **87.7** | 3.03× |
| RTX 6000D | 中文 | 1.115 | **0.392** | 53.8 | **153.1** | 2.85× |

### 四路批处理

| GPU | 音频 | Transformers 音频秒/秒 | NInfer 音频秒/秒 | 速度倍数 |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | 英文 30 秒 × 4 | 54.1 | **197.8** | 3.66× |
| RTX 5070 Ti | 中文 4.20 秒 × 4 | 56.8 | **206.3** | 3.63× |
| RTX 6000D | 英文 30 秒 × 4 | 84.3 | **299.0** | 3.55× |
| RTX 6000D | 中文 4.20 秒 × 4 | 100.3 | **352.2** | 3.51× |

### Prefill 与 decode

Prefill 包含音频编码、语言前向和首 token；decode 从第二个 token 开始，包含 EOS。四路数据为合计 token/s。

| GPU | 音频 | Transformers prefill (ms) | NInfer prefill (ms) | Transformers decode (tok/s) | NInfer decode (tok/s) | Decode 倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 英文 15.05 秒 | 39.416 | 16.456 | 68.2 | **213.4** | 3.13× |
| RTX 5070 Ti | 中文 4.20 秒 | 41.861 | 8.639 | 66.1 | **214.8** | 3.25× |
| RTX 5070 Ti | 英文 60 秒 | 74.651 | 59.059 | 66.8 | **207.8** | 3.11× |
| RTX 5070 Ti | 英文 30 秒 × 4 | 130.738 | 120.680 | 186.0 | **786.0** | 4.23× |
| RTX 6000D | 英文 15.05 秒 | 13.487 | 10.544 | 93.3 | **294.8** | 3.16× |
| RTX 6000D | 中文 4.20 秒 | 10.779 | 6.368 | 92.3 | **296.7** | 3.22× |
| RTX 6000D | 英文 60 秒 | 32.995 | 32.588 | 92.2 | **283.1** | 3.07× |
| RTX 6000D | 英文 30 秒 × 4 | 58.855 | 57.538 | 278.5 | **1096.5** | 3.94× |

### 0.6B 草稿加速

1.7B 主模型加 0.6B 草稿模型，5070 Ti 使用 K=5，6000D 使用 K=3。

| GPU | 音频 | K | Transformers (ms) | NInfer (ms) | Transformers 音频秒/秒 | NInfer 音频秒/秒 | 速度倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 英文 15.05 秒 | 5 | 798.0 | **162.0** | 18.9 | **92.9** | 4.93× |
| RTX 5070 Ti | 中文 4.20 秒 | 5 | 210.0 | **42.9** | 20.0 | **98.0** | 4.89× |
| RTX 6000D | 英文 15.05 秒 | 3 | 543.1 | **146.4** | 27.7 | **102.8** | 3.71× |
| RTX 6000D | 中文 4.20 秒 | 3 | 124.8 | **38.6** | 33.7 | **108.9** | 3.23× |

启用草稿后的完整 prefill 与 decode：

| GPU | 音频 | Transformers prefill (ms) | NInfer prefill (ms) | Transformers decode (tok/s) | NInfer decode (tok/s) | Decode 倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 英文 15.05 秒 | 39.416 | 26.193 | 68.2 | **331.0** | 4.86× |
| RTX 5070 Ti | 中文 4.20 秒 | 41.861 | 13.187 | 66.1 | **318.7** | 4.82× |
| RTX 6000D | 英文 15.05 秒 | 13.487 | 16.646 | 93.3 | **375.4** | 4.02× |
| RTX 6000D | 中文 4.20 秒 | 10.779 | 9.984 | 92.3 | **356.0** | 3.86× |

七段不同录音加静音间隔组成的 60 秒混合音频：

| GPU | K | 60 秒音频耗时 (s) | 每秒处理音频秒数 | Prefill (ms) | Decode (tok/s) | 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 5 | **0.5518** | **108.7** | 99.40 | 345.5 | 98.57% |
| RTX 6000D | 3 | **0.5958** | **100.7** | 84.55 | 327.8 | 99.21% |

### 四路 MTP

RTX 5070 Ti，BF16 1.7B 主模型＋0.6B 草稿模型，K=5。2026-10-13 原生构建验证，预热后六次中位数，统计两个模型的音频编码、prefill 和完整解码，不含加载与 CPU 特征提取。

| 输入 | 普通四路 ms | MTP 四路 ms | 速度倍数 |
|---|---:|---:|---:|
| 中英混合 × 4 | 276.9 | 208.4 | 1.33× |
| 英语 30 秒 × 4 | 548.4 | 452.1 | 1.21× |
| 中文 4.2 秒 × 4 | 70.0 | 65.4 | 1.07× |

[四路测速数据](../../docs/results/qwen3-asr-draft-batch4.json)。

### ForcedAligner 对齐耗时

BF16 单请求，统计音频编码、语言前向及时间戳分类。Transformers 参考使用 eager，NInfer 使用 CUDA Graph。

| GPU | 音频 | Transformers eager (ms) | NInfer (ms) | 速度倍数 |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | 中文 4.20 秒 | 71.336 | **4.449** | 16.03× |
| RTX 5070 Ti | 英文 15.05 秒 | 68.464 | **10.283** | 6.66× |
| RTX 6000D | 中文 4.20 秒 | 10.280 | **3.704** | 2.78× |
| RTX 6000D | 英文 15.05 秒 | 10.920 | **6.157** | 1.77× |

## 正确率测试

测试集为 73 条有人工标注的英文录音，共 481.03 秒、1,150 个词。WER 为词错误率，CER 为字符错误率，越低越好。WER 按归一化词计算，CER 不计空格。

| GPU | Transformers WER | NInfer WER | Transformers CER | NInfer CER |
| --- | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 3.9130% | **3.8261%** | 1.3553% | **1.3166%** |
| RTX 6000D | 3.5652% | **3.5652%** | 1.2585% | **1.2585%** |

### 与 Transformers 的结果一致率

| GPU | 原始文本/token 一致率 | 归一化文本一致率 | Token 编辑差异率 |
| --- | ---: | ---: | ---: |
| RTX 5070 Ti | 98.63% | 98.63% | 0.0573% |
| RTX 6000D | 97.26% | 100.00% | 0.1720% |

### 草稿解码的结果一致率

同一组 73 条录音，草稿解码与单个 1.7B 模型的输出逐 token 比较。

| GPU | K | 与单模型完整 token 一致 | 草稿接受率 |
| --- | ---: | ---: | ---: |
| RTX 5070 Ti | 5 | **100% (73/73)** | 85.53% |
| RTX 6000D | 3 | **100% (73/73)** | 88.58% |

### 时间戳一致率

四段音频共 400 个时间边界，与官方 ForcedAligner 结果比较。

| GPU | 时间戳一致率 | 最大时间差 |
| --- | ---: | ---: |
| RTX 5070 Ti | 100.00% (400/400) | 0 ms |
| RTX 6000D | 99.75% (399/400) | 80 ms |

## 使用方法

### 安装与编译

运行环境：Linux 或 WSL2、Python 3.11、支持 `sm_120a` 的 CUDA、cuDNN 9 和 cuBLAS。ASR 与对齐工具共用一个 Python 环境。构建依赖见 [项目说明](https://github.com/ByronLeeeee/ninfer-extended#build)。

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-asr ninfer-align -j
pip install -r tools/qwen3_asr/requirements.txt
```

5070 Ti 使用主分支；6000D 的 K=3 数据对应版本 [`64395f47`](https://github.com/ByronLeeeee/ninfer-extended/commit/64395f477fa8e5134f11edc5eb97697c47656d82)，复现该配置时在编译前执行 `git checkout 64395f477fa8e5134f11edc5eb97697c47656d82`。

### 下载并转写

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

`transcription.json` 保存识别文本和语言。`--audio` 可传入 1–4 个文件；语言默认自动，也可添加 `--language Chinese`、`--prompt "这是一段医学讲座"` 或 `--hotwords 阿司匹林 血红蛋白`。

### 启用草稿加速

```bash
hf download ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer \
  qwen3-asr-0.6b-bf16.ninfer --local-dir models
python tools/qwen3_asr/transcribe.py \
  --artifact models/qwen3-asr-1.7b-bf16.ninfer --engine build/apps/ninfer-asr \
  --draft-artifact models/qwen3-asr-0.6b-bf16.ninfer --draft-tokens 5 \
  --audio recording-16k.wav --warmups 0 --repeats 1 --out transcription.json
```

RTX 5070 Ti 使用 `--draft-tokens 5`，`--audio` 可同时传入 1–4 个 WAV 文件。6000D 的历史单路配方使用 `--draft-tokens 3`。

### 添加逐词时间戳

先完成转写，再运行对齐模型：

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

`timestamps.json` 中的 `cases[0].runs[-1].words[0]` 为逐词结果，每项包含 `text`、`start`、`end`，时间单位为秒。也可将校正后的文本直接传给 `--text`。对齐支持中文、英文、粤语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语和西班牙语。

[ASR 使用说明](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-asr.md) · [ForcedAligner 使用说明](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-forced-aligner.md)

## 模型文件

| 文件 | 用途 | 大小 |
| --- | --- | ---: |
| `qwen3-asr-1.7b-bf16.ninfer` | 主转写模型 | 4.088 GB |
| `qwen3-asr-0.6b-bf16.ninfer` | 可选草稿模型 | 1.576 GB |
| `qwen3-forced-aligner-0.6b-bf16.ninfer` | 逐词时间戳 | 1.840 GB |

全部模型使用 BF16，许可证为 Apache-2.0。

[完整测试数据](gpu-results.json) · [草稿加速数据](draft-results.json) · [对齐测试数据](forced-alignment-results.json) · [模型清单](artifact-manifest.json)
