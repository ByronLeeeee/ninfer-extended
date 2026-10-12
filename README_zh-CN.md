# NInfer Extended

[English](README.md) · 简体中文

基于 [Neroued/ninfer](https://github.com/Neroued/ninfer) 扩展的 C++/CUDA 推理引擎，
增加 Qwen3.5-0.8B 多模态、Qwen3 语音识别、词级时间戳对齐和文本向量支持。
Qwen3.5-0.8B 已在基于 **Qwen3.5-0.8B-Base** 的 **Xiaomi-OCR-0** 上验证。

模型通过原生 v3 `.ninfer` 文件和 Engine 接口运行，提供本地命令行与兼容
OpenAI/Anthropic 的 HTTP 服务。视觉、音频和语言推理共用 CUDA 算子。

## 功能

- **OCR 与文档解析：** Xiaomi-OCR-0 的视觉编码、语言 prefill 和 decode 均原生运行，
  模型文件包含图像处理配置和 tokenizer，支持 MTP1/MTP3、接受前缀状态恢复及 BF16 草稿输出头。
- **语音识别：** 原生音频编码器和 Qwen3 解码器，提供 `transcribe_features()`
  接口和 `ninfer-asr`。音频编码、语言 prefill 和 decode 均使用 CUDA Graph。
  支持 0.6B、1.7B 两个尺寸，也可以用 0.6B 草拟、1.7B 验证，执行一至四路转写。
- **词级时间戳：** Qwen3-ForcedAligner 通过一次音频与语言前向生成时间戳，
  支持独立批处理和 CUDA Graph 重放。
- **文本向量：** 在 GPU 上完成分段 causal attention、末 token 池化、维度截取
  和 FP32 L2 归一化，提供 `embed_tokens()` 和 `ninfer-embed`。
- **通用 BF16 算子：** 小批量投影、bias/残差与 bias/GELU 融合、多输出融合、寄存器内 SwiGLU、
  Gated DeltaNet 和分段 attention，按张量形状、数据类型和设备资源选择实现。
- **模型转换：** 提供 OCR、ASR、对齐和向量模型的 BF16 转换配方，
  保留 tokenizer、processor 和 Unicode 分词配置。
- **验证工具：** 独立数值校验、完整模型测试，以及与编译版 Transformers 的对比。

## 支持模型

| 模型 | 使用指南 | Hugging Face | 魔搭 |
|---|---|---|---|
| Xiaomi-OCR-0 BF16 | [OCR](docs/xiaomi-ocr.md) | [模型](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer) | [模型](https://modelscope.cn/models/ByronLeeee/Xiaomi-OCR-0-Ninfer) |
| Qwen3-ASR-1.7B-hf BF16 | [ASR](docs/qwen3-asr.md) | [模型](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) | [模型](https://modelscope.cn/models/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) |
| Qwen3-ASR-0.6B-hf BF16 | [ASR](docs/qwen3-asr.md) | [ASR / draft](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) | [ASR / draft](https://modelscope.cn/models/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) |
| Qwen3-ForcedAligner-0.6B BF16 | [时间戳对齐](docs/qwen3-forced-aligner.md) | [ASR 与对齐模型](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) | [ASR 与对齐模型](https://modelscope.cn/models/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer) |
| Qwen3-Embedding-0.6B BF16 | [向量模型](docs/qwen3-embedding.md) | [模型](https://huggingface.co/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer) | [模型](https://modelscope.cn/models/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer) |

同时支持上游 Qwen3.6-27B、Qwen3.8-27B 和 Qwen3.6-35B-A3B 的 v3 模型文件及生成流程。

## 与 Transformers 对比

同一张显卡上使用 BF16 权重和相同输入，表中取中位数。OCR、ASR 使用编译版
Transformers，Embedding 取 eager 和编译版中更快的结果，ForcedAligner 使用官方 eager 参考实现。

### Xiaomi-OCR-0

下表对比官方编译版 Transformers 普通解码与 NInfer MTP1，倍数包含引擎和 MTP 的共同收益。
BF16 MTP1 使用 131,072 行 BF16 草稿输出头，目标模型验证完整词表。
单路、4K 上下文、BF16 KV、256-token 输出上限；prefill 包含视觉编码与语言 prefill。
NInfer chunk 在 5070 Ti 上为 1,024，在 6000D 上为 2,048。

| 显卡 | 输入 | TF prefill tok/s | MTP1 prefill tok/s | Prefill 变化 | TF decode tok/s | MTP1 decode tok/s | Decode 倍数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 21,764 | 24,168 | +11.0% | 166.4 | 717.1 | 4.31× |
| RTX 5070 Ti | 英文合同 | 22,066 | 23,383 | +6.0% | 163.3 | 718.6 | 4.40× |
| RTX 5070 Ti | 密集公式（256 token） | 21,353 | 22,204 | +4.0% | 162.3 | 717.6 | 4.42× |
| RTX 6000D | 中文文书 | 36,383 | 40,038 | +10.0% | 232.8 | 904.4 | 3.88× |
| RTX 6000D | 英文合同 | 36,337 | 40,141 | +10.5% | 231.9 | 916.9 | 3.95× |
| RTX 6000D | 密集公式（256 token） | 36,254 | 36,284 | +0.1% | 235.4 | 910.8 | 3.87× |

两张显卡的三项测速输出均与编译版 Transformers 相同。六张标注文字图片、1,117 个计分字符均为 CER 0%。
MTP1 和 MTP3 的 11 张完整页面、四轮混合并发的 36 次输出均与普通解码完全一致，字符差异率 0%。
MTP3、四路吞吐和草稿接受率见 [完整 OCR 实测](docs/xiaomi-ocr-performance.md#bf16-mtp1-and-mtp3)。

#### 双方均启用 MTP1

**TF MTP1 为 Transformers + 自定义编译版 PyTorch MTP 适配**；Transformers 5.17.0
尚未原生支持这个有状态的 Qwen3.5 模型。双方使用相同的 131,072 行 BF16 草稿输出头，
主模型验证完整词表。

| 显卡 | 输入 | TF prefill tok/s | NInfer prefill tok/s | TF decode tok/s | NInfer decode tok/s | Decode 倍数 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 19,697 | 23,100 | 265.5 | 712.6 | 2.68× |
| RTX 5070 Ti | 英文合同 | 19,264 | 24,175 | 278.6 | 717.3 | 2.57× |
| RTX 5070 Ti | 密集公式（256 token） | 19,427 | 22,041 | 283.1 | 703.9 | 2.49× |
| RTX 6000D | 中文文书 | 32,722 | 40,214 | 366.2 | 905.5 | 2.47× |
| RTX 6000D | 英文合同 | 32,657 | 40,065 | 380.0 | 914.1 | 2.41× |
| RTX 6000D | 密集公式（256 token） | 32,918 | 36,506 | 380.0 | 914.9 | 2.41× |

两张显卡的三项输出均逐字一致。中文、英文页自然结束，分别输出 113、114 token；
公式页比较相同的 256-token 前缀。两边草稿数、接受数、接受率也一致，三项接受率分别为
**96.49%、100.00%、99.22%**。

[Prefill、请求耗时、完整输出头与 PyTorch MTP1/MTP3](docs/xiaomi-ocr-transformers-mtp.zh-CN.md)。

### Qwen3-ASR-1.7B

完整转写速度以**每秒处理多少秒音频**表示，计算方法为音频总时长除以完整热推理耗时。
四路批处理按四段音频总时长计算；不含模型加载、CPU 特征提取和首次 Graph 捕获。

| 显卡 | 输入 | TF 每秒处理音频秒数 | NInfer 每秒处理音频秒数 | 速度倍数 |
|---|---|---:|---:|---:|
| RTX 5070 Ti | 英文 60 秒 | 19.9 | **62.5** | 3.14× |
| RTX 5070 Ti | 英文 30 秒 × 4 | 54.1 | **197.8** | 3.66× |
| RTX 6000D | 英文 60 秒 | 29.0 | **87.7** | 3.03× |
| RTX 6000D | 英文 30 秒 × 4 | 84.3 | **299.0** | 3.55× |

单条 15.05 秒英文音频的 prefill 和 decode 速度：

| 显卡 | TF prefill ms | NInfer prefill ms | Prefill 倍数 | TF decode tok/s | NInfer decode tok/s | Decode 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 39.416 | 16.456 | 2.40× | 68.2 | 213.4 | 3.13× |
| RTX 6000D | 13.487 | 10.544 | 1.28× | 93.3 | 294.8 | 3.16× |

6000D 四路推理：

| 输入 | TF prefill ms | NInfer prefill ms | Prefill 倍数 | TF decode tok/s | NInfer decode tok/s | Decode 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| 英文 15.05 秒 × 4 | 34.437 | 29.374 | 1.17× | 278.9 | 1129.5 | 4.05× |
| 中文 4.20 秒 × 4 | 18.098 | 12.534 | 1.44× | 278.6 | 1153.0 | 4.14× |
| 英文 30 秒 × 4 | 58.855 | 57.538 | 1.02× | 278.5 | 1096.5 | 3.94× |

60 秒拼接英文音频的完整推理耗时：

| 显卡 | TF 推理耗时 | NInfer 推理耗时 | TF 每秒处理音频秒数 | NInfer 每秒处理音频秒数 | 速度倍数 |
|---|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 3.011 s | 0.960 s | 19.9 | 62.5 | 3.14× |
| RTX 6000D | 2.071 s | 0.684 s | 29.0 | 87.7 | 3.03× |

单请求质量测试使用 73 条带标注的英文录音，共 1,150 个参考词：

| 显卡 | TF WER | NInfer WER | 原始输出一致率 | 归一化文本一致率 |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 3.9130% | 3.8261% | 98.63% | 98.63% |
| RTX 6000D | 3.5652% | 3.5652% | 97.26% | 100.00% |

[ASR 结果与使用方法](model-cards/Qwen3-ASR-1.7B-hf-NInfer/README.md)。

#### 单路草稿加速

同一组短录音对照 Transformer，使用各 GPU 已通过正确率验证的草稿配置：

| GPU | 输入 | K | TF 音频秒/秒 | NInfer 音频秒/秒 | 推理倍数 |
|---|---|---:|---:|---:|---:|
| RTX 5070 Ti | 英文 15.05 秒 | 5 | 18.9 | **92.9** | 4.93× |
| RTX 5070 Ti | 中文 4.20 秒 | 5 | 20.0 | **98.0** | 4.89× |
| RTX 6000D | 英文 15.05 秒 | 3 | 27.7 | **102.8** | 3.71× |
| RTX 6000D | 中文 4.20 秒 | 3 | 33.7 | **108.9** | 3.23× |

原始 BF16 0.6B 草稿与 1.7B 主模型一起运行，完整 prefill 包括两份模型的音频编码和语言前向。

| GPU | K | 60 秒非循环音频推理 s | 音频秒/秒 | 完整 prefill ms | Decode tok/s | 输出与普通解码一致 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 5 | 0.5518 | **108.7** | 99.40 | 345.5 | 73/73 |
| RTX 6000D | 3 | 0.5958 | **100.7** | 84.55 | 327.8 | 73/73 |

6000D 采用已验证的 K=3 版本；运行版本、单路 Transformer 对比和接受率见 [ASR 模型卡](model-cards/Qwen3-ASR-1.7B-hf-NInfer/README_zh-CN.md)。

### Qwen3-Embedding-0.6B

单条文本，输出归一化后的 1,024 维向量。GPU 耗时包含完整语言 prefill 和池化；
向量模型没有 decode 阶段。

| 显卡 | 输入 token 数 | TF GPU ms | NInfer GPU ms | 吞吐变化 |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 128 | 3.266 | 2.578 | +26.69% |
| RTX 5070 Ti | 512 | 8.781 | 8.150 | +7.75% |
| RTX 5070 Ti | 2048 | 42.161 | 37.610 | +12.10% |
| RTX 6000D | 128 | 2.797 | 1.980 | +41.29% |
| RTX 6000D | 512 | 6.187 | 4.616 | +34.02% |
| RTX 6000D | 2048 | 21.251 | 21.298 | -0.22% |

249 个向量的平均余弦相似度约 **0.9998**。八条检索查询，双方 Top1 均命中，
**Top1/Top3 结果一致率为 100%**。
[Embedding 详细结果](model-cards/Qwen3-Embedding-0.6B-NInfer/README.md)。

### Qwen3-ForcedAligner-0.6B

单条 15.05 秒英文音频，GPU 耗时包含音频编码、语言前向和时间戳分类。

| 显卡 | TF GPU ms | NInfer GPU ms | 速度倍数 |
|---|---:|---:|---:|
| RTX 5070 Ti | 68.464 | 10.283 | 6.66× |
| RTX 6000D | 10.920 | 6.157 | 1.77× |

四组单条音频的时间戳边界一致率：5070 Ti 为 **100%**，6000D 为 **99.75%**；
最大差异分别为 **0 ms / 80 ms**。
[对齐结果与使用方法](docs/qwen3-forced-aligner.md)。

## 构建

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build -j
```

需要 64 位 Linux（Windows 使用 WSL2）、支持 `sm_120a` 的 CUDA、C++20、
CMake ≥3.28、Ninja、FFmpeg 开发库、libcurl ≥7.85、pkg-config、PCRE2、cuDNN 9 和 cuBLAS。
cuBLAS 已随 CUDA 安装时可省略 `CUBLAS_ROOT`。扩展版已在 RTX 5070 Ti 和 RTX 6000D 上测试。

## 文档

- [命令行用法](docs/cli.md)
- [HTTP 服务](docs/serving.md)
- [模型转换](docs/weight-conversion.md)
- [更新说明与版本间优化结果](CHANGELOG.md)
- [OCR 性能](docs/xiaomi-ocr-performance.md)
- [构建配置](docs/maintainer/build-system.md)

---

基于 [Neroued/ninfer](https://github.com/Neroued/ninfer)，起始版本为
`594930e7b609efa4bcea3ae4f24cd9d66b5f224f`。
[上游项目介绍](https://github.com/Neroued/ninfer#readme)。
