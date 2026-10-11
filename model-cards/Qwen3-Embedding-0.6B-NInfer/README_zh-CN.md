---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-0.6B
pipeline_tag: feature-extraction
library_name: ninfer
tags:
- ninfer
- embeddings
- bf16
- cuda
- qwen3
---

# Qwen3-Embedding-0.6B NInfer 模型

Qwen3-Embedding-0.6B 的 BF16 NInfer 版本，用于文本向量、语义检索和相似度计算。原模型的 310 个 BF16 参数完整保留，文件内含 tokenizer。已在 RTX 5070 Ti 和 RTX 6000D 上验证。

[English](README.md)

## 安装与使用

请使用 **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)** 的 `main` 分支运行本模型。

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-embed -j
pip install "transformers==5.17.0" numpy
pip install modelscope-hub
ms-hub download ByronLeeee/Qwen3-Embedding-0.6B-Ninfer qwen3-embedding-0.6b-bf16.ninfer --local-dir models
```

构建环境：64 位 Linux（Windows 可用 WSL2）、支持 `sm_120a` 的 CUDA、C++20、CMake ≥3.28、Ninja、FFmpeg 开发库、libcurl、PCRE2、cuDNN 9 和 cuBLAS。CUDA 已提供 cuBLAS 时可省略 `CUBLAS_ROOT`。

把文本保存为 `texts.json`：

```json
{"texts": ["A passage for semantic retrieval.", "向量检索可以查找语义相关的文本。"]}
```

```bash
python tools/qwen3_embedding/embed.py \
  --artifact models/qwen3-embedding-0.6b-bf16.ninfer \
  --binary build/apps/ninfer-embed --input texts.json --out vectors.json \
  --dimensions 1024
```

输出为归一化的 1024 维向量。检索时，对查询增加 `--instruction "Given a web search query, retrieve relevant passages that answer the query"`；待检索的文档直接输入正文。

默认每批最多 8 条文本、合计 32,768 个未填充 token；更多文本会自动分批。支持 32、128、256、512、1024 维输出。

[转换步骤与 Engine API](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-embedding.md)

## Transformers 与 NInfer 速度对比

两边均为 BF16，输出 1024 维归一化向量。时间包含完整语言 prefill 和向量池化的 GPU 计算，输入已预分词；加载权重、分词和首次捕获 Graph 不计入。Embedding 不生成文本，因此没有 decode 阶段。

原版每档取 eager 与完整编译路径中更快的一种。已确认 310 个参数和计算模块在 CUDA，关闭 MATH SDPA，使用快速 CUDA attention；完整编译没有图中断，错误抑制关闭。测试环境为 PyTorch 2.14.0+cu132、Transformers 5.17.0；5070 Ti 使用 CUDA 13.3，6000D 使用 CUDA 13.2。

| 显卡 | 文本数×token | TF GPU ms | NInfer GPU ms | TF token/s | NInfer token/s | 吞吐变化 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti 16 GB | 1×128 | 3.266 | 2.578 | 39,196 | 49,656 | +26.69% |
| RTX 5070 Ti 16 GB | 1×512 | 8.781 | 8.150 | 58,308 | 62,826 | +7.75% |
| RTX 5070 Ti 16 GB | 1×2048 | 42.161 | 37.610 | 48,576 | 54,454 | +12.10% |
| RTX 5070 Ti 16 GB | 4×128 | 7.418 | 7.340 | 69,019 | 69,753 | +1.06% |
| RTX 5070 Ti 16 GB | 8×512 | 62.459 | 56.905 | 65,579 | 71,979 | +9.76% |
| RTX 6000D | 1×128 | 2.797 | 1.980 | 45,763 | 64,657 | +41.29% |
| RTX 6000D | 1×512 | 6.187 | 4.616 | 82,757 | 110,912 | +34.02% |
| RTX 6000D | 1×2048 | 21.251 | 21.298 | 96,371 | 96,159 | -0.22% |

吞吐变化均相对 Transformers，计算为 `(TF 耗时 / NInfer 耗时 − 1) × 100%`。6000D 的 2048-token 输入两者基本持平。数据取中位数；NInfer 每档 30 次正式前向，本机长输入为 62 次。原版与 NInfer 在同一张卡、相同输入配置下分轮测量，6000D 的 2048-token 原版另以 31 次正式前向补测。

## 向量质量与结果一致率

对比原版 eager 和完整编译两条路径。向量对照共 249 条；语义相似度使用 STS-B 前 100 对；中英文检索集为 8 条查询。

| 显卡 | 原版路径 | 平均余弦 | TF Spearman | NInfer Spearman | Top1 正确数 | Top1 / Top3 一致率 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti 16 GB | eager | 0.99980168 | 0.946255 | 0.946417 | 8/8 | 100% / 100% |
| RTX 5070 Ti 16 GB | compiled | 0.99979238 | 0.946207 | 0.946417 | 8/8 | 100% / 100% |
| RTX 6000D | eager | 0.99980005 | 0.946255 | 0.946417 | 8/8 | 100% / 100% |
| RTX 6000D | compiled | 0.99978163 | 0.946561 | 0.946417 | 8/8 | 100% / 100% |

两边检索 Top1 都为 8/8，Top1 及 Top3 文档集合完全一致。浮点向量存在少量数值差异，平均余弦约 0.9998；Spearman 基本一致。两张卡各通过 164 项独立 FP64 数学验证。

## 显存与模型文件

GPU 权重存储约 1.110 GiB，无 KV 缓存。显式运行缓冲：4×128 约 15.04 MiB，8×512 约 120.14 MiB，32K 合计容量约 0.938 GiB；CUDA 驱动与 Graph 对象另有开销。

`qwen3-embedding-0.6b-bf16.ninfer` · 1,203,049,125 bytes · BF16 · Apache-2.0

SHA256: `3229813fdae278ab6871f712e8a943fc4c8949d5580567e89f27f2f18d4e6ae6`

[测量数据](gpu-results.json) · [模型清单](artifact-manifest.json) · [转换记录](conversion.json)

[Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [STS-B](https://huggingface.co/datasets/sentence-transformers/stsb)

## 当前主分支回测

覆盖 1–8,192 token、1/4/8 路和不等长批次。258 个向量位置在优化前后及 eager 模式下逐位一致；检索 8/8 正确，100 对 STS-B 文本的相关系数为 0.92563。没有确认的性能回退。
