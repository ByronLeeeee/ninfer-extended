# Xiaomi-OCR-0 BF16 · NInfer

适用于 **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)** 的小米 OCR 模型，
基于 Qwen3.5-0.8B-Base。模型文件包含原始语言、视觉和 MTP 权重、**128K 行 BF16 草稿输出头**，
以及分词和图像处理配置。权重与 KV 均为 BF16，草稿由主模型完整词表验证。

## 使用方法

需要当前 **ninfer-extended 主分支**，推荐使用 **MTP1**。

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
# 从本仓库下载 xiaomi-ocr-0-bf16-MTP.ninfer，放入 models 目录
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16-MTP.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy \
  --spec mtp --draft-tokens 1 --lm-head-draft
```

6000D 将 `--prefill-chunk` 设为 `2048`。尝试 MTP3 时，把 `--draft-tokens 1` 改成
`--draft-tokens 3`；它连续调用模型原有的单层 MTP。去掉三个 MTP 参数可使用普通解码，
去掉 `--lm-head-draft` 则用完整输出头生成草稿。

编译需要 64 位 Linux（Windows 使用 WSL2）、支持 `sm_120a` 的 CUDA、C++20、CMake ≥3.28、
Ninja、FFmpeg 开发库、libcurl、PCRE2、cuDNN 9 和 cuBLAS。
[详细用法](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md) · [English](README.md)

## Prefill 和 decode 速度

对比编译版 BF16 Transformers 和 NInfer MTP1。双方输入相同，单路、4K 上下文、贪心解码，
输出上限为 256 token。Prefill 包含视觉编码和语言 prefill，decode 从首 token 之后计时，
保留 CUDA Graph。Transformers 已启用 fused causal-conv1d/Gated DeltaNet，视觉、语言
prefill 和 decode 均编译执行。NInfer 阶段用 CUDA event 计时，Transformers 用同步后的墙钟时间。
每项预热两次，再交替运行三个 A-B-B-A/B-A-A-B 周期，表中取中位数。

| 显卡 | 输入 | TF prefill tok/s | MTP1 prefill tok/s | Prefill 变化 | TF decode tok/s | MTP1 decode tok/s | Decode 倍数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 21,764 | 24,168 | +11.0% | 166.4 | 717.1 | 4.31× |
| RTX 5070 Ti | 英文合同 | 22,066 | 23,383 | +6.0% | 163.3 | 718.6 | 4.40× |
| RTX 5070 Ti | 密集公式（256 token） | 21,353 | 22,204 | +4.0% | 162.3 | 717.6 | 4.42× |
| RTX 6000D | 中文文书 | 36,383 | 40,038 | +10.0% | 232.8 | 904.4 | 3.88× |
| RTX 6000D | 英文合同 | 36,337 | 40,141 | +10.5% | 231.9 | 916.9 | 3.95× |
| RTX 6000D | 密集公式（256 token） | 36,254 | 36,284 | +0.1% | 235.4 | 910.8 | 3.87× |

完整请求耗时包含图像准备、推理和本地 HTTP 往返：

| 显卡 | 输入 | TF 请求 ms | MTP1 请求 ms | 速度倍数 |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 879.0 | 333.5 | 2.64× |
| RTX 5070 Ti | 英文合同 | 890.3 | 326.1 | 2.73× |
| RTX 5070 Ti | 密集公式（256 token） | 1818.2 | 575.7 | 3.16× |
| RTX 6000D | 中文文书 | 577.2 | 214.3 | 2.69× |
| RTX 6000D | 英文合同 | 584.2 | 212.4 | 2.75× |
| RTX 6000D | 密集公式（256 token） | 1202.7 | 388.8 | 3.09× |

两张显卡上的三项测速输出均与编译版 Transformers 相同。

## MTP1 和 MTP3

以下分别与同一版 NInfer 的普通解码做成对测试。Decode 是每路速度，完整请求吞吐包含
prefill 和 decode。

| 显卡 | 输入 | 普通 → MTP1 decode tok/s | 完整请求吞吐变化 | 普通 → MTP3 decode tok/s | 完整请求吞吐变化 |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 435.1 → 682.8 | +25.8% | 462.9 → 646.5 | +19.1% |
| RTX 5070 Ti | 英文合同 | 434.2 → 688.1 | +26.7% | 454.0 → 660.2 | +22.7% |
| RTX 5070 Ti | 密集公式（256 token） | 425.1 → 678.0 | +39.0% | 444.3 → 670.7 | +32.6% |
| RTX 6000D | 中文文书 | 586.3 → 909.1 | +30.9% | 588.8 → 807.6 | +22.7% |
| RTX 6000D | 英文合同 | 586.1 → 914.8 | +32.9% | 585.6 → 750.8 | +17.9% |
| RTX 6000D | 密集公式（256 token） | 580.8 → 913.8 | +41.3% | 581.3 → 845.4 | +33.3% |

四路同时请求，按总输出 token 数除以整组请求完成时间计算吞吐：

| 显卡 | 输入 | 普通 → MTP1 输出 tok/s | 变化 | 普通 → MTP3 输出 tok/s | 变化 |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 590.6 → 676.8 | +14.6% | 626.8 → 581.8 | -7.2% |
| RTX 5070 Ti | 英文合同 | 610.7 → 678.8 | +11.2% | 633.0 → 603.2 | -4.7% |
| RTX 5070 Ti | 密集公式（256 token） | 818.9 → 952.3 | +16.3% | 817.0 → 765.6 | -6.3% |
| RTX 6000D | 中文文书 | 995.8 → 1164.3 | +16.9% | 817.0 → 841.0 | +2.9% |
| RTX 6000D | 英文合同 | 1002.8 → 1181.0 | +17.8% | 860.1 → 726.9 | -15.5% |
| RTX 6000D | 密集公式（256 token） | 1226.4 → 1509.1 | +23.1% | 1034.4 → 943.3 | -8.8% |

MTP1 在已测的单路和四路任务中均有提升。MTP3 单路可以提速，四路通常更慢，因此默认使用 MTP1。

完整页面的草稿接受率如下。输出上限为 4,096 token，所有页面均自然结束：

| 显卡 | 完整输入 | MTP1 草稿接受率 | MTP3 草稿接受率 |
| --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 96.49% | 45.83% |
| RTX 5070 Ti | 英文合同 | 100.00% | 46.81% |
| RTX 5070 Ti | 古文书法 | 78.57% | 26.19% |
| RTX 5070 Ti | 密集公式（完整页） | 96.02% | 46.73% |
| RTX 5070 Ti | 手写公式 | 98.61% | 47.60% |
| RTX 5070 Ti | 收据 | 94.12% | 44.27% |
| RTX 6000D | 中文文书 | 96.49% | 45.83% |
| RTX 6000D | 英文合同 | 100.00% | 46.81% |
| RTX 6000D | 古文书法 | 78.57% | 26.19% |
| RTX 6000D | 密集公式（完整页） | 96.02% | 46.73% |
| RTX 6000D | 手写公式 | 98.61% | 47.60% |
| RTX 6000D | 收据 | 94.12% | 44.27% |

草稿接受率是被主模型接受的草稿 token 数占全部草稿的比例。MTP3 统计三个位置的总和；
这个指标反映草稿质量，OCR 正确率见下表。

## 正确率与结果一致率

以下结果在 5070 Ti 和 6000D 上均通过：

| 测试 | 普通 BF16 | MTP1 | MTP3 |
| --- | --- | --- | --- |
| 6 张标注文字图片，共 1,117 个计分字符 | CER 0% | CER 0% | CER 0% |
| 11 张完整页面，与普通解码的原始输出一致率 | 参考结果 | 11/11（100%） | 11/11（100%） |
| 完整页面归一化字符差异率，共 7,269 字符 | 参考结果 | 0% | 0% |
| 四轮混合并发，每组 4/2/3 路，原始输出一致率 | 参考结果 | 36/36（100%） | 36/36（100%） |
| 混合并发归一化字符差异率，共 27,708 字符 | 参考结果 | 0% | 0% |
| 四轮混合并发中输出始终相同的图片数 | 9/9 | 9/9 | 9/9 |

完整页面使用 16K 上下文，混合并发使用每路 32K，均为 BF16 KV、CUDA Graph。
书法、收据和公式页用于检查结果一致性；CER 按六张带标注的文字图片计算。
投影、attention、GDN 状态和词表选择还通过了独立 FP64 数值校验、改变输入后的 CUDA Graph
重放测试，以及逐列与单 token 执行的对照。

## 模型文件

- 文件名：`xiaomi-ocr-0-bf16-MTP.ninfer`
- 大小：**2,035,979,776 字节**（2.036 GB）。
- SHA-256：`0881cb5e19af3583dc6c814f65c24dbeab3ab0c3e9e9bc7601fa4c9b975ec659`。
- 原始语言、视觉和 MTP 张量逐字节保留；草稿输出头按上游 token 排序，直接复制原始输出头对应行。
- 同一文件可运行普通解码、MTP1 和 MTP3；仓库另有不含 MTP 的 BF16 文件。
- 许可证：Apache-2.0，与原模型一致。
