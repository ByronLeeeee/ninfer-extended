# Xiaomi-OCR-0 BF16 MTP 模型

简体中文 · [English](README.md)

用于 OCR 和文档解析的 **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)** 模型，基于 Xiaomi-OCR-0 / Qwen3.5-0.8B-Base。
模型文件包含语言与视觉权重、原模型的 MTP 层、65,536 行 BF16 草稿输出头、tokenizer 和图像处理配置。
主模型、草稿权重和 KV 均保留 BF16。

## 安装与运行

使用 **[ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended) 的 main 分支**，推荐 MTP1：

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
pip install modelscope-hub
ms-hub download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16-MTP.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16-MTP.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy \
  --spec mtp --draft-tokens 1 --lm-head-draft
```

6000D 将 chunk 改为 `--prefill-chunk 2048`。将 `--draft-tokens 1` 改为 `--draft-tokens 3` 即可使用 MTP3。
模型只有一个训练好的 MTP 层，MTP3 会连续复用它预测三个位置。
删除 `--spec mtp --draft-tokens 1 --lm-head-draft` 则使用普通 BF16 解码；仅删除 `--lm-head-draft` 则用完整 BF16 输出头生成草稿。

构建需要 64 位 Linux（Windows 使用 WSL2）、支持 `sm_120a` 的 CUDA、C++20、CMake ≥3.28、Ninja、
FFmpeg 开发库、libcurl、PCRE2、cuDNN 9 和 cuBLAS。[转换与使用指南](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md)。

## 编译版 Transformers 与 BF16 MTP1

双方使用 BF16 权重和 KV、单路、4K 上下文、相同图片与提示词、greedy 解码。
每项预热两次，以 A-B-B-A/B-A-A-B 交替顺序正式测六次，取中位数。输出上限为 256 token：两张文字页完整输出，密集公式页比较相同的 256-token 前缀。

Transformers 的视觉、语言 prefill 和 decode 均已编译，causal-conv1d 和 Gated DeltaNet 使用融合算子，
运行时检查确认没有回退。Prefill 包含视觉编码与语言 prefill；Transformers 采用同步后的阶段计时，NInfer 使用 CUDA events。
Decode 从首 token 之后统计。5070 Ti 的 NInfer chunk 为 1,024，6000D 为 2,048。

| 显卡 | 输入 | TF prefill tok/s | MTP1 prefill tok/s | Prefill 变化 | TF decode tok/s | MTP1 decode tok/s | Decode 倍数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 21,238 | 23,553 | +10.9% | 162.7 | 707.5 | 4.35× |
| RTX 5070 Ti | 英文合同 | 21,003 | 23,812 | +13.4% | 155.2 | 728.1 | 4.69× |
| RTX 5070 Ti | 密集公式（256 token） | 21,937 | 21,350 | -2.7% | 159.6 | 748.5 | 4.69× |
| RTX 6000D | 中文文书 | 36,299 | 40,061 | +10.4% | 233.8 | 872.6 | 3.73× |
| RTX 6000D | 英文合同 | 36,311 | 40,212 | +10.7% | 233.4 | 926.6 | 3.97× |
| RTX 6000D | 密集公式（256 token） | 36,366 | 36,450 | +0.2% | 235.6 | 950.4 | 4.03× |

Prefill 变化按 `(MTP1 / TF − 1) × 100%` 计算，decode 倍数为 `MTP1 / TF`。

### 整次请求耗时

包含预处理、模型推理和本地 HTTP 请求：

| 显卡 | 输入 | TF 请求 ms | MTP1 请求 ms | 速度倍数 |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 866.2 | 322.6 | 2.68× |
| RTX 5070 Ti | 英文合同 | 963.2 | 331.0 | 2.91× |
| RTX 5070 Ti | 密集公式（256 token） | 1,823.8 | 566.5 | 3.22× |
| RTX 6000D | 中文文书 | 576.0 | 219.2 | 2.63× |
| RTX 6000D | 英文合同 | 576.5 | 210.7 | 2.74× |
| RTX 6000D | 密集公式（256 token） | 1,197.8 | 377.0 | 3.18× |

## 普通 BF16、MTP1 与 MTP3

普通解码与 MTP 使用相同的 NInfer Extended 程序和模型文件。每个 MTP 窗口分别与普通解码配对测试。
接受率是接受的草稿 token 数除以提出的草稿 token 数，不包含目标模型自身生成的 token。

| 显卡 | 输入 | 普通 decode tok/s | MTP1 decode tok/s | MTP1 接受率 | MTP3 decode tok/s | MTP3 接受率 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 435.0 | 662.5 | 80.6% | 674.0 | 39.7% |
| RTX 5070 Ti | 英文合同 | 434.5 | 701.1 | 93.2% | 680.8 | 40.5% |
| RTX 5070 Ti | 密集公式（256 token） | 432.1 | 721.0 | 97.7% | 761.6 | 49.5% |
| RTX 6000D | 中文文书 | 583.3 | 872.8 | 80.6% | 838.3 | 39.7% |
| RTX 6000D | 英文合同 | 582.6 | 923.4 | 93.2% | 863.7 | 40.5% |
| RTX 6000D | 密集公式（256 token） | 580.2 | 956.5 | 97.7% | 960.0 | 49.5% |

MTP3 各位置的接受比例，以全部验证轮次为分母；后一个位置被接受，也要求前面的草稿通过：

| 显卡 | 输入 | 第一个位置 | 第二个位置 | 第三个位置 |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 82.7% | 32.7% | 3.8% |
| RTX 5070 Ti | 英文合同 | 90.2% | 25.5% | 5.9% |
| RTX 5070 Ti | 密集公式（256 token） | 97.1% | 37.9% | 12.6% |
| RTX 6000D | 中文文书 | 82.7% | 32.7% | 3.8% |
| RTX 6000D | 英文合同 | 90.2% | 25.5% | 5.9% |
| RTX 6000D | 密集公式（256 token） | 97.1% | 37.9% | 12.6% |

### MTP1 与 MTP3 直接对比

两种窗口直接交替运行，每项各正式测六次：

| 显卡 | 输入 | MTP1 tok/s | MTP3 tok/s | MTP3 decode 变化 | MTP3 整次请求吞吐变化 |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 719.1 | 705.2 | -1.9% | -0.0% |
| RTX 5070 Ti | 英文合同 | 759.6 | 723.1 | -4.8% | -2.3% |
| RTX 5070 Ti | 密集公式（256 token） | 780.9 | 803.1 | +2.8% | +2.2% |
| RTX 6000D | 中文文书 | 873.1 | 839.5 | -3.9% | -3.2% |
| RTX 6000D | 英文合同 | 927.7 | 863.8 | -6.9% | -4.2% |
| RTX 6000D | 密集公式（256 token） | 955.0 | 958.7 | +0.4% | +0.5% |

MTP1 在文字页和四路吞吐上更稳定。MTP3 在密集公式前缀的单请求测试中还有小幅收益；
完整千字文样例在 6000D 上的 MTP3 接受率为 18.8%，整次请求吞吐比普通解码低 1.6%。
所有草稿仍由完整词表的目标模型验证。

### 四路整次请求吞吐

四个请求的输出 token 总数除以全部完成的实际耗时，包含 prefill 和 HTTP 开销。
下表是四路总吞吐：

| 显卡 | 输入 × 4 | 普通输出 tok/s | MTP1 输出 tok/s | MTP1 变化 | MTP3 输出 tok/s | MTP3 变化 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 620.6 | 703.0 | +13.3% | 674.5 | +9.2% |
| RTX 5070 Ti | 英文合同 | 599.9 | 698.1 | +16.4% | 660.1 | +6.7% |
| RTX 5070 Ti | 密集公式（256 token） | 817.7 | 988.5 | +20.9% | 957.6 | +17.2% |
| RTX 6000D | 中文文书 | 1,007.4 | 1,167.8 | +15.9% | 1,103.9 | +9.6% |
| RTX 6000D | 英文合同 | 1,016.7 | 1,210.3 | +19.0% | 1,118.2 | +9.9% |
| RTX 6000D | 密集公式（256 token） | 1,249.2 | 1,592.9 | +27.5% | 1,510.6 | +20.9% |

## 正确率与结果一致率

三项 Transformers/MTP1 测速输入在两张显卡上均为 **3/3 原始输出一致（100%）**。
其中两张完整文字页共 478 个标注字符，双方均为 **100% 字符正确率，CER 0%**。

完整文档测试比较 MTP 与 NInfer Extended 普通解码：十一页、16K 上下文、4,096-token 输出上限，
每种窗口、每张显卡各 44 次正式请求，全部自然结束。六张有标注的文字图片共 1,117 个评分字符，其他图片比较结果一致率。

| 显卡 | 模式 | 标注文字正确率 | 完整页原始输出一致率 | 字符差异率 |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | MTP1 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |
| RTX 5070 Ti | MTP3 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |
| RTX 6000D | MTP1 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |
| RTX 6000D | MTP3 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |

单路的三处差异是手写公式中的 `P/p` 大小写。评分统一 NFKC 字符格式并去掉空白；
文字 CER 另外去掉 Markdown 标题与强调标记，保留大小写。

混合文档并发使用 32K 上下文，依次测试四、二、三路，含九张不同图片，每种模式、每张显卡各 18 次正式输出：

| 显卡 | 模式 | 混合并发原始输出一致率 | 字符差异率 |
| --- | --- | --- | --- |
| RTX 5070 Ti | MTP1 | 16/18 (88.9%) | 0.1081% |
| RTX 5070 Ti | MTP3 | 18/18 (100.0%) | 0.0000% |
| RTX 6000D | MTP1 | 15/18 (83.3%) | 0.2163% |
| RTX 6000D | MTP3 | 18/18 (100.0%) | 0.0000% |

混合并发的 MTP1 差异出现在公式大小写、LaTeX 上标和撇号。
固定单路下，两张显卡的 **11/11 页 MTP1 输出在 eager 与 CUDA Graph 下完全一致**。
GDN 接受前缀状态恢复、MTP packing 和 BF16 草稿投影均通过独立数值校验及更换输入后的 Graph 检查。

## 显存占用

各模式单独启动，预热两次，4K 上下文：

| 显卡 | 并发 | 普通 MiB | MTP1 MiB | MTP3 MiB | MTP1 增量 MiB |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 1 | 2071 | 2249 | 2249 | 178 |
| RTX 5070 Ti | 4 | 2275 | 2483 | 2491 | 208 |
| RTX 6000D | 1 | 2317 | 2497 | 2497 | 180 |
| RTX 6000D | 4 | 2523 | 2731 | 2739 | 208 |

数据为相对系统空闲状态增加的显存，5070 Ti 使用 WDDM/WSL，按整张 GPU 的显存差值统计。
原模型的 564 个参数绑定保持相同数值。65,536 行草稿输出头按上游词频排名选择，逐行保留原输出头的 BF16 位值。

## 模型文件与许可证

`xiaomi-ocr-0-bf16-MTP.ninfer` · **1,901,499,904 bytes** · BF16 · Apache-2.0

SHA256：`c0d0e1ec96e45661657fb9ac31753a46dac5545f35a4ebc993345c5f0e9e2472`

[测量数据](../../docs/results/xiaomi-ocr-mtp-bf16.json) · [模型清单](https://modelscope.cn/models/ByronLeeee/Xiaomi-OCR-0-Ninfer/file/view/master/artifact-manifest.json?status=1)
