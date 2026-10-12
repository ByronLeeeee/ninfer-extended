# Xiaomi-OCR-0 · NInfer

[English](README.md)

## 模型介绍

这是 [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) 的 BF16 NInfer 版本，基于 Qwen3.5-0.8B-Base，支持文字识别、文档解析、表格和公式提取。模型通过 **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)** 运行，视觉编码、prefill 和 decode 均使用原生 C++/CUDA 算子。

模型文件包含语言、视觉和 MTP 权重，以及 tokenizer 和图像处理配置。推荐使用 **MTP1**，同一文件也支持普通解码和 MTP3。

## 速度对比

### Prefill 与 decode

BF16 权重和 KV，单请求，4K 上下文，greedy 解码；数据为预热后的中位数。Transformers 使用编译后的视觉、prefill 和 decode，decode 启用 CUDA Graph。Prefill 包含视觉编码和语言前向，公式样例生成 256 token。

| GPU | 输入 | Transformers prefill (tok/s) | NInfer MTP1 prefill (tok/s) | Transformers decode (tok/s) | NInfer MTP1 decode (tok/s) | Decode 倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 中文文档 | 21,764 | 24,168 | 166.4 | **717.1** | 4.31× |
| RTX 5070 Ti | 英文合同 | 22,066 | 23,383 | 163.3 | **718.6** | 4.40× |
| RTX 5070 Ti | 公式（256 token） | 21,353 | 22,204 | 162.3 | **717.6** | 4.42× |
| RTX 6000D | 中文文档 | 36,383 | 40,038 | 232.8 | **904.4** | 3.88× |
| RTX 6000D | 英文合同 | 36,337 | 40,141 | 231.9 | **916.9** | 3.95× |
| RTX 6000D | 公式（256 token） | 36,254 | 36,284 | 235.4 | **910.8** | 3.87× |

### 完整请求耗时

包含图片处理、模型推理和本地 HTTP 请求。

| GPU | 输入 | Transformers (ms) | NInfer MTP1 (ms) | 速度倍数 |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | 中文文档 | 879.0 | **333.5** | 2.64× |
| RTX 5070 Ti | 英文合同 | 890.3 | **326.1** | 2.73× |
| RTX 5070 Ti | 公式（256 token） | 1818.2 | **575.7** | 3.16× |
| RTX 6000D | 中文文档 | 577.2 | **214.3** | 2.69× |
| RTX 6000D | 英文合同 | 584.2 | **212.4** | 2.75× |
| RTX 6000D | 公式（256 token） | 1202.7 | **388.8** | 3.09× |

### Transformers MTP1 对比

Transformers MTP1 使用编译后的 PyTorch MTP 适配器，草稿输出头与 NInfer 相同。完整实现见 [MTP 对比说明](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr-transformers-mtp.md)。

| GPU | 输入 | Transformers MTP1 prefill (tok/s) | NInfer MTP1 prefill (tok/s) | Transformers MTP1 decode (tok/s) | NInfer MTP1 decode (tok/s) | Decode 倍数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | 中文文档 | 19,697 | 23,100 | 265.5 | **712.6** | 2.68× |
| RTX 5070 Ti | 英文合同 | 19,264 | 24,175 | 278.6 | **717.3** | 2.57× |
| RTX 5070 Ti | 公式（256 token） | 19,427 | 22,041 | 283.1 | **703.9** | 2.49× |
| RTX 6000D | 中文文档 | 32,722 | 40,214 | 366.2 | **905.5** | 2.47× |
| RTX 6000D | 英文合同 | 32,657 | 40,065 | 380.0 | **914.1** | 2.41× |
| RTX 6000D | 公式（256 token） | 32,918 | 36,506 | 380.0 | **914.9** | 2.41× |

## 正确率测试

5070 Ti 和 6000D 的结果如下。CER 与人工标注比较；MTP 的输出一致率和字符差异率以普通 NInfer 解码为参考。

| 测试 | 普通 BF16 | MTP1 | MTP3 |
| --- | ---: | ---: | ---: |
| 6 张标注文字图片，1,117 个字符：CER | 0% | 0% | 0% |
| 11 张完整页面：原始输出一致率 | 参考 | 100% (11/11) | 100% (11/11) |
| 完整页面，7,269 个字符：字符差异率 | 参考 | 0% | 0% |
| 36 次混合并发请求：原始输出一致率 | 参考 | 100% (36/36) | 100% (36/36) |
| 混合并发，27,708 个字符：字符差异率 | 参考 | 0% | 0% |

速度测试中的中文文档、英文合同和 256 token 公式输出，在两张显卡上均与 Transformers 完全一致；Transformers MTP1 的输出也一致。

### MTP 接受率

完整页面的草稿 token 接受率如下，5070 Ti 和 6000D 结果相同。

| 页面 | MTP1 | MTP3 |
| --- | ---: | ---: |
| 中文文档 | 96.49% | 45.83% |
| 英文合同 | 100.00% | 46.81% |
| 书法 | 78.57% | 26.19% |
| 公式 | 96.02% | 46.73% |
| 手写公式 | 98.61% | 47.60% |
| 收据 | 94.12% | 44.27% |

MTP1 在单请求和四路并发测试中均有加速，推荐作为默认配置。

## 使用方法

### 安装与编译

运行环境：Linux 或 WSL2、支持 `sm_120a` 的 CUDA、cuDNN 9 和 cuBLAS。构建依赖见 [项目说明](https://github.com/ByronLeeeee/ninfer-extended#build)。

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
```

### 下载并启动模型

```bash
pip install huggingface_hub
hf download ByronLeeee/Xiaomi-OCR-0-Ninfer \
  xiaomi-ocr-0-bf16-MTP.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16-MTP.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy \
  --spec mtp --draft-tokens 1 --lm-head-draft
```

6000D 使用 `--prefill-chunk 2048`。MTP3 将 `--draft-tokens` 改为 `3`；普通解码移除 `--spec mtp --draft-tokens 1 --lm-head-draft`。

### 识别图片

将待识别图片保存为 `page.png`，执行以下 Python 示例：

```python
import base64
import json
import urllib.request
from pathlib import Path

image = base64.b64encode(Path("page.png").read_bytes()).decode()
body = {
    "model": "ocr",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
        {"type": "text", "text": "Extract the text in the image."}
    ]}],
    "temperature": 0,
    "max_tokens": 4096,
}
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

可按任务替换提示词，提取文字、表格、公式或整页文档。更多示例见 [OCR 使用说明](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md)。

## 模型文件

| 文件 | 大小 | 精度 |
| --- | ---: | --- |
| `xiaomi-ocr-0-bf16-MTP.ninfer` | 2.036 GB | BF16 |

许可证：Apache-2.0。

[完整测试数据](gpu-results.json) · [Transformers MTP 数据](transformers-mtp-results.json) · [模型清单](artifact-manifest.json)
