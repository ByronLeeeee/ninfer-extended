---
license: apache-2.0
base_model: SeerRay-Lab/Xiaomi-OCR-0
pipeline_tag: image-text-to-text
tags:
  - ninfer
  - bf16
  - ocr
  - document-parsing
  - qwen3_5
  - cuda
---

# Xiaomi-OCR-0 BF16 NInfer

这是 [SeerRay-Lab/Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0)
的 BF16 NInfer v3 格式转换，包含文本、视觉权重和内嵌 tokenizer/processor 资源。
没有重新训练；投影矩阵使用 BF16/A16，数学标量保留 NInfer 转换规则所需格式。
共享 embedding/output head 继续共享。

**必须依赖 [ByronLeeeee/ninfer-xiaomi-ocr-0](https://github.com/ByronLeeeee/ninfer-xiaomi-ocr-0)
仓库构建的运行时。上游原版 NInfer 没有这个模型所需的小模型形状、视觉 attention、
卷积和 tokenizer 适配；不能假定上游原版二进制可直接加载运行。**
这个文件不是 Transformers safetensors 或 GGUF。

## 使用

先从上面的 fork 构建运行时，再把本仓库的 `xiaomi-ocr-0-bf16.ninfer` 下载到 `models/`。
详细的系统依赖、转换步骤、编译基线和验证脚本见
[适配文档](https://github.com/ByronLeeeee/ninfer-xiaomi-ocr-0/blob/codex/xiaomi-ocr-0/docs/xiaomi-ocr.md)。

```bash
git clone https://github.com/ByronLeeeee/ninfer-xiaomi-ocr-0.git
cd ninfer-xiaomi-ocr-0
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy
```

推荐 OCR 服务采用 32K 上下文、4 路并发、BF16 KV；这是部署示例。
256K 并非单页 OCR 的必要配置，每张图片的 token 数取决于分辨率与网格。
FP8 KV 可以降低长上下文的缓存占用，但需要单独评估输出差异。

## 兼容性与验证

已验证 RTX 5070 Ti 16GB（WSL2）和 RTX 6000D（Linux），计算能力均为 12.0。
运行时编译目标是 `sm_120a`；不保证所有 Blackwell 设备均可使用，B200/GB200
不是同一个编译架构。完整源码使用 CUDA 13.2、GCC 15.2 进行干净构建。

Python 参考实现的语言 prefill 和视觉 GPU 计算已在两台机器上分别编译。
这属于 Transformers 基线工具的优化，NInfer 本身使用 C++/CUDA 路径。
实际速度、显存和输出比较见
[测试报告](https://github.com/ByronLeeeee/ninfer-xiaomi-ocr-0/blob/codex/xiaomi-ocr-0/docs/xiaomi-ocr-performance.md)。
测试使用 BF16、4K 容量、单请求和最多 256 输出 token，不能套用到 32K/4 路配置。
三份测试输出一致，其中公式图片只比较了 256-token 前缀；输出一致不等于 OCR 准确率 100%。
历史 11 张完整输出测试有一张手写公式存在差异，报告保留了这个限制。

## 文件来源

- 原模型：`SeerRay-Lab/Xiaomi-OCR-0`。
- 原权重 revision：`e4d1c4a6804bd9ef342b93d705a73af003e2ef4e`。
- NInfer 上游基础：`Neroued/ninfer`，`594930e7b609efa4bcea3ae4f24cd9d66b5f224f`。
- 转换文件：`xiaomi-ocr-0-bf16.ninfer`，1,726,110,464 字节。
- SHA256：`1db6a88a5947ba06e88210feff0c44f8eeee6d7b506b4744cbf9f94a0e61cb9d`。

原模型和运行时遵循 Apache-2.0；原作者的许可证、署名和模型说明继续适用。
