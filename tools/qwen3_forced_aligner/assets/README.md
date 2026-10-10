# Korean word segmentation

`korean_dict_jieba.dict` is the unmodified dictionary distributed with
Qwen3-ASR 0.0.6 by Alibaba Cloud under the Apache License 2.0.
Source: [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/assets/korean_dict_jieba.dict).
The upstream license is included in `LICENSE.Qwen3-ASR`.

The alignment frontend supplies its words as `LTokenizer` scores, preserving
the original checkpoint's Korean word boundaries with the native Transformers
processor.
