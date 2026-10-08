"""Lossless BF16 import of the official Qwen3 forced-alignment checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter


def convert(model, out):
    original = json.loads((model / 'config.json').read_text())
    assert original['architectures'] == ['Qwen3ASRForConditionalGeneration']
    thinker = original['thinker_config']
    assert thinker['model_type'] == 'qwen3_forced_aligner'
    config = copy.deepcopy(thinker)
    config.update(architectures=original['architectures'], speech_task='forced_alignment', audio_attention_mode='full_sample',
                  timestamp_token_id=original['timestamp_token_id'],
                  timestamp_segment_time=original['timestamp_segment_time'],
                  support_languages=original['support_languages'], tie_word_embeddings=False)
    text, audio = config['text_config'], config['audio_config']
    text['rope_parameters'] = {'rope_type': 'default', 'rope_theta': text['rope_theta']}
    text['use_sliding_window'] = False
    audio['max_position_embeddings'] = 13
    source = model / 'model.safetensors'
    with source.open('rb') as stream:
        header_bytes = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_bytes))
    header.pop('__metadata__', None)
    assert all(row['dtype'] == 'BF16' for row in header.values())

    def native_name(name):
        if name.startswith('thinker.audio_tower.proj1.'):
            return name.replace('thinker.audio_tower.proj1.', 'model.multi_modal_projector.linear_1.')
        if name.startswith('thinker.audio_tower.proj2.'):
            return name.replace('thinker.audio_tower.proj2.', 'model.multi_modal_projector.linear_2.')
        return name.replace('thinker.audio_tower.', 'model.audio_tower.').replace(
            'thinker.model.', 'model.language_model.').replace('thinker.lm_head.', 'model.timestamp_head.')

    groups, removed = {}, set()
    for component, count in [('audio_tower', audio['encoder_layers']),
                             ('model', text['num_hidden_layers'])]:
        for index in range(count):
            prefix = f'thinker.{component}.layers.{index}'
            for suffix in (['weight', 'bias'] if component == 'audio_tower' else ['weight']):
                names = [f'{prefix}.self_attn.{kind}_proj.{suffix}' for kind in ['q', 'k', 'v']]
                groups[native_name(f'{prefix}.self_attn.qkv_proj.{suffix}')] = names
                removed.update(names)
            if component == 'model':
                names = [f'{prefix}.mlp.{kind}_proj.weight' for kind in ['gate', 'up']]
                groups[native_name(f'{prefix}.mlp.gate_up_proj.weight')] = names
                removed.update(names)
    objects = {native_name(name): [name] for name in header if name not in removed}
    objects.update(groups)
    specs, bindings = [], {}
    for name, names in objects.items():
        shape = list(header[names[0]]['shape'])
        shape[0] = sum(header[part]['shape'][0] for part in names)
        assert all(header[part]['shape'][1:] == shape[1:] for part in names)
        # Zero rows permit the existing aligned native linear geometry. Only
        # the original 5,000 classes participate in timestamp argmax.
        if name == 'model.timestamp_head.weight':
            assert shape == [5000, 1024]
            shape[0] = 5120
        specs.append(TensorSpec(name, tuple(shape), 'bf16', 'contiguous_le_v1'))
        bindings[name] = {'object': name}
    width = audio['d_model']
    inv = torch.exp(-np.log(10000) / (width // 2 - 1) * torch.arange(width // 2).float())
    scaled = torch.arange(13)[:, None] * inv[None, :]
    positions = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
    rope = (1 / (text['rope_theta'] ** (torch.arange(0, text['head_dim'], 2).float() / text['head_dim']))).numpy().tobytes()
    for name, shape, dtype in [('audio.position_embedding', (13, width), 'bf16'),
                               ('text.rope_inv_freq', (text['head_dim']//2,), 'fp32')]:
        specs.append(TensorSpec(name, shape, dtype, 'contiguous_le_v1'))
        bindings[name] = {'object': name}
    resources = {name: (model/name).read_bytes() for name in
                 ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                  'preprocessor_config.json', 'chat_template.json']}
    for name, data in resources.items():
        specs.append(ResourceSpec('resource/'+name, len(data)))
    components = {'text': {'config': config, 'resources': {n:'resource/'+n for n in resources}}}
    report = {'source': 'Qwen/Qwen3-ForcedAligner-0.6B', 'dtype': 'bf16',
              'weight_quantization': False, 'source_tensors': len(header),
              'lossless_row_concatenations': len(groups), 'zero_head_rows': 120}
    out.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as stream, ArtifactWriter(out, specs, components=components,
            bindings=bindings, metadata={'name': 'qwen3-forced-aligner-0.6b-bf16'}, provenance=report) as writer:
        for name, names in objects.items():
            offset = 0
            for part in names:
                start, end = header[part]['data_offsets']
                stream.seek(8+header_bytes+start)
                remaining = end-start
                while remaining:
                    data = stream.read(min(8*1024*1024, remaining))
                    assert data
                    writer.write_region(name, offset, data)
                    offset += len(data)
                    remaining -= len(data)
            if name == 'model.timestamp_head.weight':
                writer.write_region(name, offset, bytes(120*1024*2))
        writer.write_object('audio.position_embedding', positions)
        writer.write_object('text.rope_inv_freq', rope)
        for name, data in resources.items():
            writer.write_object('resource/'+name, data)
    report['artifact_bytes'] = out.stat().st_size
    out.with_suffix('.conversion.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    convert(a.model, a.out)
