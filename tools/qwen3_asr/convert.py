"""Losslessly import Qwen3-ASR BF16 parameters into a native v3 artifact."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
config = json.loads((a.model / 'config.json').read_text())
assert config['architectures'] == ['Qwen3ASRForConditionalGeneration']
source = a.model / 'model.safetensors'
with source.open('rb') as stream:
    header_bytes = struct.unpack('<Q', stream.read(8))[0]
    header = json.loads(stream.read(header_bytes))
header.pop('__metadata__', None)
assert all(row['dtype'] == 'BF16' for row in header.values())

groups = {}
removed = set()
for component, count in [('audio_tower', config['audio_config']['encoder_layers']),
                         ('language_model', config['text_config']['num_hidden_layers'])]:
    for index in range(count):
        prefix = f'model.{component}.layers.{index}'
        for suffix in (['weight', 'bias'] if component == 'audio_tower' else ['weight']):
            names = [f'{prefix}.self_attn.{kind}_proj.{suffix}' for kind in ['q', 'k', 'v']]
            groups[f'{prefix}.self_attn.qkv_proj.{suffix}'] = names
            removed.update(names)
        if component == 'language_model':
            names = [f'{prefix}.mlp.{kind}_proj.weight' for kind in ['gate', 'up']]
            groups[f'{prefix}.mlp.gate_up_proj.weight'] = names
            removed.update(names)

specs = []
bindings = {}
objects = {}
for name, row in header.items():
    if name not in removed:
        objects[name] = [name]
for name, names in groups.items():
    objects[name] = names
for name, names in objects.items():
    shape = list(header[names[0]]['shape'])
    shape[0] = sum(header[part]['shape'][0] for part in names)
    assert all(header[part]['shape'][1:] == shape[1:] for part in names)
    specs.append(TensorSpec(name, tuple(shape), 'bf16', 'contiguous_le_v1'))
    bindings[name] = {'object': name}

audio = config['audio_config']
channels = audio['d_model']
inv = torch.exp(-np.log(10000) / (channels // 2 - 1) * torch.arange(channels // 2).float())
scaled = torch.arange(audio['max_position_embeddings'])[:, None] * inv[None, :]
position = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
specs.append(TensorSpec('audio.position_embedding', (audio['max_position_embeddings'], channels), 'bf16', 'contiguous_le_v1'))
bindings['audio.position_embedding'] = {'object': 'audio.position_embedding'}
rope = 1.0 / (config['text_config']['rope_parameters']['rope_theta'] **
              (torch.arange(0, config['text_config']['head_dim'], 2).float() / config['text_config']['head_dim']))
rope_bytes = rope.numpy().tobytes()
specs.append(TensorSpec('text.rope_inv_freq', (len(rope),), 'fp32', 'contiguous_le_v1'))
bindings['text.rope_inv_freq'] = {'object': 'text.rope_inv_freq'}
resources = {name: (a.model / name).read_bytes() for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json', 'processor_config.json']}
for name, data in resources.items():
    specs.append(ResourceSpec('resource/' + name, len(data)))
components = {'text': {'config': config, 'resources': {name: 'resource/' + name for name in resources}}}
report = {'source': 'Qwen/Qwen3-ASR-1.7B-hf', 'dtype': 'bf16', 'weight_quantization': False,
          'source_tensors': len(header), 'artifact_tensors': len(objects) + 2, 'lossless_row_concatenations': len(groups),
          'architecture': config['architectures'][0], 'source_bytes': source.stat().st_size}
with source.open('rb') as stream, ArtifactWriter(a.out, specs, components=components, bindings=bindings,
        metadata={'name': 'qwen3-asr-1.7b-bf16'}, provenance=report) as writer:
    for name, names in objects.items():
        output_offset = 0
        for part in names:
            start, end = header[part]['data_offsets']
            stream.seek(8 + header_bytes + start)
            remaining = end - start
            while remaining:
                data = stream.read(min(8 * 1024 * 1024, remaining))
                assert data
                writer.write_region(name, output_offset, data)
                output_offset += len(data)
                remaining -= len(data)
    writer.write_object('audio.position_embedding', position)
    writer.write_object('text.rope_inv_freq', rope_bytes)
    for name, data in resources.items():
        writer.write_object('resource/' + name, data)
report['artifact_bytes'] = a.out.stat().st_size
a.out.with_suffix('.conversion.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
