"""Import Qwen3-Embedding BF16 weights without quantization or numerical changes."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.artifact.reader import Artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.model / 'config.json').read_text())
    if config['model_type'] != 'qwen3' or config['architectures'] != ['Qwen3ForCausalLM']:
        raise ValueError('Expected a Qwen3-Embedding decoder checkpoint')
    source = args.model / 'model.safetensors'
    with source.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    header.pop('__metadata__', None)
    # Embedding checkpoints serialize Qwen3Model without the CausalLM 'model.' prefix.
    header = {(name if name.startswith('model.') else 'model.' + name): row
              for name, row in header.items()}
    if any(row['dtype'] != 'BF16' for row in header.values()):
        raise ValueError('Every imported parameter must already be BF16')
    groups, removed = {}, set()
    for i in range(config['num_hidden_layers']):
        prefix = f'model.layers.{i}.'
        for target, parts in [('self_attn.qkv_proj', ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']),
                              ('mlp.gate_up_proj', ['mlp.gate_proj', 'mlp.up_proj'])]:
            names = [prefix + part + '.weight' for part in parts]
            groups[prefix + target + '.weight'] = names
            removed.update(names)
    objects = {name: [name] for name in header if name not in removed}
    objects.update(groups)
    specs, bindings = [], {}
    for name, parts in objects.items():
        shape = list(header[parts[0]]['shape'])
        if any(header[part]['shape'][1:] != shape[1:] for part in parts):
            raise ValueError(f'Incompatible concatenation: {name}')
        shape[0] = sum(header[part]['shape'][0] for part in parts)
        specs.append(TensorSpec(name, tuple(shape), 'bf16', 'contiguous_le_v1'))
        bindings[name] = {'object': name}
    head = config.get('head_dim', config['hidden_size'] // config['num_attention_heads'])
    theta = config.get('rope_theta', config.get('rope_parameters', {}).get('rope_theta'))
    rope = (1.0 / (theta ** (torch.arange(0, head, 2).float() / head))).numpy().tobytes()
    specs.append(TensorSpec('text.rope_inv_freq', (head // 2,), 'fp32', 'contiguous_le_v1'))
    bindings['text.rope_inv_freq'] = {'object': 'text.rope_inv_freq'}
    resources = {name: (args.model / name).read_bytes() for name in
                 ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                  'config_sentence_transformers.json', 'modules.json', '1_Pooling/config.json']
                 if (args.model / name).is_file()}
    specs += [ResourceSpec('resource/' + name, len(data)) for name, data in resources.items()]
    components = {'text': {'config': config, 'resources': {name: 'resource/' + name for name in resources}}}
    report = {'source': 'Qwen/Qwen3-Embedding-0.6B', 'dtype': 'bf16', 'weight_quantization': False,
              'source_tensors': len(header), 'artifact_tensors': len(objects) + 1,
              'lossless_row_concatenations': len(groups)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    with source.open('rb') as stream, ArtifactWriter(args.out, specs, components=components, bindings=bindings,
            metadata={'name': 'qwen3-embedding-0.6b-bf16'}, provenance=report) as writer:
        for name, parts in objects.items():
            offset = 0
            for part in parts:
                start, end = header[part]['data_offsets']
                stream.seek(8 + header_size + start)
                remaining = end - start
                digest = hashlib.sha256()
                while remaining:
                    data = stream.read(min(8 * 1024 * 1024, remaining))
                    if not data:
                        raise EOFError(part)
                    digest.update(data)
                    writer.write_region(name, offset, data)
                    offset += len(data)
                    remaining -= len(data)
                source_hashes[part] = digest.hexdigest()
        writer.write_object('text.rope_inv_freq', rope)
        for name, data in resources.items():
            writer.write_object('resource/' + name, data)
    # Check every original parameter against its byte range in the completed artifact.
    verified = 0
    with Artifact(args.out) as reader:
        for name, parts in objects.items():
            blob = reader.read_object(name)
            offset = 0
            for part in parts:
                start, end = header[part]['data_offsets']
                length = end - start
                if hashlib.sha256(blob[offset:offset + length]).hexdigest() != source_hashes[part]:
                    raise ValueError(f'Converted parameter changed: {part}')
                offset += length
                verified += 1
    report['verified_unchanged_parameters'] = verified
    report['artifact_bytes'] = args.out.stat().st_size
    with args.out.open('rb') as stream:
        report['artifact_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    args.out.with_suffix('.conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
