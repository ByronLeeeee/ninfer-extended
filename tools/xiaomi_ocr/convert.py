"""Convert the original Xiaomi OCR checkpoint to this fork's BF16 v3 artifact."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
args = p.parse_args()
model = args.model.resolve()
out = args.out.resolve()
repo = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='xiaomi-ocr-resources-') as directory:
    resources = Path(directory)
    processor = json.loads((model / 'processor_config.json').read_text())
    config = json.loads((model / 'config.json').read_text())
    for name, key in [('preprocessor_config.json', 'image_processor'),
                      ('video_preprocessor_config.json', 'video_processor')]:
        (resources / name).write_text(json.dumps(processor[key]), encoding='utf-8')
    (resources / 'generation_config.json').write_text(json.dumps({
        'eos_token_id': config['eos_token_id'], 'pad_token_id': config['pad_token_id'],
        'do_sample': False}), encoding='utf-8')
    AutoTokenizer.from_pretrained(model, local_files_only=True).save_pretrained(resources)
    tc = json.loads((resources / 'tokenizer_config.json').read_text())
    tc.setdefault('added_tokens_decoder', {})
    tc.setdefault('add_bos_token', False)
    (resources / 'tokenizer_config.json').write_text(json.dumps(tc, ensure_ascii=False), encoding='utf-8')
    command = [sys.executable, '-m', 'tools.convert', '--model', str(model),
               '--recipe', str(Path(__file__).with_name('recipe_bf16.py')),
               '--components', 'text,vision', '--device', 'cpu', '--name', 'xiaomi-ocr-0',
               '--out', str(out)]
    for name in ['tokenizer.json', 'tokenizer_config.json', 'generation_config.json',
                 'preprocessor_config.json', 'video_preprocessor_config.json']:
        command.extend(['--resource', name + '=' + str(resources / name)])
    subprocess.run(command, cwd=repo, check=True)
