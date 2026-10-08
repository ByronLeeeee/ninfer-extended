"""Official CPU frontend and timestamp parsing for the native alignment Engine."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.reader import Artifact


class AlignmentProcessor:
    def __init__(self, artifact, directory):
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor
        with Artifact(artifact) as source:
            component = source.directory.components['text']
            self.config = component['config']
            for name, object_id in component['resources'].items():
                (directory/name).write_bytes(source.read_object(object_id))
        self.processor = Qwen3ASRProcessor.from_pretrained(directory, local_files_only=True, fix_mistral_regex=True)
        self.text_processor = Qwen3ForceAlignProcessor()

    def prepare(self, audio, text, language, directory, index=0):
        words, prompt = self.text_processor.encode_timestamp(text, language)
        if not words:
            raise ValueError('No alignable words in the transcript')
        inputs = self.processor(text=[prompt], audio=[np.asarray(audio, dtype=np.float32)],
                                return_tensors='pt', padding=True)
        valid = int(inputs['feature_attention_mask'][0].sum())
        frames = (valid+99)//100*100
        # The official tower trims the feature prefix, then zero-pads its last
        # convolution chunk. Padded log-mel values are not valid input frames.
        features = np.zeros((128, frames), dtype=np.float32)
        features[:, :valid] = inputs['input_features'][0, :, :valid].float().numpy()
        mask = np.zeros(frames, dtype='<i4');mask[:valid] = 1
        feature_path, mask_path = directory/f'{index}.f32', directory/f'{index}.i32'
        features.tofile(feature_path);mask.tofile(mask_path)
        return {'frames': frames, 'features_path': str(feature_path), 'mask_path': str(mask_path),
                'prompt_ids': inputs['input_ids'][0].tolist()}, words

    def parse(self, words, classes, segment_ms):
        import torch
        expected = self.config['timestamp_segment_time']
        if segment_ms != expected or len(classes) != 2*len(words):
            raise ValueError('Native timestamp output does not match the transcript')
        times = torch.tensor(classes, dtype=torch.int64)*segment_ms
        output = self.text_processor.parse_timestamp(words, times)
        return [{'text': row['text'], 'start': round(row['start_time']/1000, 3),
                 'end': round(row['end_time']/1000, 3)} for row in output]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', required=True, type=Path)
    p.add_argument('--engine', required=True, type=Path)
    p.add_argument('--audio', required=True, nargs='+', type=Path)
    p.add_argument('--text', required=True, nargs='+')
    p.add_argument('--language', required=True, nargs='+')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--warmups', type=int, default=0)
    p.add_argument('--backend', choices=['native', 'cublas'], default='native')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--scalar-prefill', action='store_true')
    args = p.parse_args()
    if not (1 <= len(args.audio) <= 8 and len(args.audio) == len(args.text) == len(args.language)):
        p.error('Supply matching lists of one to eight audio, text and language inputs')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='ninfer-align-') as temporary:
        directory = Path(temporary)
        frontend = AlignmentProcessor(args.artifact, directory)
        samples, word_lists = [], []
        for i, (path, text, language) in enumerate(zip(args.audio, args.text, args.language)):
            audio, rate = sf.read(path, dtype='float32')
            if rate != 16000 or audio.ndim != 1:
                raise ValueError('Alignment input must be mono audio at 16 kHz')
            sample, words = frontend.prepare(audio, text, language, directory, i)
            samples.append(sample);word_lists.append(words)
        frontend_seconds = time.perf_counter()-started
        spec = directory/'features.json'
        spec.write_text(json.dumps({'cases': [{'id': 'alignment', 'samples': samples}]}))
        command = [str(args.engine), '--artifact', str(args.artifact), '--input', str(spec),
                   '--out', str(args.out), '--batch', str(len(samples)), '--backend', args.backend,
                   '--warmups', str(args.warmups), '--repeats', str(args.repeats)]
        if args.eager: command.append('--eager')
        if args.scalar_prefill: command.append('--scalar-prefill')
        subprocess.run(command, check=True)
        result = json.loads(args.out.read_text())
        result['frontend_seconds'] = frontend_seconds
        for run in result['cases'][0]['runs']:
            run['words'] = [frontend.parse(words, classes, run['timestamp_segment_ms'])
                            for words, classes in zip(word_lists, run['timestamp_classes'])]
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__': main()
