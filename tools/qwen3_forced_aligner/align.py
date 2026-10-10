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
        from transformers import Qwen3ASRProcessor
        with Artifact(artifact) as source:
            component = source.directory.components['text']
            self.config = component['config']
            for name, object_id in component['resources'].items():
                (directory/name).write_bytes(source.read_object(object_id))
        self.processor = Qwen3ASRProcessor.from_pretrained(directory, local_files_only=True, fix_mistral_regex=True)
        self.korean_tokenizer = None

    def split_words(self, text, language):
        from transformers.models.qwen3_asr.processing_qwen3_asr import (
            FORCED_ALIGNER_LANGUAGES, LANGUAGE_CODE_TO_NAME, _clean_tokens, prepare_language_inputs,
        )
        language = prepare_language_inputs(language, 1, LANGUAGE_CODE_TO_NAME, return_code=False)[0]
        if language not in FORCED_ALIGNER_LANGUAGES:
            raise ValueError(f'Unsupported alignment language: {language}')
        if language == 'Korean':
            # The original checkpoint uses this dictionary; the Transformers
            # default LTokenizer has no scores and changes the word boundaries.
            from soynlp.tokenizer import LTokenizer
            if self.korean_tokenizer is None:
                path = Path(__file__).with_name('assets')/'korean_dict_jieba.dict'
                scores = {line.split()[0]: 1.0 for line in path.read_text(encoding='utf-8').splitlines()
                          if line.strip()}
                self.korean_tokenizer = LTokenizer(scores=scores)
            return _clean_tokens(self.korean_tokenizer.tokenize(text))
        return self.processor.split_words_for_alignment(text, language)

    def prepare(self, audio, text, language, directory, index=0):
        words = self.split_words(text, language)
        if not words:
            raise ValueError('No alignable words in the transcript')
        # The embedded legacy chat template is an ASR template and omits the
        # transcript. Alignment requires the checkpoint's explicit timestamp prompt.
        prompt = '<|audio_start|><|audio_pad|><|audio_end|>' + ''.join(
            word+'<timestamp><timestamp>' for word in words)
        inputs = self.processor(text=[prompt], audio=[np.asarray(audio, dtype=np.float32)],
                                return_tensors='pt', padding=True)
        valid = int(inputs['input_features_mask'][0].sum())
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
        from transformers.models.qwen3_asr.processing_qwen3_asr import _fix_timestamps
        expected = self.config['timestamp_segment_time']
        if not words or segment_ms != expected or len(classes) != 2*len(words):
            raise ValueError('Native timestamp output does not match the transcript')
        times = _fix_timestamps(np.asarray(classes, dtype=np.int64)*segment_ms)
        return [{'text': word, 'start': round(times[2*i]/1000, 3),
                 'end': round(times[2*i+1]/1000, 3)} for i, word in enumerate(words)]


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
