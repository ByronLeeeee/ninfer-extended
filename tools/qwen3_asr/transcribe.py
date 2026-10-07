"""CPU audio/token frontend for the native NInfer ASR Engine; no Python GPU inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np
from transformers import AutoProcessor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.artifact.reader import Artifact


def waveform(path):
    with wave.open(str(path), 'rb') as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
            raise ValueError('Audio must be mono PCM16 WAV at 16000 Hz; resample with ffmpeg first.')
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2').astype(np.float32) / 32768


def processor_from_artifact(path, directory):
    with Artifact(path) as artifact:
        component = artifact.directory.components['text']
        (directory / 'config.json').write_text(json.dumps(component['config']))
        for name, object_id in component['resources'].items():
            (directory / name).write_bytes(artifact.read_object(object_id))
    return AutoProcessor.from_pretrained(directory, local_files_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--audio', required=True, nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--backend', choices=['native', 'cublas'], default='native')
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--no-graph', action='store_true')
    parser.add_argument('--tensorcore-prefill', action='store_true')
    args = parser.parse_args()
    if not 1 <= len(args.audio) <= 4:
        parser.error('Supply one to four WAV files.')
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='ninfer-asr-') as temporary:
        directory = Path(temporary)
        processor = processor_from_artifact(args.artifact, directory)
        specs = []
        for index, path in enumerate(args.audio):
            audio = waveform(path)
            inputs = processor.apply_transcription_request(audio=audio, return_tensors='pt')
            features = inputs['input_features'].float().contiguous()
            feature_path = directory / f'{index}.features.f32'
            mask_path = directory / f'{index}.mask.i32'
            features.numpy().tofile(feature_path)
            inputs['input_features_mask'].numpy().astype('<i4').tofile(mask_path)
            specs.append({'id': path.stem, 'bins': 128, 'frames': features.shape[-1],
                          'features_path': str(feature_path), 'mask_path': str(mask_path),
                          'audio_seconds': len(audio) / 16000, 'prompt_ids': inputs['input_ids'][0].tolist(),
                          'audio_token_id': 151676})
        frontend_seconds = time.perf_counter() - started
        request = directory / 'features.json'
        request.write_text(json.dumps({'samples': specs}))
        command = [str(args.engine.resolve()), '--artifact', str(args.artifact.resolve()), '--input', str(request),
                   '--out', str(args.out.resolve()), '--backend', args.backend, '--context', str(args.context),
                   '--max-new-tokens', str(args.max_new_tokens), '--repeats', str(args.repeats)]
        if args.no_graph:
            command.append('--no-graph')
        if args.tensorcore_prefill:
            command.append('--tensorcore-prefill')
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        result = json.loads(args.out.read_text())
        result['frontend_seconds'] = frontend_seconds
        result['audio_paths'] = [str(path.resolve()) for path in args.audio]
        for row in result['runs']:
            row['text'] = processor.decode(row['token_ids'], return_format='transcription_only')
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'text': result['runs'][-1]['text'], 'wall_seconds': result['runs'][-1]['wall_seconds'],
                          'frontend_seconds': frontend_seconds}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
