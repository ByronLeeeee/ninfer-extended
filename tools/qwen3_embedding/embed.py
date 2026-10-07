"""CPU text tokenization followed by native NInfer embedding inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.reader import Artifact
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path, help='JSON {texts:[...]} or a JSON text list')
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--instruction', default='', help='Optional retrieval instruction for query texts')
    parser.add_argument('--context', type=int, default=32768, help='Maximum total tokens in a native batch')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--dimensions', type=int, default=1024)
    parser.add_argument('--backend', choices=['native','cublas'], default='native')
    parser.add_argument('--tensorcore-attention', action='store_true')
    parser.add_argument('--attention', choices=['auto','fp32','tensorcore'], default='auto')
    parser.add_argument('--no-graph', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.context <= 32768 or not 1 <= args.batch <= 8:
        raise ValueError('Context must be 1..32768 and batch must be 1..8')
    specification = json.loads(args.input.read_text(encoding='utf-8'))
    texts = specification['texts'] if isinstance(specification, dict) else specification
    if not isinstance(texts,list) or not texts or any(not isinstance(text,str) or not text for text in texts):
        raise ValueError('Input must contain nonempty text strings')
    with tempfile.TemporaryDirectory(prefix='ninfer-embedding-') as temp:
        root = Path(temp)
        with Artifact(args.artifact) as artifact:
            component = artifact.directory.components['text']
            for name in ['tokenizer.json','tokenizer_config.json']:
                (root/name).write_bytes(artifact.read_object(component['resources'][name]))
            (root/'config.json').write_text(json.dumps(component['config']))
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
        rows = [tokenizer.encode(f'Instruct: {args.instruction}\nQuery:{text}' if args.instruction else text,
                                 add_special_tokens=True) for text in texts]
        if any(len(row)>args.context for row in rows):
            raise ValueError('A text exceeds --context; split it into passages before embedding')
        cases, pending, tokens = [], [], 0
        for row in rows:
            if pending and (len(pending)==args.batch or tokens+len(row)>args.context):
                cases.append({'id':str(len(cases)), 'input_ids':pending});pending,tokens=[],0
            pending.append(row);tokens+=len(row)
        if pending:
            cases.append({'id':str(len(cases)), 'input_ids':pending})
        token_path,result_path=root/'tokens.json',root/'result.json'
        token_path.write_text(json.dumps({'cases':cases}))
        command=[str(args.binary.resolve()),'--artifact',str(args.artifact.resolve()),'--input',str(token_path),
                 '--out',str(result_path),'--batch',str(args.batch),'--context',str(args.context),
                 '--dimensions',str(args.dimensions),'--backend',args.backend,'--attention',args.attention,'--warmups','1','--repeats','1']
        if args.tensorcore_attention:
            command.append('--tensorcore-attention')
        if args.no_graph:
            command.append('--no-graph')
        subprocess.run(command,check=True)
        result=json.loads(result_path.read_text())
        vectors=[vector for case in result['cases'] for vector in case['variants'][0]['vectors']]
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps({'dimensions':args.dimensions,'normalized':True,'vectors':vectors,
                                        'input_tokens':[len(row) for row in rows]},ensure_ascii=False,indent=2)+'\n')
        print(f'Embedded {len(vectors)} texts into {args.dimensions} dimensions: {args.out}',flush=True)


if __name__=='__main__':
    main()
