"""Use the exact installed NInfer BF16 draft shortlist in the PyTorch reference."""
from pathlib import Path
import hashlib
import json
import struct
import torch
from transformers.masking_utils import create_causal_mask
from qwen_mtp_torch import MtpHead


class IndexedMtpHead(MtpHead):
    def decode(self, previous_hidden, next_embeddings, positions):
        x=self.stem(previous_hidden,next_embeddings)
        mask=create_causal_mask(config=self.config,inputs_embeds=x,attention_mask=None,
                               past_key_values=self.cache,position_ids=None)
        x=self.layers[0](x,position_embeddings=self.rotary_emb(x,positions),
                        attention_mask=mask,past_key_values=self.cache,use_cache=True)
        hidden=self.norm(x)
        logits=torch.nn.functional.linear(hidden[:,-1:],self.proposal_weight)
        token=self.proposal_ids[logits.argmax(-1)]
        return hidden,token


def install(reference, source):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.artifact.reader import Artifact
    with Artifact(source) as artifact:
        binding = artifact.directory.bindings['proposal/token_ids']
        object_id = binding['object'] if isinstance(binding, dict) else binding
        data = artifact.read_object(object_id)
    assert len(data) % 4 == 0
    rows = len(data) // 4
    ids = sorted(struct.unpack(f'<{rows}i', data))
    assert len(set(ids)) == rows
    assert min(ids) >= 0 and max(ids) < reference.model.lm_head.weight.shape[0]
    metadata = dict(rows=rows, sha256=hashlib.sha256(data).hexdigest(),
                    source='NInfer artifact proposal/token_ids')
    token_ids=torch.tensor(ids,dtype=torch.long,device='cuda')
    head=reference.mtp
    head.__class__=IndexedMtpHead
    head.register_buffer('proposal_ids',token_ids)
    head.register_parameter('proposal_weight',torch.nn.Parameter(
        reference.model.lm_head.weight.index_select(0,token_ids).contiguous(),requires_grad=False))
    # Ascending original IDs give the same tie rule as the full output head.
    reference.propose=torch.compile(head.decode,fullgraph=True,dynamic=False,mode='reduce-overhead')
    reference.mtp_prefill=torch.compile(head.prefill,fullgraph=True,dynamic=False)
    reference.proposal_metadata=metadata
