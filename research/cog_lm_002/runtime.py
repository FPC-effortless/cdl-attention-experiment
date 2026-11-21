"""Chunked inference for the adapter, learned records and exact successor table."""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from architecture import Readout, Successors, InferenceCache, backbone, features, successor_ids


class Runtime:
    def __init__(self, model, head, disabled=False, mask_read=False):
        self.model, self.head, self.disabled = model, head, disabled
        self.mask_read = mask_read

    @classmethod
    def load(cls, base, adapter, disabled=False):
        model = backbone(base)
        weights = torch.load(adapter, map_location='cpu', weights_only=True)
        head = Readout(model.c.width, any(key.startswith('bank.') for key in weights))
        head.load_state_dict(weights)
        return cls(model, head.eval(), disabled)

    @torch.inference_mode()
    def advance(self, x, state=None, last_only=False):
        if x.ndim != 2 or x.size(1) == 0:
            raise ValueError('expected a nonempty batch of token sequences')
        layers, bank, table = state or (None, None, [Successors(self.model.c.capacity) for _ in x])
        outputs = []
        for chunk in x.split(256, 1):
            pieces = [chunk]
            if self.model.c.arm == 'mamba2':
                cut = chunk.size(1)//16*16
                pieces = [chunk[:, :cut]] if cut else []
                if cut < chunk.size(1):
                    pieces.extend(chunk[:, cut:].split(1, 1))
                if layers is None and cut == 0:
                    layers = [InferenceCache.alloc(len(x), b.mcfg, device=x.device) for b in self.model.blocks]
            for part in pieces:
                h, layers = features(self.model, part, layers)
                ids = successor_ids(part, self.model.c.capacity, table)
                pred, bank = self.head(h, ids, self.model.emb.weight, bank, self.disabled, last_only, self.mask_read)
                if last_only:
                    outputs = [pred]
                else:
                    outputs.append(pred)
        return torch.cat(outputs, 1), (layers, bank, table)

    @torch.inference_mode()
    def generate(self, tokens, count):
        scores, state = self.advance(tokens, last_only=True)
        generated = []
        for _ in range(count):
            token = scores[:, -1].argmax(-1, keepdim=True)
            generated.append(token.item())
            if token.item() == 0:
                break
            scores, state = self.advance(token, state, last_only=True)
        return generated


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('base', 'adapter', 'tokenizer'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--count', type=int, default=48)
    args = p.parse_args()
    torch.set_num_threads(2)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    system = Runtime.load(args.base, args.adapter)
    prompt = torch.tensor([tokenizer.encode(args.prompt).ids])
    print(tokenizer.decode(system.generate(prompt, args.count)))
