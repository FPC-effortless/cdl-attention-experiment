"""Frozen paired natural-text and intervention panels. Scoring receives the gold."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from runtime import Runtime
from data import corpus, eval_offsets


DISTANCES = (256, 1024, 4096, 16384)
# Each template appears only at evaluation. Two assignments, two orders of query target.
CASES = [
    ('station Alder', 'station Birch', 'north', 'south',
     'The bearing for {} is', 'Which way does {} point? It points'),
    ('depot Cedar', 'depot Maple', 'east', 'west',
     'Freight from {} heads', 'In which direction does cargo leave {}? Toward the'),
    ('hall Elm', 'hall Pine', 'large', 'small',
     'The room size at {} is', 'How much space is there in {}? The room is'),
    ('office Oak', 'office Ash', 'black', 'white',
     'The flag outside {} is', 'Which color would a visitor see at {}? It is'),
]


def natural(system, data, split):
    tokens, rows = eval_offsets(data, split)
    result = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start+4]
        offsets = np.array([o for _, o in batch])
        seq = torch.from_numpy(tokens[offsets[:, None]+np.arange(129)].astype('int64'))
        pred, _ = system.advance(seq[:, :-1])
        nll = -pred.gather(-1, seq[:, 1:, None]).squeeze(-1).sum(1)
        for (doc, offset), loss in zip(batch, nll.tolist()):
            result.append(dict(doc=doc, offset=offset, count=128, nll=loss))
    return result


def long_articles(system, data):
    tokens, offs = corpus(data, 'test')
    rows = []
    for doc in np.flatnonzero(np.diff(offs) >= DISTANCES[-1]+1):
        end = int(offs[doc+1])-1
        for distance in DISTANCES:
            seq = torch.from_numpy(tokens[end-distance:end+1].astype('int64'))[None]
            _, state = system.advance(seq[:, :-129], last_only=True)
            pred, _ = system.advance(seq[:, -129:-1], state)
            loss = -pred.gather(-1, seq[:, -128:, None]).sum().item()
            rows.append(dict(doc=int(doc), end=end, distance=distance, count=128, nll=loss))
    return rows


def changed_facts(system, data):
    tok = Tokenizer.from_file(str(data/'tokenizer.json'))
    tokens, bounds = corpus(data, 'test')
    long_docs = np.flatnonzero(np.diff(bounds) > DISTANCES[-1]+128)
    rows = []
    for i, (first, second, a, b, literal, reword) in enumerate(CASES):
        doc = int(long_docs[i])
        pos = bounds[doc]+64
        filler = tokens[pos:pos+DISTANCES[-1]]
        for flip in (0, 1):
            old, new = (a,b) if flip == 0 else (b,a)
            prefix = ' '.join((literal.format(first)+' '+old+'.', literal.format(second)+' '+new+'.',
                               'Correction.', literal.format(first)+' '+new+'.', literal.format(second)+' '+old+'.'))
            _, state = system.advance(torch.tensor([tok.encode(prefix).ids]), last_only=True)
            seen = 0
            for distance in (0, *DISTANCES):
                if distance:
                    seq = torch.from_numpy(filler[seen:distance].astype('int64'))[None]
                    _, state = system.advance(seq, state, last_only=True)
                seen = distance
                for entity, gold, stale in ((first,new,old),(second,old,new)):
                    for kind, phrasing in (('literal',literal),('paraphrase',reword)):
                        query = torch.tensor([tok.encode(' '+phrasing.format(entity)).ids])
                        scores, queried = system.advance(query, copy.deepcopy(state), last_only=True)
                        target, = tok.encode(' '+gold).ids
                        foil, = tok.encode(' '+stale).ids
                        logp = scores[0,-1]
                        table = queried[2][0]
                        rows.append(dict(template=i, entity=entity, flip=flip, doc=doc, delay=distance,
                            kind=kind, target=target, stale=foil, top=int(logp.argmax()),
                            logodds=float(logp[target]-logp[foil]),
                            lookup=table.table.get(table.tail,-1), context=len(tok.encode(prefix).ids)+distance+query.size(1)))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('base','adapter','data','out'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--lambada',type=Path)
    p.add_argument('--disabled',action='store_true')
    p.add_argument('--threads',type=int,default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    s = Runtime.load(a.base,a.adapter,a.disabled)
    name = a.adapter.stem + ('_off' if a.disabled else '')
    for panel, fn in [('test',lambda:natural(s,a.data,'test')), ('transfer',lambda:natural(s,a.data,'transfer')),
                      ('revisions',lambda:changed_facts(s,a.data)), ('context',lambda:long_articles(s,a.data))]:
        result = fn()
        (a.out/(name+'.'+panel+'.json')).write_text(json.dumps(result)+'\n')
        print(name,panel,len(result),flush=True)
    if a.lambada:
        from fresh import lambada
        rows = lambada(s,a.data,a.lambada)
        (a.out/(name+'.lambada.json')).write_text(json.dumps(rows)+'\n')
