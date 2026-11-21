"""Pinned natural-text preparation, article-isolated sampling and scored windows."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def sha(data):
    return hashlib.sha256(data).hexdigest()


def articles(path):
    docs, current = [], []
    for line in pq.read_table(path)['text'].to_pylist():
        s = line.strip()
        if not s:
            continue
        if re.fullmatch(r'= (?![=]).+ (?<!=)=(?!.)', s) and current:
            docs.append('\n'.join(current))
            current = []
        current.append(s)
    if current:
        docs.append('\n'.join(current))
    return docs


def prepare(raw, dialogue, out, pinned_tokenizer=None):
    out.mkdir(parents=True,exist_ok=True)
    docs = {s:articles(raw/(s+'.parquet')) for s in ('train','validation','test')}
    docs['transfer'] = ['\n'.join(t['speaker'].lower()+': '+t['utterance'] for t in d['turns'])
                        for d in json.loads(dialogue.read_text())]
    if pinned_tokenizer:
        tokenizer=Tokenizer.from_file(str(pinned_tokenizer))
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(docs['train'],trainers.BpeTrainer(vocab_size=2048,
            special_tokens=['<|endoftext|>'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False))
    tokenizer.save(str(out/'tokenizer.json'))
    assert tokenizer.get_vocab_size()==2048
    manifest = {'sources':{p.name:sha(p.read_bytes()) for p in [*sorted(raw.glob('*.parquet')),dialogue]},
                'tokenizer_sha256':sha((out/'tokenizer.json').read_bytes()),'splits':{}}
    seen = set()
    for split, texts in docs.items():
        arrays=[]; hashes=[]; dropped=0
        for txt,enc in zip(texts,tokenizer.encode_batch(texts)):
            key=sha(txt.encode())
            if key in seen:
                dropped+=1
                continue
            seen.add(key); hashes.append(key)
            arrays.append(np.array(enc.ids+[0],dtype=np.uint16))
        offsets=np.array([0,*np.cumsum([len(x) for x in arrays])],dtype=np.int64)
        tokens=np.concatenate(arrays)
        np.save(out/(split+'.npy'),tokens)
        np.save(out/(split+'.offsets.npy'),offsets)
        manifest['splits'][split]=dict(documents=len(arrays),tokens=len(tokens),duplicates_dropped=dropped,
                                      tokens_sha256=sha(tokens.tobytes()),document_sha256=hashes)
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def corpus(root,split):
    return np.load(root/(split+'.npy'),mmap_mode='r'),np.load(root/(split+'.offsets.npy'))


def train_offsets(root,seed,steps):
    tokens,offs=corpus(root,'train')
    lengths=np.maximum(np.diff(offs)-128,0)
    rng=np.random.default_rng(seed+800)
    ranks=rng.integers(lengths.sum(),size=(steps,4))
    cumulative=np.cumsum(lengths)
    docs=np.searchsorted(cumulative,ranks,side='right')
    return tokens,offs[docs]+ranks-(cumulative-lengths)[docs]


def eval_offsets(root,split):
    tokens,offs=corpus(root,split)
    rng=np.random.default_rng({'validation':321,'test':917,'transfer':818}[split])
    rows=[]
    for i,(lo,hi) in enumerate(zip(offs,offs[1:])):
        slots=(hi-lo-1)//128
        if slots:
            rows.extend((i,int(lo+j*128)) for j in rng.choice(slots,size=min(slots,2 if split!='transfer' else 1),replace=False))
    if split=='transfer':
        rows=[rows[i] for i in sorted(rng.choice(len(rows),size=min(96,len(rows)),replace=False))]
    return tokens,rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('raw','dialogue','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,help='use the published training-only vocabulary for exact replay')
    args=p.parse_args()
    prepare(args.raw,args.dialogue,args.out,args.tokenizer)
