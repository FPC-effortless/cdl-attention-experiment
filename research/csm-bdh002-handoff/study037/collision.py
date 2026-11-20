"""Audit whether swapping only the reference words can change either pair-model input."""
import json
import numpy as np
from experiment import ROOT, encode

rows=[]
for seed in (11,23,37):
    w=np.load(ROOT/f'weights-{seed}.npz')['pair']
    for size in (1,2,3):
        keys=[' '.join(['dax','rel','fox'][:size]),' '.join(['yup','sol','nev'][:size])]
        for pos,neg in [('accepted','discarded'),('discarded','accepted')]:
            texts=[f'of {keys[0]} and {keys[1]} the {a} was {pos} and the {z} was {neg}'
                   for a,z in [('former','latter'),('latter','former')]]
            _,a,c,_=encode(texts[0],keys)
            _,b,d,_=encode(texts[1],keys)
            assert np.array_equal(a,b) and np.array_equal(c,d)
            rows.append(dict(seed=seed,texts=texts,alignment_logit=float(a@w[:3]),polarity_logit=float(c@w[3:])))
(ROOT/'collisions.json').write_text(json.dumps(rows,indent=2))
