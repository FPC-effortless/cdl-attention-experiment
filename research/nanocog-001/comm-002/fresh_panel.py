"""Build a fresh complete factorial panel without opening any historical test."""
import itertools
import json
from hashlib import sha256
from pathlib import Path
import random
from cog_comm.role_control import CELLS,COLORS,NAMES,families,world
from cog_comm.role_panel import save

def build(out):
    registered=families()
    seen=set(registered['fit'])|set(registered['control'])
    assert len(seen)==440
    keys=sorted(set(itertools.product(itertools.combinations(NAMES,2),
                          itertools.combinations(COLORS,2)))-seen)
    random.Random(20261123).shuffle(keys)
    cases=keys[:120]
    assert len(set(cases))==120 and not seen.intersection(cases)
    fs,actor=[],[]
    train_names={k[0] for k in registered['fit']}
    for i,k in enumerate(cases):
        text=[world(k,cell) for cell in CELLS]
        # Independently derive targets by resolving last identity against clauses.
        answers=[{p[0]:p[2],p[3]:p[5]}[p[-1]] for p in text]
        actor.extend(' '.join(p) for p in text)
        fs.append(dict(key=k,rows=list(range(i*8,i*8+8)),answers=answers,
                       familiar_name_pair=k[0] in train_names))
    assert len(actor)==len(set(actor))==960
    out.mkdir(exist_ok=False)
    a=save(out,'confirm_actor.json',dict(schema='cog-comm-role-001.actor.v1',prompts=actor))
    j=save(out,'confirm_judge.json',dict(panel='confirm',families=fs))
    print(json.dumps(dict(actor_sha256=a,judge_sha256=j,cases=len(cases),worlds=len(actor),
                          unseen_name_pair=sum(not f['familiar_name_pair'] for f in fs)),sort_keys=True))
if __name__=='__main__':build(Path('panel'))
