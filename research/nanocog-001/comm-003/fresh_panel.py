"""Construct ROLE-003's prospective synthetic panel; never opens predictions or outputs."""
from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random
from cog_comm.role_control import CELLS,COLORS,NAMES,families,world
from cog_comm.role_panel import save

OLD='ce80c30bfbb9c38ae8b078d16c0fd8187bc8a641bba4315bd05dad5253927118'
HERE=Path(__file__).resolve().parent

def require(ok,msg):
    if not ok: raise ValueError(msg)

def build(out):
    registered=families()
    prior=set(registered['fit'])|set(registered['control'])
    require(len(prior)==440,'wrong original roles')
    pool=set(itertools.product(itertools.combinations(NAMES,2), itertools.combinations(COLORS,2)))
    checked=(HERE/'../cog-role-002/panel/confirm_judge.json').read_bytes()
    require(sha256(checked).hexdigest()==OLD,'old markout bytes changed')
    old=[tuple(map(tuple,f['key'])) for f in json.loads(checked)['families']]
    check_order=sorted(pool-prior)
    random.Random(20261123).shuffle(check_order)
    require(len(set(old))==120 and old==check_order[:120] and not prior.intersection(old),
            'old markout differs from exact prospectively assigned keys')
    forbidden=prior|set(old)
    order=sorted(pool-forbidden)
    random.Random(20261124).shuffle(order)
    cases=order[:120]
    require(len(set(cases))==120 and not forbidden.intersection(cases),'not fresh')
    fa,actor=[],[]
    train_names={k[0] for k in registered['fit']}
    train_colors={k[1] for k in registered['fit']}
    for i,k in enumerate(cases):
        text=[world(k,cell) for cell in CELLS]
        answers=[{p[0]:p[2],p[3]:p[5]}[p[-1]] for p in text]
        require(Counter(answers)=={color:4 for color in k[1]},'unbalanced factorial answers')
        actor.extend(' '.join(p) for p in text)
        fa.append(dict(key=k,rows=list(range(i*8,i*8+8)),answers=answers,
                       familiar_name_pair=k[0] in train_names))
    require(len(actor)==len(set(actor))==960 and len(forbidden)==560,'panel geometry')
    out.mkdir(exist_ok=False)
    a=save(out,'confirm_actor.json',dict(schema='cog-comm-role-001.actor.v1',prompts=actor))
    j=save(out,'confirm_judge.json',dict(panel='confirm',families=fa))
    meta=dict(seed=20261124,actor_sha256=a,judge_sha256=j,role002_exclusion_sha256=OLD,
              fitting_families_excluded=320,role001_control_families_excluded=120,
              role002_confirm_families_excluded=120,cases=len(cases),worlds=len(actor),
              familiar_name_pair=sum(f['familiar_name_pair'] for f in fa),
              unseen_name_pair=sum(not f['familiar_name_pair'] for f in fa),
              unseen_color_pair=sum(k[1] not in train_colors for k in cases),
              answer_class_worlds=dict(Counter(a for f in fa for a in f['answers'])))
    h=save(out,'panel_receipt.json',meta)
    print(json.dumps(dict(panel_receipt_sha256=h,**meta),sort_keys=True))
if __name__=='__main__': build(HERE/'panel')
