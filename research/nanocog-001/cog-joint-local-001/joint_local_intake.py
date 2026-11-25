"""TRAIN-only wire verification and literal encoding for the native joint fitter."""
from collections import Counter
import json
from hashlib import file_digest,sha256
from pathlib import Path
from joint_local_objective import need
from joint_local_schedule import create,encoded

TRAIN_FILES=('edit_bundles.json','ordinary_rows.json')
TRAIN_TYPES=('joint-local.train-edit.v1','joint-local.train-ordinary.v1')

def digest(path):
    with path.open('rb') as f:return file_digest(f,'sha256').hexdigest()

def read(path,pin):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<2_000_000,'bounded regular JSON input')
    raw=path.read_bytes();need(sha256(raw).hexdigest()==pin,'wrong exact bytes: '+path.name)
    return json.loads(raw)

def parse_prompt(p):
    need(type(p)is str and ' '.join(p.split())==p,'raw seven-word format')
    s=tuple(p.split())
    need(len(s)==7 and s[1]==s[4]=='is' and s[0]!=s[3] and s[2]!=s[5] and s[6] in (s[0],s[3]),
         'typed causal raw grammar')
    return s

def inspect_records(edits,ordinary,locked_words):
    need(type(locked_words)is tuple and len(locked_words)==len(set(locked_words))==25 and 'is' in locked_words,
         'registered whole vocab')
    parts=[];seen=set();ids={w:262+i for i,w in enumerate(locked_words)}
    for d,n,sch in zip((edits,ordinary),(128,640),TRAIN_TYPES,strict=True):
        need(set(d)=={'role','schema','rows'} and d['role']=='TRAIN' and d['schema']==sch and
             type(d['rows'])is list and len(d['rows'])==n,'fixed TRAIN wire shape')
        out=[]
        for i,r in enumerate(d['rows']):
            need(type(r['slot'])is int and r['slot']==i,'TRAIN slot ordinal')
            if n==128:
                need(set(r)=={'slot','prompts','answers','margin_foils'} and
                     len(r['prompts'])==len(r['answers'])==6 and len(r['margin_foils'])==4,'six-row bundle')
                px=[parse_prompt(v) for v in r['prompts']];ys=r['answers'];fo=r['margin_foils']
                h=px[0][:-1]
                for j in range(0,6,2):
                    a,b=px[j:j+2]
                    need(a[:-1]==b[:-1] and a[6]==h[0] and b[6]==h[3] and
                         set((a[0],a[3]))==set((h[0],h[3])) and [a[2],b[5]]==ys[j:j+2],
                         'identical history and opposite literal queries')
                changed=lambda q:[j for j,(a,b) in enumerate(zip(h,q[:-1],strict=True)) if a!=b]
                need(changed(px[2])==[2] and changed(px[4])==[5] and ys[2]==ys[5] and ys[2] not in ys[:2]
                   and ys[3]==ys[1] and ys[4]==ys[0] and fo==[ys[0],ys[2],ys[5],ys[1]],
                   'exact directed clause edits and margin foils')
            else:
                need(set(r)=={'slot','prompt','answer'},'ordinary row')
                p=parse_prompt(r['prompt']);px=[p];ys=[r['answer']];fo=[]
                need(ys[0]==p[2 if p[0]==p[6] else 5],'ordinary grammar labels')
            need(len(set(px))==len(px) and seen.isdisjoint(px),'TRAIN duplicate observations')
            need(all(w in ids for sent in px for w in sent) and all(w in ids for w in ys+fo),'exact literal vocab membership')
            seen.update(px)
            out.append((tuple(tuple(ids[w] for w in sent) for sent in px),tuple(ids[w] for w in ys),
                        tuple(ids[w] for w in fo)))
        parts.append(out)
    need(len(seen)==1408 and {w for t in seen for w in t}==set(locked_words),'TRAIN coverage full fixed vocabulary')
    # Role adjacency/family for ordinary scoring must be fixed by exact generation.
    for start in range(0,640,8):
        xs=[x[0][0] for x in parts[1][start:start+8]]
        name_sets={tuple(sorted((p[0],p[3]))) for p in xs}
        color_sets={tuple(sorted((p[2],p[5]))) for p in xs}
        need(len(name_sets)==len(color_sets)==1 and len({(p[0],p[2],p[6]) for p in xs})==8,
             'ordinary exact complete factorial')
    return tuple(parts)

def batches(root,pins,seed,words):
    e,o=(read(root/f,pins[f]) for f in TRAIN_FILES)
    edit,ordinary=inspect_records(e,o,words)
    slots=read(root/'slots.json',pins['slots.json'])
    need(slots==create(seed) and (root/'slots.json').read_bytes()==encoded(slots),'prescheduled abstract slot trajectory')
    return edit,ordinary,slots['steps']

def gather(edit,ordinary,slot):
    sid=slot['bundle_slot']
    e=[] if sid is None else [edit[sid]]
    s=e+[ordinary[x] for x in slot['ordinary_slots']]
    rows=tuple(q for x,_,_ in s for q in x); targets=tuple(q for _,y,_ in s for q in y)
    need(len(rows)==len(targets)==8,'exact batch materialization')
    return rows,targets,e[0][2] if e else None
