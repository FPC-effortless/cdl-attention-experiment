"""Recover exact UNIFIED-014 source and checkpoints; replay original closure counts."""
import argparse,csv,hashlib,json,os,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
LOCK=json.loads((HERE/'SOURCE_LOCK.json').read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def blob(p):
    b=Path(p).read_bytes();return hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
def restore(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);archives=LOCK['unified014']['archive_blobs']
    for name,expected in archives.items():
        p=HERE/'original014'/name;assert blob(p)==expected,(name,'archive identity')
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            for m in z.infolist():
                q=(out/m.filename).resolve();assert q.is_relative_to(out.resolve()),m.filename
                if m.is_dir():q.mkdir(parents=True,exist_ok=True);continue
                b=z.read(m);q.parent.mkdir(parents=True,exist_ok=True)
                if q.exists():assert q.read_bytes()==b,('conflicting member',m.filename)
                else:q.write_bytes(b)
    rows=list(csv.DictReader((out/'checkpoint_manifest.csv').open()))
    assert len(rows)==42 and len({x['checkpoint_path'] for x in rows})==42
    for r in rows:
        p=out/r['checkpoint_path'];assert p.is_file() and p.stat().st_size==int(r['checkpoint_bytes']) and sha(p)==r['sha256'],r['run']
    assert len(list((out/'fits').glob('*.pt')))==42
    result=dict(status='EXACT_ARCHIVE_CUSTODY',source_commit=LOCK['unified014']['commit'],archives=archives,checkpoint_count=len(rows),verified_checkpoints=rows)
    (out/'recovery_receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result
def replay(out,receipt):
    out=Path(out);sys.path.insert(0,str(out));import torch,closure014,curriculum,presence
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    prior=json.loads((out/'closure.json').read_text());assert len(prior)==9;rows=[]
    for r in prior:
        arm,seed=r['arm'],r['seed'];ck=torch.load(out/'fits'/f'{arm}_{seed}.pt',weights_only=True,map_location='cpu')
        m=presence.Model(arm) if arm.startswith('presence') else curriculum.Model(arm);m.load_state_dict(ck['weights'],strict=True)
        z=closure014.check(m,arm,seed)
        for k in ('n','action_correct','address_correct','write_correct','write_n','exact_policy','two_entity_stress'):assert z[k]==r[k],(arm,seed,k,z[k],r[k])
        rows.append(dict(arm=arm,seed=seed,original=z,reproduced_discrete_counts=True));print(arm,seed,'closure replay passed',flush=True)
    receipt['closure_replays']=rows;receipt['closure_replay_count']=len(rows);receipt['new_fits']=0
    (out/'recovery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
    r=restore(a.out)
    if a.replay:replay(a.out,r)
    print('ARCHIVES_VERIFIED',r['checkpoint_count'],'CLOSURE_REPLAYS',r.get('closure_replay_count',0),flush=True)
