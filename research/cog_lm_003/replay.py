"""Check exact prepared inputs, checkpoint lineage, data balance and cached-to-native outputs."""
import argparse
from common import Path, hashlib, json, torch, corpus, sha, read, save, build_docs
from fit import CONFIGS
from assess import system


@torch.inference_mode()
def run(a):
    torch.set_num_threads(1)
    receipt=dict(parents={},native={},data={},documents={})
    manifest=read(Path(__file__).resolve().parents[1]/'cog_lm_001/evidence/data_manifest.json')
    for split in ('train','test','transfer'):
        tokens,_=corpus(a.data,split)
        digest=hashlib.sha256(tokens.tobytes()).hexdigest()
        assert digest==manifest['splits'][split]['tokens_sha256']
        receipt['data'][split]=digest
    assert sha(a.data/'tokenizer.json')==manifest['tokenizer_sha256']
    for split in ('train','test'):
        doc=build_docs(a.data,split)
        for x,y in zip(doc[::2],doc[1::2]):
            assert x['filler']==y['filler'] and x['family']==y['family']
            for q,r in zip(x['questions'],y['questions']):
                assert (q['target'],q['foil'],q['q'])==(r['foil'],r['target'],r['q'])
        receipt['documents'][split]=dict(n=len(doc),sha256=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest())
    doc=build_docs(a.data,'test')[0]
    for n in CONFIGS:
        net=system(a,n)
        rows=read(a.out/(n+'.probes.json'))
        rr=[r for r in rows if r['family']==0 and r['flip']==0 and r['delay']==256]
        delta=[]
        for q,r in zip(doc['questions'],rr,strict=True):
            ids=doc['prefix']+doc['filler'][:256]+q['q']
            p,_=net.advance(torch.tensor([ids]),last_only=True)
            score=p[0,-1]
            assert int(score.argmax())==r['full']['top']
            delta.append(abs(float(score[q['target']]-score[q['foil']])-r['full']['odds']))
        assert max(delta)<.0003,(n,delta)
        receipt['native'][n]=max(delta)
        receipt['parents'][n]=sha(a.out/(n+'.pt'))
    save(a.out/'replay.json',receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','parents','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
