"""Fresh-machine replay of the held tracker panel and authentic public data."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_bytes())
def main():
    with TemporaryDirectory() as tmp:
        dest=Path(tmp);env=dict(os.environ,TRAINLAB_OUTPUT_DIR=tmp,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        subprocess.run([sys.executable,'experiments.py','rep'],cwd=HERE,env=env,check=True)
        expected=(HERE/'results/rep/stable.sha256').read_text()
        actual=(dest/'stable.sha256').read_text()
        if actual!=expected:
            old=read(HERE/'results/rep/sessions.json');new=read(dest/'sessions.json')
            for a,b in zip(old,new):
                changed=sum([v for k,v in r.items() if k!='ns']!=[v for k,v in s.items() if k!='ns'] for r,s in zip(a['rows'],b['rows']))
                if changed:print(a['seed'],a['arm'],changed)
            raise AssertionError((expected,actual))
        sys.path.insert(0,str(HERE))
        import authentic as ac
        held=ac.banking();prior=read(HERE/'results/authentic-banking.json');assert held['source']==prior['source']
        for arm in held['arms']:
            for metric in (1,5):
                def count(study):
                    return sum(y in picks[:metric] for _,y,picks,_,_ in study['arms'][arm]['rows'])
                assert count(held)==count(prior),(arm,metric,count(held),count(prior))
        ac.original.SOURCE=dest
        for name,h in read(ac.BASE/'results/dialogue-transfer.json')['source'].items():
            if not name.startswith('test/'):continue
            blob=urlopen('https://raw.githubusercontent.com/budzianowski/multiwoz/fe0c8e65cfcd8462bd33c86e35f21addc84ca82b/data/MultiWOZ_2.2/'+name,timeout=60).read()
            assert hashlib.sha256(blob).hexdigest()==h
            p=dest/name;p.parent.mkdir(exist_ok=True);p.write_bytes(blob)
        assert ac.dialogue()==read(HERE/'results/authentic-dialogue.json')
        for name,script,arguments in [('transfer','transfer.py',['score']),('admission','admission.py',[]),
                                      ('corrections','corrections.py',[])]:
            path=HERE/'results'/(name+'.json')
            old=read(path)
            subprocess.run([sys.executable,script,*arguments],cwd=HERE,env=env,check=True)
            new=read(path)
            if name=='corrections':
                assert old==new
                continue
            if name=='admission':
                assert old['dictionary']==new['dictionary'] and old['causal']==new['causal']
                old,new=old['sessions'],new['sessions']
            for left,right in zip(old,new,strict=True):
                assert (left['arm'],left['seed'])==(right['arm'],right['seed'])
                for a,b in zip(left['rows'],right['rows'],strict=True):
                    for k in ('gold','top5','provisional','proposal','status','semantic_error','bad_permission'):
                        assert a[k]==b[k],(name,left['arm'],left['seed'],a['i'],k)
    print('Replicated transfer, admission, and correction decisions (teacher logits supplied).')
    print('Replicated the reserved-seed panel, banking top-1/top-5 counts, and all dialogue read/write traces.')
if __name__=='__main__':main()
