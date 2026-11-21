"""Record the held panels with the pinned training model and puzzle sources."""
import hashlib,json,os,shutil,sys
from pathlib import Path
from research_lab.core import Controller
from research_lab.evidence import put_node,link
from trainlab.runner import run_experiment,verify_run
ROOT=Path(__file__).resolve().parent
def main():
    temp=Path(os.environ['SKILL_LAB'])
    ctl=Controller(temp/'os');ctl.program('selector','State-conditioned exact-search skills',1800,0)
    files=['tasks.py','search.py','skills.py','panel.py','PROTOCOL.md','results/model.json']
    upstream=ROOT.parent/'cog-081-086'
    sources=['puzzles.py','evaluate_submissions.py','bridge/cog066_source.py','bridge/experiment_086.py']
    pins={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    pins.update({name:hashlib.sha256((upstream/name).read_bytes()).hexdigest() for name in sources})
    put_node(ctl,'selector.protocol','experiment',pins)
    receipts=[]
    for part,seed in [('held',8923),('rep',9001)]:
        dest=temp/part
        for f in files:
            (dest/f).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,dest/f)
        for f in sources:
            p=dest/'upstream'/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(upstream/f,p)
        os.environ['PUZZLE_BASE']=str(dest/'upstream')
        cfg=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',name='selector-'+part,
                 description='Seven matched exact puzzle arms',command=[sys.executable,'panel.py',part],seed=seed,
                 timeout_seconds=600,inputs=files+['upstream/'+f for f in sources],artifacts=[part+'.json'],
                 metrics={'solved':'max'},source_kind='CONSTRUCTED',split_identity=part,budget={'cpu':1})
        p=dest/'recipe.json';p.write_text(json.dumps(cfg))
        run=run_experiment(p,dest/'runs');d=dest/'runs'/run['id']
        assert run['status']=='succeeded' and verify_run(d)['valid'],d
        record={k:run[k] for k in ('id','inputs','artifacts','metrics','elapsed_seconds')}
        receipts.append(record);put_node(ctl,'selector.'+part,'evidence',record);link(ctl,'selector.'+part,'tests','selector.protocol')
        shutil.copyfile(d/'output'/(part+'.json'),ROOT/'results'/(part+'.json'))
    (ROOT/'results/lab.json').write_text(json.dumps(receipts,indent=2)+'\n')
if __name__=='__main__':main()
