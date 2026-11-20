"""Execute two TrainLab panels, validate receipts and bind them to the Research Lab OS graph."""
import hashlib,json,os,shutil,sys
from pathlib import Path
from research_lab.core import Controller
from research_lab.evidence import put_node,link
from trainlab.runner import run_experiment,verify_run

ROOT=Path(__file__).resolve().parent
LAB=Path(os.environ['COG_ADAPTIVE_LAB'])
def main():
    lab=Controller(LAB/'os')
    lab.program('graded','Adaptive commitment and verified skill acquisition',1800,0)
    files=['PROTOCOL.md','experiments.py','language.py','tracker.py','plans.py','causal.py','boundary.py','check_boundary.py']
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'vendor').rglob('*.py'))]
    pins={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    put_node(lab,'graded.protocol','experiment',pins)
    receipts=[]
    for panel in ('primary','rep'):
        work=LAB/panel;work.mkdir(parents=True,exist_ok=True)
        for f in files:
            (work/f).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/f,work/f)
        cfg=dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',name='graded-'+panel,
                 description='Integrated local simulated issue workflows',command=[sys.executable,'experiments.py',panel],
                 seed=7103 if panel=='primary' else 8011,timeout_seconds=600,inputs=files,
                 artifacts=['sessions.json','subsystems.json','stable.sha256'],metrics={'unauthorized':'min'},
                 source_kind='CONSTRUCTED',split_identity=panel,budget={'cpu':1})
        path=work/'config.json';path.write_text(json.dumps(cfg))
        run=run_experiment(path,work/'runs');d=work/'runs'/run['id']
        assert run['status']=='succeeded' and verify_run(d)['valid'],d
        record={k:run[k] for k in ('id','inputs','artifacts','metrics','elapsed_seconds')}
        receipts.append(record);put_node(lab,'graded.'+panel,'evidence',record)
        link(lab,'graded.'+panel,'tests','graded.protocol')
        dest=ROOT/'results'/panel;dest.mkdir(exist_ok=True)
        for f in cfg['artifacts']:shutil.copyfile(d/'output'/f,dest/f)
    (ROOT/'results/lab.json').write_text(json.dumps(receipts,indent=2)+'\n')
if __name__=='__main__':main()
