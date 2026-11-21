"""Freeze constructed fixture identities in Forge; measure and score with verified TrainLab receipts."""
import json
import os
from pathlib import Path
from shutil import copyfile
import sys
from forge_data.core import canonical,release
from research_lab.core import Controller
from research_lab.evidence import put_node,link
from trainlab.runner import digest, run_experiment, verify_run, write_json
from amortization import cases, ROOT

def main(work):
    work=work.resolve()
    work.mkdir(parents=True,exist_ok=False)
    base=ROOT.parent/'cog-081-086'
    ctl=Controller(work/'os')
    ctl.program('cog-r1','COG-ATTN persistent Hanoi index diagnosis',1800,0)
    at='2026-11-21T00:00:00Z'
    fixtures=[dict(episode_id=t.id,subject_id=key,record_id=t.id,observed_at=at,
                   available_at=at,observation=dict(fixture=t.digest(),order=i,regime=regime),
                   action=None,outcome=dict(status='unknown',verified=False))
              for key,seed,regime,tasks in cases() for i,t in enumerate(tasks)]
    raw=work/'source.jsonl'
    raw.write_bytes(b''.join(canonical(r) for r in fixtures))
    spec=work/'source.json'
    write_json(spec,dict(dataset_id='cog-hanoi-amortization-r1',source_id='constructed/cog-attn',
        source_version=digest(raw),adapter='operations-jsonl',license_id='OPERATOR_AUTHORIZED_RESEARCH',
        provenance_uri='https://github.com/FPC-effortless/cdl-attention-experiment/tree/c5285d386d6ec98b3d2ff7dc21f12e428e18a8f9/research/cog-087-090',
        ai_use_allowed=True,redistribution_allowed=True,privacy_reviewed=True,split_salt='cog-hanoi-r1'))
    release(raw,spec,work/'dataset')
    sources=['amortization.py','amortization_lab.py','AMORTIZATION_PROTOCOL.md','tasks.py','panel.py','search.py','skills.py']
    upstream=['puzzles.py','evaluate_submissions.py','bridge/cog066_source.py','bridge/experiment_086.py']
    lineage=dict(forge_manifest_sha256=digest(work/'dataset/manifest.json'),
                 fixture_count=len(fixtures), protocol_sha256=digest(ROOT/'AMORTIZATION_PROTOCOL.md'),
                 stage='constructed engineering diagnosis; Forge split labels are archival, not protected datasets or training')
    put_node(ctl,'r1.protocol','experiment',lineage)
    earlier=None
    receipts={}
    for stage,artifact,metric in [('measure','measurements.json','sessions'),('score','amortization_summary.json','verified_tasks')]:
        d=work/stage
        d.mkdir()
        for f in sources: copyfile(ROOT/f,d/f)
        for f in upstream:
            p=d/'upstream'/f
            p.parent.mkdir(parents=True,exist_ok=True)
            copyfile(base/f,p)
        copyfile(work/'dataset/manifest.json',d/'forge_manifest.json')
        inputs=sources+['upstream/'+f for f in upstream]+['forge_manifest.json']
        args=[sys.executable,'amortization.py',stage]
        if earlier:
            copyfile(earlier,d/'measurements.json')
            inputs+=['measurements.json']
            args+=['measurements.json']
        os.environ['PUZZLE_BASE']=str(d/'upstream')
        config=d/'recipe.json'
        write_json(config,dict(schema='trainlab.experiment.v1',execution_purpose='engineering_diagnostic',
            name='cog-r1-'+stage,description='Constructed staged cumulative-cost experiment',
            command=args,seed=9419,timeout_seconds=550,inputs=inputs,artifacts=[artifact],
            metrics={metric:'max'},source_kind='CONSTRUCTED',
            split_identity=lineage['forge_manifest_sha256'],budget={'cpu':1}))
        rec=run_experiment(config,d/'runs')
        rd=d/'runs'/rec['id']
        verification=verify_run(rd)
        if rec['status']!='succeeded' or not verification['valid']:
            raise RuntimeError((stage,rd,rec['status'],verification))
        earlier=rd/'output'/artifact
        copyfile(earlier,work/artifact)
        evidence={k:rec[k] for k in ('id','inputs','artifacts','metrics','elapsed_seconds')}
        put_node(ctl,'r1.'+stage,'evidence',evidence)
        link(ctl,'r1.'+stage,'tests','r1.protocol')
        receipts[stage]=evidence
    write_json(work/'receipts.json',dict(lineage=lineage,receipts=receipts))

if __name__=='__main__': main(Path(sys.argv[1]))
