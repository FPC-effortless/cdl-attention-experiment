import json
import random
from pathlib import Path
from trainlab.runner import verify_run
from research_lab.ecosystem import _fixture
from research_lab.core import Controller
from research_lab.evidence import put_node
from investigation_world.core.models import InvestigationResult
from investigation_world.companyworld.verifier import verify_companyworld

root=Path(__file__).resolve().parent
labroot=root/'ecosystem/research_lab'
summary={}
for artifact in sorted((labroot/'runs').glob('*/output/predictions.json')):
    run=artifact.parents[1]
    assert verify_run(run)['valid']
    recipe=json.loads((run/'experiment.json').read_text())
    seed=recipe['seed']
    arm=recipe['command'][-1]
    rng=random.Random(seed)
    rows=json.loads(artifact.read_text())
    assert len(rows)==192
    counts={}
    for i,row in enumerate(rows):
        item=i//4
        index=300*seed+3*item+item%2
        change=('unchanged','projection','revision','supersession')[i%4]
        assert (row['index'],row['change'])==(index,change)
        episode=_fixture(index)
        fact=episode.oracle.facts[0]
        value=fact.expected_value+rng.randint(1,500)
        if change in ('revision','supersession'):
            fact.expected_value=value
            record=episode.records[0]
            if change=='supersession':
                record=record.model_copy(deep=True)
                record.record_id+='-NEW'
                episode.records.insert(0,record)
                fact.supporting_record_ids=[record.record_id]
            record.fields[fact.field_name]=value
        elif change=='projection':
            episode.records[1].fields[fact.field_name]=value
        result=InvestigationResult.model_validate(row['submission'])
        reward=verify_companyworld(result,episode).overall_reward
        assert reward==row['reward']
        counts[change]=counts.get(change,0)+int(reward==1)
    summary[f'{arm}-{seed}']=counts
assert len(summary)==6
(root/'audit.json').write_text(json.dumps(summary,indent=2))
put_node(Controller(labroot),'wr008.cache-observation','evidence',
         dict(scope='ENGINEERING_ONLY',results=summary,verification='same-host saved-submission rescoring'))
