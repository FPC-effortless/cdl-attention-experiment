import json
import statistics
import sys
from pathlib import Path
from trainlab.runner import verify_run
from research_lab.core import Controller
from research_lab.evidence import put_node
from investigation_world.companyworld.verifier import verify_companyworld
from investigation_world.core.models import InvestigationResult

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root/"recipe"))
from workload import ARMS, CHECKPOINT, fixture, setup, trace

lab = root/"ecosystem/research_lab"
rows = []
for path in sorted((lab/"runs").glob("*/output/results.json")):
    rd = path.parents[1]
    assert verify_run(rd)["valid"]
    cfg = json.loads((rd/"experiment.json").read_text())
    seed = cfg["seed"]
    _,_,bank = setup(seed)
    for cell in json.loads(path.read_text()):
        mode,ratio = cell["mode"],cell["ratio"]
        gold = list(range(4))*8
        actual = gold.copy()
        worlds = [fixture(seed,k) for k in range(32)]
        wi = qi = errors = stale = correct = 0
        for kind,key,_ in trace(seed,ratio):
            if kind=="write":
                prediction = cell["actions"][wi]
                truth = bank[key][1]
                errors += prediction!=truth
                for state,action in [(gold,truth),(actual,prediction)]:
                    dst,src,op = action%32,action//32%32,action//1024
                    state[dst] = state[src] if op==0 else (state[src]+state[dst])%4
                wi += 1
            else:
                result = InvestigationResult.model_validate(cell["reads"][qi])
                episode = worlds[key]
                fact = episode.oracle.facts[0]
                fact.expected_value = gold[key]
                episode.records[0].fields[fact.field_name] = gold[key]
                correct += verify_companyworld(result,episode).overall_reward==1
                stale += result.claims[0]["value"]!=actual[key]
                qi += 1
        assert (len(cell["reads"]),len(cell["actions"]))==(qi,wi)
        rows.append(dict(seed=seed,ratio=ratio,mode=mode,reads=qi,writes=wi,errors=errors,
                         stale=stale,correct=correct,hits=cell["hits"],compiled=cell["compiled"],
                         asks=cell["asks"],ns=statistics.median(cell["times"])))
assert {(r['seed'],r['mode'],r['ratio']) for r in rows}=={(s,m,w) for s in CHECKPOINT for modes in ARMS.values() for m in modes for w in (1,10,50,90)}
for r in rows:
    if r['mode']!='delayed':
        assert r['stale']==0
    if r['mode'] not in ('unassisted','delayed'):
        ref=next(x for x in rows if (x['seed'],x['ratio'],x['mode'])==(r['seed'],r['ratio'],'fresh'))
        assert all(r[k]==ref[k] for k in ('errors','asks','correct'))
(root/"audit.json").write_text(json.dumps(rows,indent=2))
put_node(Controller(lab),"wr010.learned-runtime","evidence",dict(scope="ENGINEERING_ONLY",results=rows))
