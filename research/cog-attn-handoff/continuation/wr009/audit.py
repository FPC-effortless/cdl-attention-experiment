import json
import statistics
from collections import defaultdict
from pathlib import Path
from investigation_world.core.models import InvestigationResult
from investigation_world.companyworld.verifier import verify_companyworld
from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run
from recipe.workload import N, episode, trace

root = Path(__file__).resolve().parent
lab = root/"ecosystem/research_lab"
stats = defaultdict(dict)
verified = 0
for path in sorted((lab/"runs").glob("*/output/results.json")):
    directory = path.parents[1]
    assert verify_run(directory)["valid"]
    config = json.loads((directory/"experiment.json").read_text())
    seed,arm = config["seed"],config["command"][-1]
    cells = json.loads(path.read_text())
    assert len(cells)==8
    for cell in cells:
        padding,percent = cell["padding"],cell["percent"]
        cases = [episode(seed,i) for i in range(N)]
        # Oracle tracks two versions and priority independently of the live record list.
        active = [0]*N
        values = [[c.oracle.facts[0].expected_value,c.oracle.facts[0].expected_value+200] for c in cases]
        source_ids = [c.records[0].record_id for c in cases]
        answers = iter(cell["answers"])
        for key,op,delta in trace(seed,percent):
            winner = active[key]
            if op=="swap":
                active[key] = 1-winner
            elif op in ("revision","lower_revision"):
                values[key][winner if op=="revision" else 1-winner] += delta
            elif op=="read":
                case = cases[key]
                fact = case.oracle.facts[0]
                fact.expected_value = values[key][winner]
                rid = source_ids[key]+("-B" if winner else "")
                fact.supporting_record_ids = [rid]
                case.records[0].record_id = rid
                case.records[0].fields[fact.field_name] = fact.expected_value
                answer = InvestigationResult.model_validate(next(answers))
                assert verify_companyworld(answer,case).overall_reward==1
                verified += 1
        assert next(answers,None) is None
        pair = stats[padding,percent,seed]
        assert arm not in pair
        pair[arm] = statistics.median(cell["times_ns"])
assert len(stats)==24 and all(set(pair)=={"control","candidate"} for pair in stats.values())
rows = []
for (padding,percent,seed),pair in sorted(stats.items()):
    rows.append(dict(padding=padding,percent=percent,seed=seed,**pair,
                     speedup=pair["control"]/pair["candidate"],
                     throughput_gain=2048e9*(1/pair["candidate"]-1/pair["control"])))
report = dict(verified=verified,rows=rows)
(root/"audit.json").write_text(json.dumps(report,indent=2))
put_node(Controller(lab),"wr009.mixed-workload","evidence",dict(scope="ENGINEERING_ONLY",**report))
print(verified)
for n in (0,128):
    for w in (1,10,50,90):
        print(n,w,[round(r["speedup"],2) for r in rows if r["padding"]==n and r["percent"]==w])
