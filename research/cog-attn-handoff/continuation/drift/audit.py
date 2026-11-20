import json
import sys
from pathlib import Path
from investigation_world.companyworld.verifier import verify_companyworld
from investigation_world.core.models import InvestigationResult
from research_lab.core import Controller
from research_lab.evidence import put_node
from trainlab.runner import verify_run
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"recipe"))
from workload import CHECKPOINT,fixture,setup,trace
from drift_trial import ARMS,RATIOS

lab=root/"ecosystem/research_lab"
rows=[]
for path in sorted((lab/"runs").glob("*/output/results.json")):
    rd=path.parents[1]
    assert verify_run(rd)["valid"]
    seed=json.loads((rd/"experiment.json").read_text())["seed"]
    _,_,bank=setup(seed)
    for cell in json.loads(path.read_text()):
        epochs=[[0,0] for _ in range(4)]
        selector=[0]*4
        state=list(range(4))*8
        worlds=[fixture(seed,k) for k in range(32)]
        wi=ri=bad=wrong=asks=races=bad_at_race=0
        for kind,key,_ in trace(seed,cell["ratio"]):
            if kind=="write":
                command,base=bank[key]
                g=sum(ord(ch) for ch in command.split()[1])%4
                phase=(wi+1)%32
                if phase==24:
                    selector[g]^=1
                elif phase in (8,16,31):
                    epochs[g][selector[g]^int(phase==16)]+=1
                truth=base^(1024*((epochs[g][selector[g]]+selector[g])%2))
                actual,calls,_=cell["writes"][wi]
                wrong+=actual!=truth
                asks+=calls
                races+=phase==31
                bad_at_race+=phase==31 and actual!=truth
                dst=truth%32
                state[dst]=(state[truth//32%32]+(truth//1024)*state[dst])%4
                wi+=1
            else:
                episode=worlds[key]
                fact=episode.oracle.facts[0]
                fact.expected_value=state[key]
                episode.records[0].fields[fact.field_name]=state[key]
                result=InvestigationResult.model_validate(cell["reads"][ri])
                bad+=verify_companyworld(result,episode).overall_reward!=1
                ri+=1
        assert wi==len(cell["writes"]) and ri==len(cell["reads"])
        rows.append(dict(seed=seed,mode=cell["mode"],ratio=cell["ratio"],writes=wi,reads=ri,
                         wrong=wrong,bad_reads=bad,calls=asks,races=races,bad_at_race=bad_at_race))
assert len(rows)==54 and len({(r["mode"],r["seed"],r["ratio"]) for r in rows})==54
(root/"audit.json").write_text(json.dumps(rows,indent=2))
put_node(Controller(lab),"drift.confirmed-bindings","evidence",dict(scope="ENGINEERING_ONLY",rows=rows))
for mode in sorted({r["mode"] for r in rows}):
    group=[r for r in rows if r["mode"]==mode]
    print(mode,{k:sum(r[k] for r in group) for k in ("wrong","bad_reads","calls","writes","reads","races","bad_at_race")})
