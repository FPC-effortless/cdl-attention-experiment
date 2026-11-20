import json
import sys
from pathlib import Path
from research_lab.core import Controller
from research_lab.evidence import put_node
from research_lab.execution import run_worker

root=Path(__file__).resolve().parent
inputs=root/"recipe"
lab=Controller(root/"ecosystem/research_lab")
lab.program("cog-047-050","Grounded interaction, original modules and continual runtime",3600,0)
lab.resource("qualification-cpu","cpu",1,0)
for suite,description in {
    "continual_v3":"Matched from-scratch controllers with explicit active-frame type and reusable KV buffers."
}.items():
    for arm in ("candidate","control"):
        modes=(("local",) if arm=="candidate" else ("attention",)) if suite=="authentic" else (
            ("local",) if arm=="candidate" else ("full","scoped"))
        files=["entry.py","state_store.py"]
        if suite=="authentic":
            files+=["authentic.py",*[f"multiwoz/{s}/dialogues_001.json" for s in ("train","dev","test")]]
        else:
            files+=["causal_v3.py","continual_v3.py","lineage.py"]
        spec=dict(schema="trainlab.experiment.v1",name=f"{suite}-{arm}",
                  execution_purpose="engineering_diagnostic",description=description,
                  command=[sys.executable,"entry.py",suite,arm],seed=947,timeout_seconds=180,
                  source_kind="CONSTRUCTED",
                  split_identity=f"cog-{suite}-v1",inputs=files,
                  artifacts=["results.json","results.sha256",*[m+".pt" for m in modes]],
                  metrics={"score":"max"},budget={"seeds":3})
        (inputs/f"{suite}-{arm}.json").write_text(json.dumps(spec,indent=2))
    protocol=dict(schema="research_lab.protocol.v1",id=f"cog-qualification-{suite}",program="cog-047-050",
        author="codex",question=description,hypothesis="The runtime can execute the declared scope under the acceptance gates.",
        alternatives=["Exact actions remain below gate.","High fixed costs dominate runtime."],
        predictions=["Scoped and full attention can differ at long context.","Accuracy gates constrain timing interpretations."],
        controls=["Independent fits and matched data as specified in the written protocol."],
        ablations=["Per-cell context lengths and full/scoped context." if suite=="continual" else "Distinct development and test dialogue shards."],
        metric="score",direction="max",minimum_effect=0,seeds=[947,971,997],resource="qualification-cpu",
        slots=1,max_attempts=1,stopping_rule="Run the declared CPU workload cells.",
        scope="ENGINEERING_ONLY",priority=1,source_commit="eec439492407b0aa26922c23b5ae40d95289f4eb",
        next_test="Acquire the remaining trained modules and live external user and authority interaction.",
        limitations=["Scope labels refer to designed tests. The authentic vocabulary is crowdsourced and state patches are typed."],
        candidate=f"{suite}-candidate.json",control=f"{suite}-control.json",replication_tolerance=0,
        replay_artifacts=["results.sha256"])
    (inputs/f"{suite}-protocol.json").write_text(json.dumps(protocol,indent=2))
    lab.register(protocol,inputs)
    put_node(lab,f"cog.qualification.{suite}.protocol","experiment",protocol)
lab.worker("qualification-v3-local","experiment","qualification-cpu")
run_worker(lab,"qualification-v3-local")
(root/"lab-status-v3.json").write_text(json.dumps(lab.status(),indent=2,default=str))
