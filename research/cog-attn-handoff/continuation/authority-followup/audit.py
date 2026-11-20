import json
import random
from pathlib import Path
from recipe.cache_trial import CHANGES, fixture_pair
from investigation_world.core.models import InvestigationResult
from investigation_world.companyworld.verifier import verify_companyworld
from trainlab.runner import verify_run
from research_lab.core import Controller
from research_lab.evidence import put_node

root = Path(__file__).resolve().parent
lab = root / "ecosystem/research_lab"
summary = {}
for path in sorted((lab/"runs").glob("*/output/predictions.json")):
    run = path.parents[1]
    assert verify_run(run)["valid"]
    config = json.loads((run/"experiment.json").read_text())
    seed,arm = config["seed"], config["command"][-1]
    rng = random.Random(seed)
    key = f"{arm}-{seed}"
    assert key not in summary
    summary[key] = counts = {}
    rows = json.loads(path.read_text())
    assert len(rows) == 48 * len(CHANGES)
    for i,row in enumerate(rows):
        case, offset = divmod(i,len(CHANGES))
        idx = seed*300 + case*3 + case%2
        inc,change = rng.randint(1,500),CHANGES[offset]
        assert (idx,inc,change)==(row["index"],row["increment"],row["change"])
        orig, episode = fixture_pair(idx,inc,change)
        if change=="unobservable_override":
            assert orig.public_payload()==episode.public_payload()
        expected_refresh = change=="revision" or (arm=="candidate" and change in ("supersession","restored_precedence"))
        assert expected_refresh==row["refreshed"]
        reward = verify_companyworld(InvestigationResult.model_validate(row["submission"]),episode).overall_reward
        assert reward==row["reward"] and (row["reference"]==1)==(change!="unobservable_override")
        bucket = counts.setdefault(change,[0,0])
        bucket[0] += int(reward==1)
        bucket[1] += int(row["refreshed"])
assert set(summary)=={f"{a}-{s}" for a in ("candidate","control") for s in (101,113,127)}
(root/"audit.json").write_text(json.dumps(summary,indent=2))
put_node(Controller(lab),"wr008.authority-observation","evidence",dict(scope="ENGINEERING_ONLY",counts=summary))
