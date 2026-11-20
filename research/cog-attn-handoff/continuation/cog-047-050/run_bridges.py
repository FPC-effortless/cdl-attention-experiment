"""Regenerate the local integration artifacts without modifying the frozen runs."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"recipe"))

mode=sys.argv[1]
if mode=="modules":
    from modules import run
    from lineage import experiment
    results={"modules":run(),"lineage":experiment()}
elif mode=="online":
    from online import run
    results={"online":run(ROOT/"fusion.json")}
elif mode=="dialogue-repair":
    from dialogue_repair import run
    results={"dialogue-repair":run()}
else:
    raise SystemExit("valid options: modules, online, dialogue-repair")
for key,value in results.items():
    (ROOT/f"{key}.json").write_text(json.dumps(value,indent=2))
