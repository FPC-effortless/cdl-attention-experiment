"""Re-induce the physical models from source on another CPU."""
import json
from pathlib import Path
from panel import run
from audit import run as audit
root=Path(__file__).resolve().parent
for seed in (5101,5107,5113):
    actual=run(seed)
    saved=json.loads((root/'results'/str(seed)/'acquisition-candidate/result.json').read_text())['panel']
    assert actual['discovery']==saved['discovery']
    assert actual['retention_matrix']==saved['retention_matrix']
    assert [row[:8] for row in actual['trials']]==[row[:8] for row in saved['trials']]
print(json.dumps(audit()),flush=True)
