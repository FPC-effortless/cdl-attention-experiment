"""Static preparation: keep gold metadata out of evaluator's prediction stage."""
import json
from pathlib import Path
from forge_data.core import canonical
p=Path(__file__).parent/'data'
e=json.loads((p/'E.json').read_bytes())
for name,keys in (
 ('E-input.json',('dialogue','group','turn','text')),
 ('E-gold.json',('dialogue','group','turn','truth','prior','delta','quality'))):
    with (p/name).open("xb") as f: f.write(canonical([{k:v[k] for k in keys} for v in e]))
