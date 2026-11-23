"""Portable access to the immutable COG-FUSION-001 implementation and inputs."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.with_name('cog_fusion_001')
if not (PARENT / 'fusion/model.py').is_file():
    raise FileNotFoundError('COG-FUSION-002 requires the pinned sibling cog_fusion_001 source')
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

SEEDS = (17, 29, 41)
ARMS = ('continuous', 'reset_training')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

def parent_checkpoint(seed):
    return PARENT / 'evidence/runs' / f'memory_seed{seed}' / 'final.pt'

