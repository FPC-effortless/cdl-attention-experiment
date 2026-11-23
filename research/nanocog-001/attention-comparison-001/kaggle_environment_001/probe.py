"""Bounded Kaggle hardware/runtime admission; no model, fit, dataset or benchmark."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

started = time.monotonic()
record = {
    'schema': 'cog.kaggle-environment.v1',
    'run_id': 'COG-ATTENTION-KAGGLE-ENV-001',
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'purpose': 'hardware_and_framework_qualification_only',
    'model_fits': 0,
    'protected_data_access': False,
    'python': sys.version,
    'platform': platform.platform(),
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'packages': {},
    'devices': [],
    'status': 'RUNNING',
}
for name in ('torch', 'numpy', 'triton', 'transformers', 'datasets', 'pyarrow',
             'tokenizers', 'tiktoken', 'rustbpe', 'kernels', 'pytest'):
    try:
        record['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        record['packages'][name] = None

try:
    import torch
    torch.set_num_threads(1)
    record['torch_cuda'] = torch.version.cuda
    record['cudnn_version'] = torch.backends.cudnn.version()
    record['cuda_available'] = torch.cuda.is_available()
    record['device_count'] = torch.cuda.device_count()
    if not record['cuda_available']:
        raise RuntimeError('Requested CUDA accelerator is not visible.')
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            x = torch.arange(16, device=f'cuda:{index}', dtype=torch.float32,
                             requires_grad=True)
            loss = x.square().sum()
            loss.backward()
            torch.cuda.synchronize(index)
            gradient_ok = torch.equal(x.grad, 2 * x.detach())
            if not gradient_ok:
                raise RuntimeError('CUDA analytic gradient check failed.')
            record['devices'].append({
                'index': index,
                'name': props.name,
                'compute_capability': [props.major, props.minor],
                'total_memory_bytes': props.total_memory,
                'bf16_supported': torch.cuda.is_bf16_supported(),
                'analytic_gradient_check': gradient_ok,
            })
    record['status'] = 'PASS_ENVIRONMENT_ONLY'
except Exception as exc:
    record['status'] = 'FAILED_ENVIRONMENT'
    record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    record['elapsed_seconds'] = time.monotonic() - started
    Path('environment.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)

if record['status'] != 'PASS_ENVIRONMENT_ONLY':
    raise SystemExit(1)
