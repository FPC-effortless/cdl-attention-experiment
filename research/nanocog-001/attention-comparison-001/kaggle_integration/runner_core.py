"""Private GPU engineering qualification and capacity probe; zero optimizer steps."""
from datetime import datetime, timezone
import gc
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


class BlockOptionalNetworkKernels(importlib.abc.MetaPathFinder):
    """Select the documented SDPA fallback without loading unpinned HF kernels."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kernels' or fullname.startswith('kernels.'):
            raise ModuleNotFoundError('Optional external kernels disabled by FP32 engineering protocol')
        return None


class CoreDeadline(BaseException):
    """Bypasses individual check exception handlers; caught only by the runner."""


def save(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def main():
    started = time.monotonic()
    os.environ['NANOCHAT_DTYPE'] = 'float32'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    sys.meta_path.insert(0, BlockOptionalNetworkKernels())
    if 'kernels' in sys.modules:
        raise RuntimeError('External kernels package was already imported')
    output = Path.cwd() / 'adapter_results'
    output.mkdir(exist_ok=True)
    record = {
        'schema': 'cog.kaggle-adapter-engineering.v1',
        'run_id': 'COG-ATTENTION-KAGGLE-ADAPTER-001',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'RUNNING', 'optimizer_updates': 0,
        'training_or_evaluation_corpus': None,
        'scientific_comparison_admitted': False,
        'quality_or_efficiency_ranking': None,
        'precision': 'float32',
        'optional_external_kernel_loading': 'BLOCKED; unchanged upstream SDPA fallback',
        'capacity_probe': [],
        'scheduled_arms': [{'arm': name, 'status': 'NOT_RUN'}
                           for name in ('gqa', 'packet', 'local')],
        'current_stage': 'runtime_setup', 'current_arm': None,
    }

    def deadline_handler(signum, frame):
        raise CoreDeadline('Engineering worker exceeded its 720-second core limit')

    signal.signal(signal.SIGALRM, deadline_handler)
    signal.alarm(720)
    save(output / 'run_result.json', record)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This capacity run requires the requested CUDA allocation')
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = 'cuda:0'
        record['runtime'] = {
            'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'device_count': torch.cuda.device_count(),
            'devices': [{'index': i, 'name': torch.cuda.get_device_name(i),
                         'capability': list(torch.cuda.get_device_capability(i)),
                         'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                        for i in range(torch.cuda.device_count())],
            'executed_model_device': device,
            'packages': {name: importlib.metadata.version(name)
                         for name in ('tiktoken', 'rustbpe', 'numpy', 'filelock')},
        }
        from nanochat import flash_attention
        if flash_attention.USE_FA3 or flash_attention.HAS_FA3:
            raise RuntimeError('Unexpected external attention implementation')
        record['attention'] = {'HAS_FA3': flash_attention.HAS_FA3,
                               'USE_FA3': flash_attention.USE_FA3,
                               'wrapper': 'unchanged nanochat SDPA fallback'}
        from test_adapters import run_checks
        record['current_stage'] = 'adapter_checks'
        save(output / 'run_result.json', record)
        checks = run_checks(device)
        save(output / 'adapter_checks.json', checks)
        record['checks_status'] = checks['status']
        if checks['status'] != 'PASS':
            raise RuntimeError('Adapter engineering checks failed; capacity stage not entered')

        from tokenizer_assets import load
        record['current_stage'] = 'tokenizer_load'
        save(output / 'run_result.json', record)
        tokenizer, token_manifest = load(Path(__file__).resolve().parent.parent / 'tokenizer')
        record['tokenizer'] = token_manifest
        # A real 50,257-token vocabulary, meaningful model geometry and full
        # 1,024-token backward are used only to size the future training budget.
        from adapters import build_model_triplet
        record['current_stage'] = 'model_construction'
        save(output / 'run_result.json', record)
        models, provenance = build_model_triplet(
            vocab_size=tokenizer.get_vocab_size(), width=512, layers=12,
            n_head=8, n_kv_head=2, max_seq_len=1024, seed=1729, device=device,
        )
        save(output / 'model_provenance.json', provenance)
        record['capacity_geometry'] = provenance['geometry']
        record['capacity_matched'] = False
        record['capacity_interpretation'] = (
            'Untrained native modules with explicitly identical common-shell weights. '
            'Different backend-specific parameter counts remain visible. '
            'No optimizer, training adequacy, learned capability or speed comparison.'
        )
        record['all_three_models_resident_during_capacity_probe'] = True
        generator = torch.Generator(device='cpu').manual_seed(7331)
        schedule = torch.randint(0, tokenizer.get_vocab_size(), (1, 1025), generator=generator)
        inputs = schedule[:, :-1].to(device)
        targets = schedule[:, 1:].to(device)
        prompt = torch.tensor([tokenizer.encode(
            'The key moved from Alice to Bob. Who has the key?',
            prepend=tokenizer.get_bos_token_id())], dtype=torch.long, device=device)
        for name in ('gqa', 'packet', 'local'):
            record['current_stage'], record['current_arm'] = 'capacity', name
            scheduled = next(row for row in record['scheduled_arms'] if row['arm'] == name)
            scheduled['status'] = 'RUNNING'
            save(output / 'run_result.json', record)
            model = models[name]
            parameters = model.parameter_report()
            if parameters['total_unique_parameters'] > 125_000_000:
                raise RuntimeError('Model exceeds this engineering probe parameter cap')
            model.train()
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocation_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            stage_started = time.monotonic()
            loss = model(inputs, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite capacity loss')
            loss.backward()
            torch.cuda.synchronize()
            backward_seconds = time.monotonic() - stage_started
            gradient_entries = [{'name': p_name, 'present': p.grad is not None,
                                 'finite': p.grad is not None and bool(torch.isfinite(p.grad).all()),
                                 'nonzero': p.grad is not None and bool(p.grad.ne(0).any())}
                                for p_name, p in model.model.named_parameters() if p.requires_grad]
            if not all(entry['present'] and entry['finite'] for entry in gradient_entries):
                raise RuntimeError('Missing/nonfinite native capacity gradient')
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
            del loss
            model.zero_grad(set_to_none=True)
            model.eval()
            with torch.inference_mode():
                session = model.new_session(batch_size=1, max_tokens=1024, session_id=name)
                logits = model.prefill(prompt, session, session_id=name)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('Nonfinite natural-text prefill')
                token = logits[:, -1].argmax(-1, keepdim=True)
                for _ in range(8):
                    logits = model.step(token, session, session_id=name)
                    if not bool(torch.isfinite(logits).all()):
                        raise RuntimeError('Nonfinite owned decode')
                    token = logits[:, -1].argmax(-1, keepdim=True)
                state = model.state_report(session, session_id=name)
                model.reset(session, session_id=name)
            row = {
                'arm': name, 'status': 'PASS_CAPACITY_ONLY',
                'total_parameters': parameters['total_unique_parameters'],
                'common_shell_parameters': parameters['common_shell_parameters'],
                'backend_specific_parameters': parameters['backend_specific_parameters'],
                'backward_shape': [1, 1024], 'optimizer_updates': 0,
                'forward_backward_seconds_for_resource_planning_only': backward_seconds,
                'allocation_before_bytes': allocation_before,
                'peak_allocated_bytes_with_other_models_resident': peak_allocated,
                'peak_reserved_bytes_with_other_models_resident': peak_reserved,
                'gradient_entries': gradient_entries,
                'native_zero_gradient_names': [e['name'] for e in gradient_entries if not e['nonzero']],
                'natural_text_prefill_tokens': prompt.shape[1],
                'owned_decode_steps': 8, 'retained_state': state,
            }
            record['capacity_probe'].append(row)
            scheduled['status'] = row['status']
            save(output / 'run_result.json', record)
            print(json.dumps({'stage': 'capacity', 'arm': name,
                              'parameters': row['total_parameters'], 'status': row['status']}), flush=True)
            del session, logits, token

        # Observe the actual SDPA operator in a separate untimed profiling call.
        record['current_stage'], record['current_arm'] = 'backend_profile', 'gqa'
        save(output / 'run_result.json', record)
        with torch.inference_mode(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            models['gqa'](inputs[:, :16])
            torch.cuda.synchronize()
        record['attention']['observed_operator_names'] = sorted({
            event.key for event in profile.key_averages()
            if any(term in event.key.lower() for term in ('scaled_dot', 'flash_attention', 'efficient_attention'))
        })
        if not record['attention']['observed_operator_names']:
            raise RuntimeError('No actual SDPA operator was observed')
        record['status'] = 'PASS_ENGINEERING_ONLY'
        record['current_stage'], record['current_arm'] = 'complete', None
    except (Exception, CoreDeadline) as error:
        record['status'] = 'FAILED_ENGINEERING'
        for row in record['scheduled_arms']:
            if row['status'] == 'RUNNING':
                row['status'] = 'FAILED'
        record['error'] = {'type': type(error).__name__, 'message': str(error),
                           'traceback': traceback.format_exc()}
    finally:
        signal.alarm(0)
        record['elapsed_seconds'] = time.monotonic() - started
        save(output / 'run_result.json', record)
        print(json.dumps({'run_id': record['run_id'], 'status': record['status'],
                          'elapsed_seconds': record['elapsed_seconds']}), flush=True)
    return 0 if record['status'] == 'PASS_ENGINEERING_ONLY' else 1


if __name__ == '__main__':
    raise SystemExit(main())
