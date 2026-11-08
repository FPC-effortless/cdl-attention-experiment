"""Replay VB-001 evidence and measure reference CPU inference after all fits."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path
import torch
from .study import ARMS, Config, build, dataset, digest_data, evaluate, source_hash


def run(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = Config()
    meta = json.loads((out/'manifest.json').read_text())
    assert meta['source_sha256'] == source_hash()
    assert meta['config'] == json.loads(json.dumps(asdict(cfg)))
    train = torch.load(out/'train.pt', weights_only=True)
    assert digest_data(train) == meta['train_sha256'] == digest_data(dataset(cfg.train_seed, cfg.train_groups))
    tests = torch.load(out/'evaluation_data.pt', weights_only=True)
    results = json.loads((out/'results.json').read_text())
    predictions = torch.load(out/'predictions.pt', weights_only=True)
    assert len(results['results']) == 15
    for name, data in tests.items():
        assert digest_data(data) == results['evaluation_sha256'][name]
    models, receipts = {}, []
    checks = 0
    for item in results['results']:
        arm, seed = item['arm'], item['seed']
        ck = torch.load(out/f'{arm}_{seed}.pt', weights_only=True)
        receipt = json.loads((out/f'{arm}_{seed}.json').read_text())
        assert ck['receipt'] == receipt
        assert receipt['steps'] == cfg.steps and receipt['source_sha256'] == source_hash()
        generator = torch.Generator().manual_seed(100000+seed)
        order_hash = hashlib.sha256()
        for _ in range(cfg.steps):
            idx = torch.randint(len(train['answer']), (cfg.batch,), generator=generator)
            order_hash.update(idx.numpy().tobytes())
        assert order_hash.hexdigest() == receipt['batch_order_sha256']
        torch.testing.assert_close(generator.get_state(), ck['order_rng'], rtol=0, atol=0)
        model = build(arm, cfg).eval()
        model.load_state_dict(ck['model'])
        assert sum(p.numel() for p in model.parameters()) == meta['parameters'][arm]
        for name, data in tests.items():
            metrics, replay = evaluate(model, data)
            assert metrics == item[name]
            original = predictions[f'{arm}/{seed}/{name}']
            for field, value in replay.items():
                if value is not None:
                    torch.testing.assert_close(value, original[field], rtol=0, atol=0)
            for suffix, kwargs in (('_hard', dict(hard=True)), ('_memory_off', dict(ablation='zero'))):
                diag, _ = evaluate(model, data, **kwargs)
                assert diag == item[name+suffix]
            assert item[name+'_memory_off']['accuracy'] == .25
            checks += 3
        models[f'{arm}/{seed}'] = model
        receipts.append(receipt)
    # Each seed's mini-batch sequence is identical in all arms.
    for seed in cfg.seeds:
        assert len({r['batch_order_sha256'] for r in receipts if r['seed']==seed}) == 1
    audit = dict(checkpoint_count=len(receipts), optimizer_steps=sum(r['steps'] for r in receipts),
                 exact_replayed_model_split_conditions=checks, source_sha256=source_hash(),
                 data_hashes_verified=True, minibatch_hashes_verified=True, memory_off_exact_chance=True)
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    # All arms/seeds interleaved in a rotating order. Unbatched classification,
    # same held-out input and one thread; no timed data generation/file IO.
    record, query = tests['iid']['records'][:1], tests['iid']['query'][:1]
    names = list(models)
    timings = {k:[] for k in names}
    with torch.inference_mode():
        for model in models.values():
            for _ in range(32):
                model(record, query)
        for repeat in range(30):
            ordered = names[repeat%len(names):] + names[:repeat%len(names)]
            if repeat % 2:
                ordered = ordered[::-1]
            for name in ordered:
                start = time.perf_counter_ns()
                for _ in range(32):
                    models[name](record, query)
                timings[name].append((time.perf_counter_ns()-start)/32/1e6)
    cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    timing = dict(unit='milliseconds per complete batch-one forward', repeats=30, calls_per_repeat=32,
                  warmup_calls=32, threads=1, cpu=cpu, torch=torch.__version__,
                  raw=timings, median={k:statistics.median(v) for k,v in timings.items()})
    (out/'timings.json').write_text(json.dumps(timing, indent=2)+'\n')
    print(json.dumps(dict(audit=audit, latency_ms=timing['median']), indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)
