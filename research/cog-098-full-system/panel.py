"""Common driver; evaluation never forwards outcomes to the learner."""
import argparse
import gzip
import hashlib
import json
import random
import resource
import time
from pathlib import Path

import torch
from agent import FullCOGAgent
from tasks import environment, inventory, sha


def args(training):
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True, help='directory containing pinned quix and mw clones')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', choices=['cog','attention'] if training else ['cog','attention','heuristic','no-memory'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--unaided',action='store_true')
    return p.parse_args()


def play(agent, env, domain, training):
    agent.reset()
    public = env.public() if agent.assisted else env.public(assisted=False)
    steps = []
    while True:
        agent.observe(*public)
        action = agent.propose()
        provenance = agent.explain_provenance()
        reward, done = agent.act(env)
        steps.append([provenance['state'], provenance['menu'], action, int(reward)])
        public = None if done else (env.public() if agent.assisted else env.public(assisted=False))
        if training:
            agent.learn(reward, None if done else public[0], done)
        if done:
            break
    if training:
        agent.finish(delayed=domain=='interactive')
    # Software held-out cases are never run on training episodes.
    if training and domain == 'software':
        result = dict(public_pass=bool(reward), attempts=env.tried)
    else:
        result = env.result()
    return steps, result


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cfg, training):
    start = time.perf_counter()
    cfg.out.mkdir(parents=True, exist_ok=True)
    phase = 'train' if training else 'test'
    ckpt = cfg.out / f"{'cog' if cfg.arm=='no-memory' else cfg.arm}-{cfg.seed}.pt"
    checkpoint_sha = None
    if training or cfg.arm == 'heuristic':
        agent = FullCOGAgent(cfg.arm,cfg.seed)
        agent.assisted = not cfg.unaided
        if not training:
            agent.freeze()
    else:
        checkpoint_sha = file_sha(ckpt)
        agent = FullCOGAgent.restore(torch.load(ckpt, map_location='cpu', weights_only=True), cfg.arm!='no-memory')
        if agent.assisted == cfg.unaided:
            raise ValueError('action advice configuration mismatch')
        if agent.arm != ('cog' if cfg.arm=='no-memory' else cfg.arm) or agent.seed != cfg.seed:
            raise ValueError('checkpoint configuration mismatch')
    cases, dialogues = inventory(cfg.data.resolve(),phase)
    order = list(cases)
    random.Random(cfg.seed if training else 99).shuffle(order)
    rows = []
    for domain, ident in order:
        t = time.perf_counter()
        env = environment(domain,ident,cfg.data.resolve(),dialogues)
        steps, outcome = play(agent,env,domain,training)
        rows.append(dict(domain=domain,task=ident,actions=steps,result=outcome,wall_s=time.perf_counter()-t))
    if training:
        agent.freeze()
        torch.save(agent.snapshot(),ckpt)
        checkpoint_sha = file_sha(ckpt)
    prefix = cfg.out / f'{phase}-{cfg.arm}-{cfg.seed}'
    trace = prefix.with_suffix('.jsonl.gz')
    with gzip.open(trace,'wt') as f:
        for row in rows:
            f.write(json.dumps(row,separators=(',',':'))+'\n')
    summary = dict(arm=cfg.arm,assisted=agent.assisted,seed=cfg.seed,phase=phase,updates=agent.updates,
                   checkpoint_sha256=checkpoint_sha,parameters=sum(p.numel() for p in agent.policy.parameters()) if cfg.arm!='heuristic' else 0,
                   tasks=len(rows),actions=sum(len(r['actions']) for r in rows),trace_sha256=file_sha(trace),
                   decision_sha256=sha(json.dumps([[r['task'],r['actions'],r['result']] for r in rows],sort_keys=True)),
                   task_list_sha256=sha(json.dumps(cases)),wall_s=time.perf_counter()-start,
                   process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if not training:
        summary['scores'] = {d:sum(r['result']['success'] for r in rows if r['domain']==d)
                             for d in ('software','dialogue','interactive')}
    prefix.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
