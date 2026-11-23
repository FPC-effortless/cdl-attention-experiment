"""Bounded continuation launcher; no changes to the frozen training program."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


ROOT = Path('/workspace/scratch/2a5ba884844e/cog_tiny_10m')
RUN = ROOT / 'runs' / 'tiny10m_seed20261122'


def append(payload):
    with (RUN/'launcher_sessions.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--max-wall-seconds',type=int,default=900)
    parser.add_argument('--allocator',choices=('glibc','jemalloc'),default='glibc')
    parser.add_argument('--log-every',type=int,choices=(1,10),default=10)
    args=parser.parse_args()
    if args.threads not in (2,4,8) or not 60 <= args.max_wall_seconds <= 1100:
        raise ValueError('Unadmitted runtime setting')
    manifest=json.loads((RUN/'run_config.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen source changed: '+name)
    cfg=manifest['training_config']
    model_cfg_path=RUN/'launcher_model_config.json'
    model_text=json.dumps(manifest['model_config'],sort_keys=True,indent=2)+'\n'
    if model_cfg_path.exists() and model_cfg_path.read_text()!=model_text:
        raise RuntimeError('Launcher model configuration differs')
    if not model_cfg_path.exists():model_cfg_path.write_text(model_text)
    argv=[sys.executable,'-u',str(ROOT/'train.py'),
          '--data-dir',str(ROOT/'data'),'--out-dir',str(RUN),
          '--device','cpu','--threads',str(args.threads),
          '--eval-batch-size','16','--log-every',str(args.log_every),
          '--max-wall-seconds',str(args.max_wall_seconds),
          '--model-config',str(model_cfg_path),'--resume',str(RUN/'resume.pt')]
    for name in ('seed','batch_size','gradient_accumulation','context_length','max_steps',
                 'learning_rate','min_lr_ratio','warmup_steps','weight_decay','grad_clip',
                 'eval_every','patience','minimum_steps','minimum_improvement','precision'):
        argv.extend(['--'+name.replace('_','-'),str(cfg[name])])
    env=os.environ.copy()
    env['MALLOC_ARENA_MAX']='2'
    allocator={'MALLOC_ARENA_MAX':'2'}
    if args.allocator=='jemalloc':
        library=Path('/usr/lib/x86_64-linux-gnu/libjemalloc.so.2')
        if not library.is_file():raise RuntimeError('Approved allocator library unavailable')
        existing=env.get('LD_PRELOAD','')
        env['LD_PRELOAD']=str(library)+(':'+existing if existing else '')
        allocator.update(LD_PRELOAD=env['LD_PRELOAD'],library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    start=time.time()
    token=f'{int(start)}-{os.getpid()}'
    log_path=RUN/f'launcher_{token}.log'
    common={'session_token':token,'start_unix':start,'argv':argv,
            'allocator_environment':allocator,
            'threads':args.threads,'max_wall_seconds':args.max_wall_seconds,
            'source_sha256':manifest['source_sha256'],'log':str(log_path),
            'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    append({'event':'launcher_start',**common})
    child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,text=True,bufsize=1)
    def forward(signum,frame):
        if child.poll() is None:child.send_signal(signum)
    signal.signal(signal.SIGTERM,forward)
    signal.signal(signal.SIGINT,forward)
    with log_path.open('w',encoding='utf-8') as stream:
        for line in child.stdout:
            stream.write(line);stream.flush()
            print(line,end='',flush=True)
    code=child.wait()
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt={'event':'launcher_end',**common,'end_unix':time.time(),
             'returncode':code,'max_child_rss_kib':usage.ru_maxrss,
             'child_user_seconds':usage.ru_utime,'child_system_seconds':usage.ru_stime}
    append(receipt)
    print(json.dumps(receipt),flush=True)
    return code if code>=0 else 128-code


if __name__=='__main__':raise SystemExit(main())
