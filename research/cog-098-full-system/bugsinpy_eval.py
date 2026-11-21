"""Frozen real-project repair transfer; fixed production source is evaluator-only."""
import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tokenize
import xml.etree.ElementTree as ET

import torch
from agent import FullCOGAgent
from panel import file_sha, play

BENCH = '11c5f1eea954a42132cfd06bf257766a7963e0fd'
SWAPS = [('+','-','*'), ('<','<=','>','>=','==','!='), ('and','or'), ('0','1','2')]


def archive(repo, rev, root):
    raw = subprocess.check_output(['git','-C',str(repo),'archive',rev])
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        tar.extractall(root,filter='data')


def check(py, root, node='tests'):
    with tempfile.TemporaryDirectory() as d:
        report = Path(d)/'junit.xml'
        cmd = [str(py),'-B','-m','pytest','-p','no:cacheprovider','-q','--tb=short',node,
               '--junitxml='+str(report)]
        p = subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=45,
                           env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'})
        cases = {}
        for c in ET.parse(report).iter('testcase'):
            name = c.get('classname','')+'::'+c.get('name','')
            cases[name] = next((s for s in ('failure','error','skipped') if c.find(s) is not None),'passed')
        return dict(code=p.returncode,cases=cases,log=(p.stdout+p.stderr).replace(str(root),'.'))


class RealRepair:
    def __init__(self,root,py,node,initial,fixed_cases):
        self.root,self.py,self.node,self.fixed_cases = root,py,node,fixed_cases
        self.public_error = '\n'.join(line for line in initial['log'].splitlines() if 'xml file:' not in line and not re.search(r'in [0-9.]+s', line))
        self.public_error = re.sub(r'0x[0-9a-fA-F]+', '0xADDR', self.public_error)
        refs = re.findall(r'(pysnooper/[\w/]+\.py):(\d+)',self.public_error)
        mentions = set(re.findall(r'\w+',self.public_error.lower()))
        candidates = []
        self.sources = {}
        for path in sorted((root/'pysnooper').rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            code = path.read_text()
            self.sources[rel] = code
            lines = code.splitlines(keepends=True)
            for token in tokenize.generate_tokens(io.StringIO(code).readline):
                group = next((g for g in SWAPS if token.string in g),None)
                if not group:
                    continue
                line,col = token.start
                context = lines[line-1].strip()
                proximity = max((1/(1+abs(line-int(loc))) for f,loc in refs if rel==f),default=0.)
                overlap = len(set(re.findall(r'\w+',context.lower())) & mentions)
                for v in group:
                    if v == token.string:
                        continue
                    action = f'{rel} line {line} col {col} replace {token.string} with {v} in {context}'
                    replacement = lines.copy()
                    replacement[line-1] = lines[line-1][:col]+v+lines[line-1][token.end[1]:]
                    candidate = ''.join(replacement)
                    try:
                        compile(candidate, rel,'exec')
                    except SyntaxError:
                        continue
                    candidates.append((proximity,overlap,action,rel,candidate))
        candidates.sort(key=lambda x:(-x[0],-x[1],x[2]))
        self.pool_size = len(candidates)
        self.menu = {a:(rel,code) for _,_,a,rel,code in candidates[:64]}
        self.tried = []
        self.passed = False
        self.selected = None

    def public(self, assisted=True):
        if not assisted:
            menu = tuple(sorted(self.menu, key=lambda a: hashlib.sha256(a.encode()).digest()))
            return self.public_error,menu,[0.] * len(menu)
        return self.public_error,tuple(self.menu),[1/(1+i) for i in range(len(self.menu))]

    def step(self,action):
        rel,code = self.menu.pop(action)
        target = self.root/rel
        target.write_text(code)
        try:
            outcome = check(self.py,self.root,self.node)
            self.passed = outcome['code']==0 and set(outcome['cases'].values())=={'passed'}
            self.tried.append(dict(action=action,code=outcome['code'],cases=outcome['cases']))
            if self.passed:
                self.selected = (rel,code)
        finally:
            target.write_text(self.sources[rel])
        return self.passed,self.passed or len(self.tried)>=5 or not self.menu

    def result(self):
        suite = {}
        if self.selected:
            rel,code = self.selected
            target = self.root/rel
            target.write_text(code)
            try:
                suite = check(self.py,self.root)['cases']
            finally:
                target.write_text(self.sources[rel])
        return dict(success=bool(self.passed and all(suite.get(c)=='passed' for c,s in
                    self.fixed_cases.items() if s=='passed')),target_passed=self.passed,
                    suite=suite,attempts=self.tried,pool_size=self.pool_size)


def run(cfg):
    out = {}
    bench = cfg.data/'bugs'
    if subprocess.check_output(['git','-C',str(bench),'rev-parse','HEAD'],text=True).strip()!=BENCH:
        raise ValueError('wrong BugsInPy revision')
    receipt = cfg.out / ('bugsinpy-compat.json' if cfg.compat else 'bugsinpy.json')
    for i in (1,2,3):
        folder = bench/'projects/PySnooper/bugs'/str(i)
        info = dict(re.findall(r'(\w+)="([^"]+)"',(folder/'bug.info').read_text()))
        node = (folder/'run_test.sh').read_text().split()[-1]
        if cfg.compat:
            os.environ.update(LC_ALL='C.UTF-8', PYTHONUTF8='1')
        with tempfile.TemporaryDirectory() as tmp:
            buggy,fixed = Path(tmp)/'buggy', Path(tmp)/'fixed'
            repo = cfg.data/'snooper'
            archive(repo,info['buggy_commit_id'],buggy)
            archive(repo,info['fixed_commit_id'],fixed)
            testfile = info['test_file']
            (buggy/testfile).write_bytes((fixed/testfile).read_bytes())
            if cfg.compat and i == 1:
                os.environ.update(LC_ALL='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0')
            if cfg.compat and i == 2:
                shutil.copytree(fixed/'tests/mini_toolbox',buggy/'tests/mini_toolbox')
            fail = check(cfg.python,buggy,node)
            positive = check(cfg.python,fixed,node)
            suite = check(cfg.python,fixed)
            valid = (fail['code']==1 and set(fail['cases'].values())=={'failure'}
                     and positive['code']==0 and set(positive['cases'].values())=={'passed'})
            row = dict(metadata=info,node=node,valid=valid,buggy=fail,fixed=positive,fixed_suite=suite,arms=[])
            out[i] = row
            if not valid:
                receipt.write_text(json.dumps(out,indent=2)+'\n')
                continue
            for arm in ('heuristic','cog','attention'):
                for seed in ((1701,) if arm=='heuristic' else (1701,1707,1711)):
                    start = time.perf_counter()
                    checkpoint = None
                    if arm=='heuristic':
                        agent = FullCOGAgent('heuristic',seed)
                        agent.freeze()
                    else:
                        path = cfg.out/f'{arm}-{seed}.pt'
                        checkpoint = file_sha(path)
                        agent = FullCOGAgent.restore(torch.load(path,weights_only=True,map_location='cpu'))
                    env = RealRepair(buggy,cfg.python,node,fail,suite['cases'])
                    steps, result = play(agent,env,'software',False)
                    row['arms'].append(dict(arm=arm,seed=seed,steps=steps,result=result,checkpoint=checkpoint,
                                            wall_s=time.perf_counter()-start))
        receipt.write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    p = argparse.ArgumentParser()
    for field in ('data','out','python'):
        p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--compat',action='store_true',help='post-preflight fixture and locale sensitivity')
    cfg = p.parse_args()
    cfg.data = cfg.data.resolve()
    run(cfg)
