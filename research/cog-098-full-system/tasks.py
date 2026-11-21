"""Public observations, local actions and independently scored environment adapters."""
import ast
from collections import deque
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

import gymnasium as gym
import minigrid
from minigrid.core.constants import IDX_TO_OBJECT, STATE_TO_IDX

IDX_TO_STATE = {value: name for name, value in STATE_TO_IDX.items()}

HERE = Path(__file__).resolve().parent
QUIX = '4257f44b0ff1181dedaedee6a447e133219fcebf'
MULTI = 'fe0c8e65cfcd8462bd33c86e35f21addc84ca82b'
AREAS = ('unspecified', 'north', 'south', 'east', 'west', 'centre', 'dontcare')
PRICES = ('unspecified', 'cheap', 'moderate', 'expensive', 'dontcare')
PAIRS = tuple(itertools.product(AREAS, PRICES))
LABELS = tuple('hotel area ' + a + ' price ' + p for a, p in PAIRS)
FAMILIES = ('MiniGrid-Empty-Random-6x6-v0', 'MiniGrid-DoorKey-6x6-v0')
MOVES = ('turn left', 'turn right', 'move forward', 'pick up', 'drop', 'toggle door')


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=8)
def inventory(data, split):
    quix, mw = data/'quix', data/'mw'
    for path, pin in [(quix, QUIX), (mw, MULTI)]:
        rev = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()
        if rev != pin or subprocess.check_output(['git','-C',str(path),'status','--porcelain']):
            raise ValueError('external checkout differs from pinned clean tree: ' + str(path))
    names = sorted((p.stem for p in (quix/'json_testcases').glob('*.json')), key=sha)
    shard = mw/'data/MultiWOZ_2.2'/split/'dialogues_001.json'
    records = json.loads(shard.read_text())
    dialogues = {r['dialogue_id']: r for r in records if 'hotel' in r['services']}
    offset = 1000 if split == 'train' else 2000
    cases = [('software', n) for n in names[0 if split=='train' else 1::2]]
    cases.extend(('dialogue', k) for k in sorted(dialogues))
    cases.extend(('interactive', (f, s)) for f in FAMILIES for s in range(offset, offset+24))
    return cases, dialogues


class Dialogue:
    def __init__(self, record):
        self.turns = []
        system = ''
        gold = ('unspecified', 'unspecified')
        for turn in record['turns']:
            if turn['speaker'] == 'SYSTEM':
                system = turn['utterance']
                continue
            for frame in turn['frames']:
                if frame['service'] == 'hotel':
                    vals = frame['state']['slot_values']
                    gold = tuple(vals.get('hotel-'+s, ['unspecified'])[0] for s in ('area','pricerange'))
            self.turns.append((system, turn['utterance'], gold))
        self.i = 0
        self.last = ('unspecified', 'unspecified')
        self.pred = []

    def public(self, assisted=True):
        system, user, _ = self.turns[self.i]
        query = 'user ' + user + ' previous system ' + system
        # Simple lexical state tracker carries its previous self-selected state.
        if not assisted:
            return query, LABELS, [0.] * len(LABELS)
        prior = list(self.last)
        for j, options in enumerate((AREAS, PRICES)):
            found = [v for v in options[1:-1] if re.search(r'\b'+v+r'\b', user.lower())]
            if found:
                prior[j] = found[-1]
        return query, LABELS, [float(p == tuple(prior)) for p in PAIRS]

    def step(self, action):
        self.last = PAIRS[LABELS.index(action)]
        self.pred.append(self.last)
        ok = self.last == self.turns[self.i][2]
        self.i += 1
        return ok, self.i == len(self.turns)

    def result(self):
        goals = [t[2] for t in self.turns]
        correct = [a == b for a,b in zip(self.pred,goals)]
        changes = [i for i,g in enumerate(goals) if g != (goals[i-1] if i else PAIRS[0])]
        return dict(success=all(correct), final=correct[-1], correct_turns=sum(correct),
                    turns=len(goals), changes=len(changes), correct_changes=sum(correct[i] for i in changes),
                    unavailable_gold=sum(g not in PAIRS for g in goals))


class Navigation:
    def __init__(self, family, seed):
        self.env = gym.make(family)
        self.obs, _ = self.env.reset(seed=seed)
        # Layout digest is evaluator-only. Include starting pose.
        world = self.env.unwrapped
        self.layout = hashlib.sha256(world.grid.encode().tobytes() +
                       repr((tuple(world.agent_pos),world.agent_dir)).encode()).hexdigest()
        self.steps = 0
        self.reward = 0

    def public(self, assisted=True):
        im = self.obs['image']
        tiles = {(x,y): IDX_TO_OBJECT[int(im[x,y,0])] for x,y in itertools.product(range(7),repeat=2)}
        doors = {p:IDX_TO_STATE[int(im[*p,2])] for p,t in tiles.items() if t == 'door'}
        grid = ' / '.join(' '.join(doors.get((x,y),tiles[x,y]) for x in range(7)) for y in range(7))
        text = f"{self.obs['mission']} heading {self.obs['direction']} view {grid}"
        if not assisted:
            return text, MOVES, [0.] * len(MOVES)
        start = (3,6)
        carrying = tiles[start] == 'key'
        # Breadth-first route through publicly visible free tiles to an interaction.
        queue = deque([(start, ())])
        seen = {start}
        choices = []
        while queue:
            (x,y), path = queue.popleft()
            for d,(dx,dy) in enumerate(((0,-1),(1,0),(0,1),(-1,0))):
                p = (x+dx, y+dy)
                if p in seen or p not in tiles:
                    continue
                t = tiles[p]
                seen.add(p)
                route = path + (d,)
                if t == 'goal' or (t == 'key' and not carrying) or (p in doors and doors[p] != 'open' and carrying):
                    choices.append((len(route), route, t))
                if t in ('empty','floor') or doors.get(p) == 'open':
                    queue.append((p, route))
        if choices:
            _, route, target = min(choices)
            d = route[0]
            move = (1 if d in (1,2) else 0) if d else (
                3 if len(route)==1 and target=='key' else 5 if len(route)==1 and target=='door' else 2)
        else:
            move = 2 if tiles[3,5] in ('empty','floor') and self.steps % 4 else 1
        return text, MOVES, [float(i==move) for i in range(6)]

    def step(self, action):
        self.obs, r, term, trunc, _ = self.env.step(MOVES.index(action))
        self.reward = max(self.reward, float(r))
        self.steps += 1
        return bool(r), term or trunc or self.steps >= 48

    def result(self):
        self.env.close()
        return dict(success=self.reward > 0, reward=self.reward, layout=self.layout, steps=self.steps)


OPERATORS = ((ast.Add, ast.Sub, ast.Mult), (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq),
             (ast.And,ast.Or), (ast.BitAnd,ast.BitOr,ast.BitXor))


def edits(source):
    tree = ast.parse(source)
    edits = {}
    for n in ast.walk(tree):
        attrs = []
        if isinstance(n, (ast.BinOp, ast.BoolOp)):
            attrs.append((n,'op',n.op))
        elif isinstance(n, ast.Compare) and len(n.ops)==1:
            attrs.append((n,'ops',n.ops[0]))
        for node, field, old in attrs:
            for group in OPERATORS:
                if type(old) not in group:
                    continue
                for typ in group:
                    if typ is type(old):
                        continue
                    replacement = [typ()] if field=='ops' else typ()
                    original = getattr(node,field)
                    before = ast.unparse(node)
                    setattr(node, field, replacement)
                    action = f'line {n.lineno} col {n.col_offset} replace {before} with {ast.unparse(node)}'
                    edits[action] = ast.unparse(tree)
                    setattr(node,field,original)
        if isinstance(n, ast.Constant) and type(n.value) is int and n.value in (-1,0,1,2):
            old = n.value
            for v in (0,1,2):
                if v != old:
                    n.value = v
                    edits[f'line {n.lineno} col {n.col_offset} change {old} to {v}'] = ast.unparse(tree)
            n.value = old
    return edits


class Repair:
    def __init__(self, root, name):
        self.root, self.name = root,name
        self.source = (root/'python_programs'/(name+'.py')).read_text()
        self.edits = edits(self.source)
        self.remaining = sorted(self.edits, key=sha) or ['keep original']
        self.edits['keep original'] = self.source
        self.tried = []
        self.selected = 'keep original'

    def check(self, action, split):
        try:
            p = subprocess.run([sys.executable,'-B',str(HERE/'quix_worker.py'),str(self.root),self.name,split],
                               input=self.edits[action], capture_output=True, text=True, timeout=6)
        except subprocess.TimeoutExpired:
            return 'timeout'
        if p.returncode:
            raise RuntimeError(p.stderr[-1500:] or f'repair worker exited {p.returncode}')
        status = p.stdout.strip().splitlines()[-1]
        if status not in ('passed','no_cases','timeout','failed'):
            raise RuntimeError(status)
        return status

    def public(self, assisted=True):
        menu = tuple(self.remaining)
        if not assisted:
            return 'repair program '+self.source, menu, [0.] * len(menu)
        prior = [1. if ('replace' in a and any(t in a for t in ('<','>','=='))) else 0. for a in menu]
        return 'repair program '+self.source, menu, prior

    def step(self, action):
        self.remaining.remove(action)
        status = self.check(action,'visible')
        self.tried.append((action,status))
        if status == 'passed':
            self.selected = action
        return status=='passed', status=='passed' or len(self.tried)>=5 or not self.remaining

    def result(self):
        public = self.check('keep original','visible')
        hidden = self.check('keep original','hidden')
        accepted = self.selected != 'keep original'
        scored = self.check(self.selected,'hidden') if accepted else hidden
        success = accepted and scored=='passed' and (public=='failed' or hidden=='failed')
        return dict(success=success, selected=self.selected, selected_hidden=scored, original_visible=public,
                    original_hidden=hidden, original_pass=(public==hidden=='passed'), attempts=self.tried)


def environment(domain, ident, data, dialogues):
    if domain == 'software':
        return Repair(data/'quix', ident)
    if domain == 'dialogue':
        return Dialogue(dialogues[ident])
    return Navigation(*ident)
