"""Rescore saved actions, verify artifacts, and report overlap on the public test panel."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import statistics

from panel import file_sha
from tasks import environment, inventory, sha


def rows(prefix):
    with gzip.open(prefix.with_suffix('.jsonl.gz'),'rt') as f:
        return [json.loads(line) for line in f]


def key(row):
    return row['domain'], json.dumps(row['task'])


def rescore(data,dials,row,cache):
    k = key(row),tuple(tuple(a) for a in row['actions'])
    if k in cache:
        return cache[k]
    env = environment(row['domain'],row['task'],data,dials)
    for i, (obs, menu, action, reward) in enumerate(row['actions']):
        text, legal, _ = env.public()
        # The unassisted QuixBugs order and prime panel's assisted order are identical.
        assert obs == sha(text) and menu == sha(repr(tuple(legal))) and action in legal
        bit,done = env.step(action)
        assert bool(reward) == bit and done == (i+1==len(row['actions']))
    score = json.loads(json.dumps(env.result()))
    cache[k] = score
    return score


def stats(test,train):
    d=[x for x in test if x['domain']=='dialogue']
    m=[x for x in test if x['domain']=='interactive']
    s=[x for x in test if x['domain']=='software']
    train_layouts = {x['result']['layout'] for x in train if x['domain']=='interactive'}
    decisions = collections.Counter(a[2] for r in d for a in r['actions'])
    novel = {}
    for x in m:
        r = x['result']
        if r['layout'] not in train_layouts:
            novel[r['layout']] = r['success']
    return dict(software_repaired=[x['task'] for x in s if x['result']['success']],
                dialogue_complete=sum(r['result']['success'] for r in d),
                dialogue_final=sum(r['result']['final'] for r in d),
                correct_changed_turns=sum(r['result']['correct_changes'] for r in d),
                changed_turns=sum(r['result']['changes'] for r in d),
                turns=sum(r['result']['turns'] for r in d),
                unspecified=decisions['hotel area unspecified price unspecified'],
                navigation_by_family={f:sum(x['result']['success'] for x in m if x['task'][0]==f)
                                      for f in sorted({x['task'][0] for x in m})},
                unique_novel_layouts=len(novel),unique_novel_solved=sum(novel.values()),
                overlapped_test_episodes=sum(x['result']['layout'] in train_layouts for x in m),
                actions=sum(len(x['actions']) for x in test),
                episode_wall_s={d:round(sum(x['wall_s'] for x in test if x['domain']==d),3)
                                for d in ('software','dialogue','interactive')})


def run(data, evidence, replay, unaided_replay):
    cases, dials=inventory(data,'test')
    expected={(d,json.dumps(i)) for d,i in cases}
    cache={}
    output={}
    for label,base,rep in [('assisted',evidence,replay),('unaided',evidence/'unaided',unaided_replay)]:
        armstats={}
        names=[f'{arm}-{s}' for arm in ('attention','cog','no-memory') for s in (1701,1707,1711)]
        if label=='assisted':
            names+=['heuristic-1701']
        for name in names:
            prefix=base/('test-'+name)
            test=rows(prefix)
            assert len(test)==len(expected) and {key(x) for x in test}==expected
            meta=json.loads(prefix.with_suffix('.json').read_text())
            assert meta['trace_sha256']==file_sha(prefix.with_suffix('.jsonl.gz'))
            assert meta['decision_sha256']==sha(json.dumps([[r['task'],r['actions'],r['result']] for r in test],sort_keys=True))
            model=name.replace('no-memory','cog').replace('heuristic','cog')
            training=rows(base/('train-'+model))
            train_meta=json.loads((base/('train-'+model+'.json')).read_text())
            assert train_meta['trace_sha256']==file_sha(base/('train-'+model+'.jsonl.gz'))
            assert len(training)==train_meta['updates']==281
            assert not ({key(x) for x in training} & expected)
            if not name.startswith('heuristic'):
                assert meta['checkpoint_sha256']==train_meta['checkpoint_sha256']==file_sha(base/(model+'.pt'))
            for r in test:
                assert rescore(data,dials,r,cache)==r['result'], (name,r['task'])
            res=stats(test,training)
            if name.startswith('cog-') and rep:
                repeat=json.loads((rep/(prefix.name+'.json')).read_text())
                res['restore_replay_equal']=repeat['decision_sha256']==meta['decision_sha256']
                assert res['restore_replay_equal']
            res.update(test_wall_s=meta['wall_s'],parameters=meta['parameters'],
                       decision_sha256=meta['decision_sha256'],checkpoint_sha256=meta['checkpoint_sha256'])
            armstats[name]=res
        output[label]=armstats
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('data','evidence','replay','unaided-replay','out'):
        p.add_argument('--'+field,type=Path,required=field not in ('replay','unaided-replay'))
    a=p.parse_args()
    result=run(a.data.resolve(),a.evidence,a.replay,a.unaided_replay)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
