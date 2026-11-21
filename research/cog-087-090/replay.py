"""Replay frozen model decisions on another CPU and check the original shift control."""
import json
from copy import deepcopy
from audit import read,summarize
from panel import run
from shifts import run as shifts
from subgoal import score
def strip(value):
    if isinstance(value,dict):return {k:strip(v) for k,v in value.items() if k not in ('ns','times')}
    if isinstance(value,list):return [strip(v) for v in value]
    if isinstance(value,tuple):return [strip(v) for v in value]
    return value
def main():
    for p in ('held','rep'):
        assert strip(run(p))==strip(read(p+'.json')),p
    assert strip(shifts())==strip(read('shifts.json'))
    assert strip(score())==strip(read('subgoal.json'))
    assert summarize()==read('summary.json')
    print('Matched held and replication actions, work counters, reference costs, shifted records and supplementary Hanoi results.')
if __name__=='__main__':main()
