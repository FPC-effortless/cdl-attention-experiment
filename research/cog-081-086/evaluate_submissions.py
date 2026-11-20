"""Score model actions for frozen puzzle_cases.jsonl WITHOUT model access to reference paths.

Input: JSONL records {"id": "primary/8101/maze_7/0", "actions": ["D", ...]}
For Sudoku: actions [[zero_based_cell_index, digit], ...]
For chess: actions [["Q", source_square_index, destination_square_index]],
where square indices are a1=0 through h8=63.
For Hanoi: actions [[source_peg, destination_peg], ...]
For sliding and maze: actions are U, D, L or R.
"""
import argparse,json
from collections import defaultdict
from pathlib import Path
from puzzles import Maze,Sudoku,Hanoi,Sliding,ChessEndgame

def from_case(r):
    d=r['domain'];p=r['input']
    if d.startswith('chess_'):
        parse=lambda s:(int(s[1])-1)*8+'abcdefgh'.index(s[0])
        pcs=p['pieces'];heavy='white_queen' if d=='chess_Q' else 'white_rook'
        return ChessEndgame(parse(pcs['white_king']),parse(pcs[heavy]),parse(pcs['black_king']),d[-1])
    if d.startswith('sudoku_'):
        return Sudoku(p['n'],tuple(x for row in p['grid'] for x in row))
    if d.startswith('maze_'):
        n=p['n'];walls={tuple(x) for x in p['walls']}
        return Maze(n,frozenset((i,j) for i in range(n) for j in range(n) if (i,j) not in walls),tuple(p['start']),tuple(p['goal']))
    if d.startswith('hanoi_'):return Hanoi(p['discs'])
    if d.startswith('slide_'):return Sliding(tuple(x for row in p['grid'] for x in row))
    raise ValueError(d)

def normalize(domain,raw):
    if domain.startswith('chess_'):return [tuple(x) for x in raw]
    if domain.startswith(('hanoi_','sudoku_')):return [tuple(x) for x in raw]
    return raw

def evaluate(cases,submission):
    instances={r['id']:r for r in [json.loads(x) for x in Path(cases).read_text().splitlines() if x.strip()]}
    predictions={}
    for line in Path(submission).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if row['id'] in predictions:raise ValueError('duplicate prediction ID: '+row['id'])
        if row['id'] not in instances:raise ValueError('unknown prediction ID: '+row['id'])
        predictions[row['id']]=row
    report=defaultdict(lambda:dict(cases=0,attempted=0,solved=0,steps=0,failed=[]))
    for id,r in instances.items():
        domain=r['domain'];rec=report[domain];rec['cases']+=1
        if id not in predictions:continue
        rec['attempted']+=1
        try:
            actions=normalize(domain,predictions[id]['actions'])
            ok=bool(from_case(r).apply(actions))
            rec['steps']+=len(actions)
        except (ValueError,TypeError,IndexError,KeyError,AssertionError):ok=False
        rec['solved']+=int(ok)
        if not ok:rec['failed'].append(id)
    total=dict(cases=sum(v['cases'] for v in report.values()),attempted=sum(v['attempted'] for v in report.values()),solved=sum(v['solved'] for v in report.values()))
    total['coverage_pct']=round(100*total['attempted']/total['cases'],2)
    total['all_case_success_pct']=round(100*total['solved']/total['cases'],2)
    return {'total':total,'by_domain':dict(sorted(report.items()))}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('cases');ap.add_argument('predictions');ap.add_argument('--out');args=ap.parse_args()
    r=evaluate(args.cases,args.predictions);s=json.dumps(r,indent=2)
    if args.out:Path(args.out).write_text(s+'\n')
    print(s)
