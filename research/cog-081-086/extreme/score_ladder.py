"""Reference-independent action replay for the COG-ATTN-087 difficulty ladder.

This scorer validates submitted actions and does not load hidden reference metrics.
Chess strategies must cover all adversarial Black replies to be scored successful.
"""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
from difficulty import Sliding4,encode_puzzle
from puzzles import ChessEndgame,Hanoi,Maze,Sliding,Sudoku

MOVE_LIMIT=100000


def environment(case):
    d=case['domain'];b=case['input']
    if d=='chess':
        # Only KQK/KRK positions are supported, not arbitrary chess FENs.
        rows=b['fen'].split()[0].split('/');board={}
        for rank,row in enumerate(rows):
            file=0
            for c in row:
                if c.isdigit():file+=int(c)
                else:board[c]=8*(7-rank)+file;file+=1
            if file!=8:raise ValueError('Invalid chess row')
        heavy='Q' if 'Q' in board else 'R'
        if sorted(board.keys())!=sorted(['K','k',heavy]):raise ValueError('Outside restricted chess scope')
        return ChessEndgame(board['K'],board[heavy],board['k'],heavy)
    if d=='maze':
        walls={tuple(p) for p in b['walls']}
        return Maze(b['n'],frozenset((r,c) for r in range(b['n']) for c in range(b['n']) if (r,c) not in walls),
                    tuple(b['start']),tuple(b['goal']))
    if d=='sudoku':return Sudoku(b['n'],tuple(v for row in b['grid'] for v in row))
    if d=='sliding3':return Sliding(tuple(v for row in b['grid'] for v in row))
    if d=='sliding4':return Sliding4(tuple(v for row in b['grid'] for v in row))
    if d=='hanoi':return Hanoi(b['discs'])
    raise ValueError(d)


def score_chess(p,entry,max_moves):
    strategy=entry.get('strategy') or {}
    # Single-action backward compatibility, legal only for initial position.
    if 'actions' in entry and len(entry['actions'])==1:
        strategy={**strategy,encode_puzzle(p)['fen']:entry['actions'][0]}
    def verify(pos,remaining):
        if remaining<=0:return False
        key=encode_puzzle(pos)['fen']
        m=strategy.get(key)
        if not isinstance(m,(list,tuple)) or len(m)!=3:return False
        choices={(name,old,new):child for name,old,new,child in pos.white_moves()}
        try:child=choices.get((str(m[0]),int(m[1]),int(m[2])))
        except (TypeError,ValueError):return False
        if child is None:return False
        if child.mate():return True
        if remaining==1:return False
        replies=child.black_replies()
        if not replies:return False
        return all(bk!=child.heavy and verify(ChessEndgame(child.wk,child.heavy,bk,child.kind),remaining-1)
                   for bk in replies)
    return verify(p,max_moves)


def score_case(case,submission):
    p=environment(case);domain=case['domain']
    if submission is None:return {'status':'missing','success':False}
    if domain=='chess':
        return dict(status='scored',success=bool(score_chess(p,submission,case['rules']['maximum_required_mate_in'])))
    path=submission.get('actions',[])
    if not isinstance(path,list):return {'status':'invalid_format','success':False}
    if len(path)>MOVE_LIMIT:
        return {'status':'action_budget_exceeded','success':False,'budget':MOVE_LIMIT}
    if domain=='hanoi' and p.discs>12:
        # The offline JSONL scorer has no streaming protocol for millions of moves.
        return {'status':'requires_streaming_evaluator','success':False,'scored':False}
    try:
        if domain in ('maze','sliding3','sliding4'):
            if not all(a in ('U','D','L','R') for a in path):return {'status':'invalid_action','success':False}
        if domain=='sudoku':
            if not all(isinstance(a,list) and len(a)==2 and type(a[0])==int and type(a[1])==int for a in path):
                return {'status':'invalid_action','success':False}
        if domain=='hanoi':
            if not all(isinstance(a,list) and len(a)==2 and all(type(x)==int for x in a) for a in path):
                return {'status':'invalid_action','success':False}
            path=[tuple(a) for a in path]
        ok=bool(p.apply(path))
    except (ValueError,KeyError,IndexError,AssertionError,TypeError):
        ok=False
    result={'status':'scored','success':ok}
    if domain=='maze' and 'shifted_input' in case['rules']:
        if 'shift_actions' in submission:
            amended={**case,'input':case['rules']['shifted_input']}
            shifted=environment(amended)
            shift_path=submission['shift_actions']
            result['shift_success']=isinstance(shift_path,list) and len(shift_path)<=MOVE_LIMIT and \
                all(a in ('U','D','L','R') for a in shift_path) and shifted.apply(shift_path)
        else:result['shift_success']=None
    return result


def score(cases_path,submissions_path,out_path):
    cases=[json.loads(s) for s in Path(cases_path).read_text().splitlines() if s]
    entries=[json.loads(s) for s in Path(submissions_path).read_text().splitlines() if s]
    ids={r['id'] for r in cases}
    seen=set()
    for e in entries:
        assert e['id'] not in seen, 'Duplicate submission ID'
        assert e['id'] in ids, 'Unknown submission ID'
        seen.add(e['id'])
    sub={e['id']:e for e in entries}
    rows=[]
    for c in cases:
        r=score_case(c,sub.get(c['id']))
        rows.append(dict(id=c['id'],domain=c['domain'],level=c['level'],split=c['split'],**r))
    by_level={str(level):{
        'total':sum(r['level']==level for r in rows),
        'submitted':sum(r['level']==level and r['status']!='missing' for r in rows),
        'scored':sum(r['level']==level and r['status']=='scored' for r in rows),
        'solved':sum(r['level']==level and r['success'] for r in rows)
        } for level in range(1,11)}
    result={'status':'agent-submission action replay (not reference solution evaluation)',
            'submitted':len(entries),'total':len(rows),'scored':sum(r['status']=='scored' for r in rows),
            'solved':sum(r['success'] for r in rows),'by_level':by_level,'rows':rows}
    Path(out_path).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',default=str(Path(__file__).parent/'results'/'cases.jsonl'))
    p.add_argument('--submissions',required=True)
    p.add_argument('--out',default=str(Path(__file__).parent/'results'/'submission_score.json'))
    args=p.parse_args();r=score(args.cases,args.submissions,args.out)
    print('Scored',r['scored'],'solved',r['solved'],'of',r['total'])
