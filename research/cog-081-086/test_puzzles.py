"""Invariant tests for environment mechanics and independent scoring."""
import json
from puzzles import *
from evaluate_submissions import from_case
from run_pilot import generate,payload,tag


def test_chess_unique_mate_and_fen_conversion():
    for kind in 'QR':
        pool=chess_candidates(kind)
        assert len(pool)>100
        for position in pool[:30]:
            assert not position.black_checked()
            assert len(position.mates())==1
            assert position.apply(position.shortest().path)
            assert not position.apply([('K',0,0)])
            rec={'domain':'chess_'+kind,'input':payload(position)}
            from_payload=from_case(rec)
            assert from_payload==position
            assert rec['input']['fen'].count('K')==1
            assert rec['input']['fen'].count('k')==1


def test_sudoku_uniqueness_and_invalid_solution():
    for n,remove in [(4,9),(9,49)]:
        for seed in (3,41,207):
            p=make_sudoku(seed,n,remove)
            result,count=p.solve(2)
            assert count==1 and result.path
            assert p.apply(result.path)
            assert from_case({'domain':f'sudoku_{n}','input':payload(p)})==p
            assert not p.apply([(-1,10)])
            assert not p.apply(result.path[:-1])


def test_maze_shortest_and_shifts():
    shifts=0
    for seed in range(50):
        p=make_maze(seed,11,0.3)
        path=p.shortest().path
        assert path and p.apply(path)
        assert from_case({'domain':'maze_11','input':payload(p)})==p
        alt=shift_maze(p)
        if alt:
            shifts+=1
            assert not alt.apply(path)
            assert alt.apply(alt.shortest().path)
            assert tag(('maze',payload(p)))!=tag(('maze',payload(alt)))
    assert shifts>=3


def test_hanoi_lengths_and_legality():
    for discs in (3,4,5,6):
        p=Hanoi(discs);solution=p.shortest()
        assert len(solution.path)==2**discs-1
        assert p.apply(solution.path)
        assert from_case({'domain':f'hanoi_{discs}','input':payload(p)})==p
        assert not p.apply([(0,0)])


def test_sliding_solvability():
    for seed in range(10):
        for depth in (8,14):
            p=make_sliding(seed,depth)
            solution=p.shortest()
            assert p.apply(solution.path)
            assert len(solution.path)<=depth
            assert from_case({'domain':'slide_8','input':payload(p)})==p
            assert not p.apply(['X'])
