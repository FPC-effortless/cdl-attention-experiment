"""Independent arithmetic audit of retained 097b per-query raw records.

No model calls, solver secrets, source labels beyond released traces, or timing
replays. Check this before running any experiment that overwrites result JSON.
"""
import argparse, json
from pathlib import Path

ARMS=('overlap','word','char','frozen','learned','hybrid','witness','shuffled_witness','oracle')
KINDS=('familiar','held_paraphrase','unseen_field')


def audit(report):
    assert len(report['results'])==len(report['seeds'])
    for row in report['results']:
        trace=row['traced']; answered=[t for t in trace if t['gold'] is not None]
        unsupported=[t for t in trace if t['gold'] is None]
        assert len(answered)==113 and len(unsupported)==18,(row['seed'],len(trace))
        assert set(r['kind'] for r in trace)==set(KINDS)|{'ambiguous','missing'}
        assert all(r['candidate_count']==7 for r in answered)
        assert sum(r['kind']=='unseen_field' for r in answered)==23
        assert len({(t['kind'],t.get('text_hash')) for t in answered})==len(answered)
        for arm in ARMS:
            correct=sum(t['predictions'][arm]==t['gold'] for t in answered)
            assert row['reader'][arm]['queries']==113
            assert row['reader'][arm]['safe_defer']==18
            assert row['reader'][arm]['top1']==correct,(row['seed'],arm,'top1',correct)
            assert row['reader'][arm]['candidate_checks']==113*7
            assert row['reader'][arm]['latest_version']==113
            for kind in KINDS:
                k=[t for t in answered if t['kind']==kind]
                assert row['by_kind'][kind]['n']==len(k)
                assert row['by_kind'][kind][arm]==sum(x['predictions'][arm]==x['gold'] for x in k)
        assert row['reader']['oracle']['top1']==113
    for arm in ARMS:
        assert report['aggregate'][arm]['top1']==sum(r['reader'][arm]['top1'] for r in report['results'])
        assert report['aggregate'][arm]['queries']==113*len(report['seeds'])
        assert report['aggregate'][arm]['safe_defer']==18*len(report['seeds'])
    return {'seeds':len(report['seeds']), 'answerable':sum(r['reader']['oracle']['queries'] for r in report['results']),
            'unresolved':sum(r['unresolved'] for r in report['results']),
            'top1':{a:report['aggregate'][a]['top1'] for a in ARMS}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--replay',default=None);args=p.parse_args()
    for file in ('097b-results.json','097b-scale-2048.json','097b-scale-32768.json'):
        print(file,audit(json.loads(Path(file).read_text())))
    if args.replay:
        r=json.loads(Path(args.replay).read_text());print('fresh replay:',audit(r))
        assert r['aggregate']['witness']['top1']>=r['aggregate']['char']['top1']


if __name__=='__main__':main()
