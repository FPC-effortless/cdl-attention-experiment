"""Decode compact, exact case decisions and reward-outcome trace prefixes."""
from pathlib import Path
import argparse,base64,gzip,json

def unpack(path='EVIDENCE_COMPACT.json.gz.b64'):
    b=base64.b64decode(Path(path).read_text().strip(),validate=True)
    d=json.loads(gzip.decompress(b));assert set(d)=={'1701','1707','1711'}
    for seed,v in d.items():
        assert len(v['test'])==5
        assert all(len(rows)==184 for rows in v['test'].values())
        assert v['events']
    return d

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='EVIDENCE_DECODED.json');a=parser.parse_args()
    obj=unpack();Path(a.out).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print('Unpacked',sum(len(x['events']) for x in obj.values()),'training actions and',sum(sum(len(z) for z in x['test'].values()) for x in obj.values()),'test-method decisions')
