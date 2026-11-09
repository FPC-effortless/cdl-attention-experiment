#!/usr/bin/env python3
"""Reconstruct the original byte-identical Foundation ZIP from transport-sized parts."""
from pathlib import Path
import hashlib
p=Path(__file__).resolve().parent
out=p/'CSM-FOUNDATION-001-review.zip'
data=b''.join((p/(out.name+'.part'+str(i))).read_bytes() for i in (1,2))
assert len(data)==14034393
assert hashlib.sha256(data).hexdigest()=='ac8291f94c04a4d73e395f5eba005b8ea5773813e2829cd86514ced605c1233f'
if out.exists():
    assert out.read_bytes()==data, 'Refusing differing archive overwrite'
else:
    with out.open('xb') as f: f.write(data)
print('Original Foundation archive reconstructed and SHA256 verified.')
