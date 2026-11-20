"""Search reachable branch-head trees and release asset names for historical evidence."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import os
import subprocess
import requests
HERE=Path(__file__).resolve().parent
repo='FPC-effortless/cdl-attention-experiment'
base='https://api.github.com/repos/'+repo
headers={'Authorization':'Bearer '+os.environ['GH_TOKEN']}
def get(url):
    r=requests.get(url,headers=headers,timeout=30);r.raise_for_status();return r
inventory={Path(line.split()[1]).name:line.split()[0] for line in (HERE.parent.parent/'ARCHIVE_SHA256SUMS').read_text().splitlines()}
refs=subprocess.check_output(['git','ls-remote','--heads','origin'],cwd=HERE,text=True).splitlines()
def search(line):
    sha,ref=line.split(); tree=get(f'{base}/git/trees/{sha}?recursive=1').json()
    return dict(ref=ref,sha=sha,truncated=tree['truncated'],paths=[x['path'] for x in tree['tree'] if Path(x['path']).name in inventory])
with ThreadPoolExecutor(4) as pool: branches=list(pool.map(search,refs))
assets=[]
r=get(base+'/releases?per_page=100')
while True:
    assets.extend(a['name'] for release in r.json() for a in release['assets'])
    if 'next' not in r.links:break
    r=get(r.links['next']['url'])
local=[]
for folder in ('/workspace','/mnt/data','/tmp'):
    for root,dirs,files in os.walk(folder):
        dirs[:]=[n for n in dirs if n not in {'.git','site-packages','node_modules'}]
        for n in inventory.keys()&set(files):
            p=Path(root)/n;local.append([str(p),hashlib.sha256(p.read_bytes()).hexdigest()==inventory[n]])
(HERE/'results/archive-search.json').write_text(json.dumps(dict(branches=branches,assets=assets,local=local,targets=inventory),indent=2))
