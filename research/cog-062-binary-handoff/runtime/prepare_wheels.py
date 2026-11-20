"""Derive a local SHA256 lock from verified official versioned metadata; no installs."""
from pathlib import Path
from hashlib import sha256
import json,subprocess,concurrent.futures,sys
from urllib.parse import urlparse
from importlib.metadata import version,PackageNotFoundError
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename,canonicalize_name
from packaging.requirements import Requirement
from zipfile import ZipFile
from email.parser import BytesParser
root=Path(__file__).resolve().parent
metadata=root/'metadata'
wanted={'pyproj':'3.7.2','shapely':'2.1.2','SQLAlchemy':'2.0.44','duckdb':'1.4.2','typer':'0.20.0','shellingham':'1.5.4','rich':'14.2.0','markdown-it-py':'4.0.0','mdurl':'0.1.2'}
tags={t:i for i,t in enumerate(sys_tags())}
pinned={canonicalize_name(k):v for k,v in wanted.items()}
index=[];base={}
def requirements(items):
    for q in items or ():
        r=Requirement(q)
        if r.marker and not r.marker.evaluate({'extra':''}):continue
        n=canonicalize_name(r.name)
        if n in pinned:v=pinned[n]
        else:
            try:v=version(n)
            except PackageNotFoundError: raise ValueError("unresolved transitive package: "+str(r)) from None
            base[n]=v
        if r.url or not r.specifier.contains(v,prereleases=True):raise ValueError("dependency pin does not satisfy "+str(r))
for name,v in wanted.items():
    p=metadata/(name+'.json')
    d=json.loads(p.read_bytes());meta=d['info']
    if canonicalize_name(meta['name'])!=canonicalize_name(name) or meta['version']!=v:raise ValueError('metadata identity mismatch')
    requirements(meta['requires_dist'])
    choices=[]
    for f in d['urls']:
        if f['packagetype']!='bdist_wheel' or f['yanked']:continue
        _,fv,_,ts=parse_wheel_filename(f['filename']); score=min((tags.get(t,999999) for t in ts),default=999999)
        if str(fv)==v and score<999999:choices.append((score,f['filename'],f))
    if not choices:raise ValueError('no supported wheel for '+name)
    _,_,item=min(choices,key=lambda x:x[:2])
    u=urlparse(item['url'])
    if (u.scheme,u.netloc)!=('https','files.pythonhosted.org') or not u.path.endswith('/'+item['filename']):raise ValueError('untrusted wheel destination')
    index.append(dict(name=name,version=v,filename=item['filename'],url=item['url'],sha256=item['digests']['sha256'],size=item['size'],metadata_sha256=sha256(p.read_bytes()).hexdigest()))
(root/'WHEEL_PLAN.json').write_text(json.dumps(dict(wheels=index,requirements_satisfied_by_base=base,status='OFFICIAL_METADATA_ONLY_NOT_INSTALLED'),sort_keys=True,indent=2)+'\n')
wheelhouse=root/'wheelhouse';wheelhouse.mkdir(exist_ok=True)
def fetch(item):
    p=wheelhouse/item['filename']
    if not p.exists():
        t=p.with_suffix('.whl.part')
        subprocess.run(['curl','--fail','--silent','--show-error','--retry','1','--retry-max-time','14','--connect-timeout','6','--max-time','45','--max-filesize',str(item['size']),item['url'],'--output',str(t)],check=True)
        t.replace(p)
    if p.stat().st_size!=item['size'] or sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('wheel integrity mismatch '+p.name)
    with ZipFile(p) as z:
        routes=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
        if len(routes)!=1:raise ValueError('ambiguous wheel metadata: '+p.name)
        m=BytesParser().parsebytes(z.read(routes[0]))
        if canonicalize_name(m['Name'])!=canonicalize_name(item['name']) or m['Version']!=item['version']:raise ValueError('wheel metadata mismatch')
        return m.get_all('Requires-Dist',[])
if sys.argv[1:]==['plan']: raise SystemExit(0)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for raw in pool.map(fetch,index):requirements(raw)
(root/'WHEEL_LOCK.json').write_text(json.dumps(dict(wheels=index,requirements_satisfied_by_base=base),sort_keys=True,indent=2)+'\n')
(root/'REQUIREMENTS-ADDITIONAL.lock').write_text(''.join(f"{x['name']}=={x['version']} --hash=sha256:{x['sha256']}\n" for x in index))
print('verified',len(index),'compatible official wheels; base dependencies',base)
