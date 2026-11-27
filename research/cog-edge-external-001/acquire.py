"""Acquire the independently published CMU battery telemetry; retain only licensed battery fields."""
import argparse,csv,hashlib,io,json,math,pathlib,re,time,urllib.request,zipfile
import numpy as np
URL='https://api.figshare.com/v2/articles/12683453/versions/1'
DOI='10.1184/R1/12683453'
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def get(url,limit):
 for attempt in range(3):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'COG-EDGE academic data reproduction'})
   with urllib.request.urlopen(req,timeout=90) as r:
    buf=r.read(limit+1);assert len(buf)<=limit,'download exceeds declared limit';return buf
  except Exception:
   if attempt==2:raise
   time.sleep(2**attempt)
def download(meta,name,limit):
 f=[v for v in meta['files'] if v['name']==name];assert len(f)==1,(name,[x['name'] for x in meta['files']])
 f=f[0];assert 0<f['size']<=limit
 b=get(f['downloadUrl'] if 'downloadUrl' in f else f['download_url'],limit)
 assert len(b)==f['size']
 md5=hashlib.md5(b).hexdigest();expected=f.get('computed_md5') or f.get('supplied_md5')
 assert isinstance(expected,str) and md5.lower()==expected.lower(),(name,md5,expected)
 return b,{'name':name,'id':f['id'],'bytes':len(b),'md5':md5,'sha256':sha(b)}
def normkey(x):return x.strip().lower()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 raw=get(URL,4*1024*1024);m=json.loads(raw)
 assert DOI.lower() in (m.get('doi') or '').lower() and int(m.get('version',1))==1
 license=m.get('license',{});s=json.dumps(license).lower()
 assert 'cc by 4.0' in s or 'creativecommons.org/licenses/by/4.0' in s,license
 receipt={'doi':m.get('doi'),'version':1,'title':m.get('title'),'license':license,'authors':[x.get('full_name',x.get('name')) for x in m.get('authors',[])],'metadata_sha256':sha(raw),'published_files':[{'name':x['name'],'id':x['id'],'size':x['size']} for x in m['files']]}
 write(out/'repository-metadata.json',receipt)
 params,pr=download(m,'parameters.csv',2*1024*1024);archive,zr=download(m,'flights.zip',300*1024*1024)
 receipt['downloads']=[pr,zr];write(out/'repository-metadata.json',receipt)
 records={}
 for x in csv.DictReader(io.StringIO(params.decode('utf-8-sig'))):
  x={normkey(k):v for k,v in x.items() if k is not None}
  fid=int(float(x['flight']));assert fid not in records,'duplicate metadata flight identifier'
  records[fid]={'date':(x.get('date') or '').strip()}
 assert 180<=len(records)<=220,len(records)
 accepted=[];rejected=[];arr={};seen=set();digests=set()
 with zipfile.ZipFile(io.BytesIO(archive)) as z:
  members=sorted([x for x in z.infolist() if not x.is_dir() and pathlib.PurePosixPath(x.filename).suffix.lower()=='.csv' and '__MACOSX' not in x.filename],key=lambda x:x.filename)
  assert sum(x.file_size for x in members)<1024*1024*1024
  for member in members:
   b=z.read(member);reader=csv.DictReader(io.StringIO(b.decode('utf-8-sig')))
   fields={normkey(x) for x in (reader.fieldnames or [])}
   stem=pathlib.PurePosixPath(member.filename).stem;match=re.search(r'(\d+)$',stem)
   fid=int(match.group(1)) if match else None
   entry={'member':member.filename,'flight_id':fid,'source_sha256':sha(b)}
   reason=None;rows=[];ids=set()
   if not {'time','battery_voltage','battery_current'}<=fields:reason='missing documented battery/time columns'
   else:
    for x in reader:
     try:
      x={normkey(k):v for k,v in x.items() if k is not None}
      if x.get('flight') not in (None,''):ids.add(int(float(x['flight'])))
      t,v,c=(float(x[k]) for k in ('time','battery_voltage','battery_current'))
      if all(math.isfinite(q) for q in (t,v,c)) and v>0:rows.append((t,v,c))
     except (ValueError,TypeError,KeyError,OverflowError):continue
    if ids:
     if len(ids)!=1 or (fid is not None and fid not in ids):reason='inconsistent member flight identifier'
     else:fid=next(iter(ids));entry['flight_id']=fid
    if reason is None and fid not in records:reason='flight identifier not in source metadata'
    if reason is None and fid in seen:reason='duplicate flight identifier'
    if reason is None and len(rows)<2:reason='insufficient numeric measurements'
   if reason is None:
    seen.add(fid);a=np.asarray(rows,np.float64);a=a[np.argsort(a[:,0],kind='stable')]
    assert a[-1,0]-a[0,0]<=3600,'unexpected time units/record duration'
    ticks=np.arange(math.ceil(a[0,0]),math.floor(a[-1,0])+1)
    idx=np.searchsorted(a[:,0],ticks,side='right')-1
    ok=(idx>=0)&(ticks-a[np.maximum(idx,0),0]<=.4)
    valid=np.flatnonzero(ok);runs=np.split(valid,np.flatnonzero(np.diff(valid)!=1)+1);longest=max(runs,key=len)
    if len(longest)<96:reason='longest causal 1Hz segment shorter than 96s'
    else:
     values=a[idx[longest],1:].astype('<f4');digest=sha(values.tobytes())
     if digest in digests:reason='duplicate derived battery values'
     else:
      key=f'cmu-{fid:04d}';assert key not in arr;digests.add(digest);arr[key]=values
      entry.update({'key':key,'date':records[fid]['date'],'seconds':len(values),'start_s':int(ticks[longest[0]]),'end_s':int(ticks[longest[-1]]),'array_sha256':digest})
      accepted.append(entry)
   if reason:rejected.append(entry|{'reason':reason})
 for fid in sorted(set(records)-{x['flight_id'] for x in accepted+rejected}):
  rejected.append({'flight_id':fid,'reason':'metadata flight without matched CSV member'})
 manifest={'source':receipt,'license_attribution':'Rodrigues et al., Data Collected with Package Delivery Quadcopter Drone, Carnegie Mellon University, DOI 10.1184/R1/12683453.v1, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Modified: battery voltage/current only, causal 1 Hz extraction, no route/position fields.','admission_rule':'longest contiguous causal 1 Hz segment, sample at or before tick within 0.4s; finite voltage >0 and finite current; minimum 96 seconds; no target-based error filtering','metadata_flights':len(records),'source_csv_members':len(members),'admitted':accepted,'rejected':rejected,'physical_pack_identity':'unavailable; date groups are not independent airframes or physical battery identities'}
 write(out/'manifest.json',manifest)
 assert len(arr)>=100,('insufficient admitted independent-dataset flights',len(arr))
 np.savez_compressed(out/'battery-only.npz',**{k:arr[k] for k in sorted(arr)})
 print('admitted',len(arr),'seconds',sum(len(x) for x in arr.values()),'days',len({x['date'] for x in accepted if x['date']}),'rejections',len(rejected),flush=True)
if __name__=='__main__':main()
