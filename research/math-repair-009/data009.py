"""Pinned WikiText-103 raw corpus; never downloads or reads the official TEST split."""
import argparse,hashlib,json,re,urllib.request
from pathlib import Path
import numpy as np
REV='00aa25585682d4957f9e86edc73f59be7419af99'
BASE=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REV}/wikitext-103-raw-v1/'
FILES={'train':['train-00000-of-00002.parquet','train-00001-of-00002.parquet'],'validation':['validation-00000-of-00001.parquet']}
T=768;B=16;STEPS=8192;STAGE=2048;SEEDS=(9001,9011,9029)
HEADING=re.compile(r'^\s*=\s+([^=\s](?:.*?[^=\s])?)\s+=\s*$')
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical_title(x):return ' '.join(x.casefold().split())
def articles(rows):
    title=None;buf=[]
    for text in rows:
        if not isinstance(text,str):raise ValueError('nontext row')
        # Preserve source bytes and insert a line break only when absent.
        for line in (text if text.endswith('\n') else text+'\n').splitlines(keepends=True):
            m=HEADING.fullmatch(line.rstrip('\r\n'))
            if m:
                if buf and ''.join(buf).strip():yield title or '__preamble__',''.join(buf).encode('utf-8')
                title=canonical_title(m.group(1));buf=[]
            buf.append(line)
    if buf and ''.join(buf).strip():yield title or '__preamble__',''.join(buf).encode('utf-8')
def rows(files):
    import pyarrow.parquet as pq
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=8192,columns=['text']):
            yield from batch.column(0).to_pylist()
def fetch(out,split):
    out.mkdir(parents=True,exist_ok=True);sources=[]
    for name in FILES[split]:
        p=out/name
        if not p.is_file():
            tmp=p.with_suffix('.part');urllib.request.urlretrieve(BASE+name,tmp);tmp.replace(p)
        sources.append({'file':name,'bytes':p.stat().st_size,'sha256':sha(p)})
    return [out/x['file'] for x in sources],sources
def write_docs(docs,out,name):
    path=out/(name+'.bin');offsets=[0];titles=[];hashes=[];start=[]
    with path.open('wb') as f:
        for title,body in docs:
            start.extend(range(offsets[-1],offsets[-1]+len(body)-T,T));f.write(body);offsets.append(offsets[-1]+len(body));titles.append(title);hashes.append(hashlib.sha256(body).hexdigest())
    np.save(out/(name+'_offsets.npy'),np.asarray(offsets,dtype=np.int64));np.save(out/(name+'_starts.npy'),np.asarray(start,dtype=np.int64));(out/(name+'_documents.json')).write_text(json.dumps({'titles':titles,'hashes':hashes},ensure_ascii=False))
    return dict(documents=len(titles),bytes=offsets[-1],eligible_windows=len(start))
def make(split,out,train_manifest=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);paths,sources=fetch(out/'raw',split);docs=list(articles(rows(paths)));assert len(docs)>(100 if split=='train' else 20),(split,len(docs))
    all_count=len(docs);unique=[];seen=set()
    for title,body in docs:
        h=hashlib.sha256(body).hexdigest()
        if h not in seen:seen.add(h);unique.append((title,body))
    docs=unique;manifest={'revision':REV,'split':split,'sources':sources,'raw_documents':all_count,'duplicate_documents_removed':all_count-len(docs),'tokenizer':'UTF-8 bytes, identity ids 0..255','normalization':'none; source rows without newline get one newline','context':T,'batch':B,'steps':STEPS}
    if split=='train':
        train=[];dev=[]
        for title,body in docs:
            is_dev=int(hashlib.sha256(('repair009-dev-v1:'+title).encode()).hexdigest(),16)%100==0
            (dev if is_dev else train).append((title,body))
        assert not (set(x[0] for x in train)&set(x[0] for x in dev));assert len(dev)>20
        manifest['train']=write_docs(train,out,'train');manifest['dev']=write_docs(dev,out,'dev');assert manifest['train']['eligible_windows']>=B*STEPS
        assert manifest['train']['bytes']>=200_000_000,'wrong dataset or incomplete train split'
    else:
        assert train_manifest is not None
        a=json.loads((Path(train_manifest)/'train_documents.json').read_text());b=json.loads((Path(train_manifest)/'dev_documents.json').read_text());prior_titles=set(a['titles']+b['titles']);prior_hashes=set(a['hashes']+b['hashes']);overlap=[(t,hashlib.sha256(d).hexdigest()) for t,d in docs if t in prior_titles or hashlib.sha256(d).hexdigest() in prior_hashes]
        assert not overlap,('validation overlap; do not repair after seeing scores',overlap[:20]);manifest['validation']=write_docs(docs,out,'validation');assert manifest['validation']['bytes']>500_000
    manifest['files']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
def verify(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());assert m['revision']==REV and m['context']==T;assert all(sha(root/k)==v for k,v in m['files'].items());return m

def order(root,seed):
    starts=np.load(Path(root)/'train_starts.npy',allow_pickle=False);assert len(starts)>=B*STEPS;g=np.random.Generator(np.random.PCG64(seed+900090));return g.permutation(starts)[:B*STEPS].reshape(STEPS,B)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=list(FILES),required=True);p.add_argument('--out',required=True);p.add_argument('--train-manifest');a=p.parse_args();make(a.split,a.out,a.train_manifest)
