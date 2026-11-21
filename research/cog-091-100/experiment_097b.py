"""COG-ATTN-097b: no supplied ID, raw episode semantic retrieval (synthetic diagnostic).

No execution authority and no pretrained language model. Learned query/document
vectors use contrastive *witnessed query-to-evidence pairs*, never gold relation
labels as model inputs. All evaluation entities are unseen; one field and one
paraphrase family are held out. Raw history is retained after corrections.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiment_097b.py
"""
from __future__ import annotations
import argparse, hashlib, json, random, re, statistics
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter_ns
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FIELDS = (
    ('meeting venue', ('where is the meeting held', 'what venue was selected'), ('which location hosts the gathering',)),
    ('shipment arrival window', ('when will the shipment arrive', 'what arrival time was set'), ('when should the package reach us',)),
    ('security badge color', ('what badge color was issued', 'which badge color was chosen'), ('what hue is on the access pass',)),
    ('invoice reference number', ('what is the invoice reference', 'which number is on the invoice'), ('give me the billing identifier',)),
    ('network access password', ('what network password was set', 'which access password was assigned'), ('tell me the wifi secret',)),
    ('emergency contact name', ('who is the emergency contact', 'which emergency contact was recorded'), ('who should we call in an emergency',)),
    ('lunar registry checksum', (), ('what is the lunar registry checksum',)),
)
SEEDS=(1103,1117,1129,2111,2129,2137)
SIZES=(1536,8192,32768)
VEC_DIM=1024


def digest(s): return hashlib.sha256(s.encode()).hexdigest()


def make_name(seed, i):
    # Unseen entity names, never an evaluator-provided structured entity identifier.
    return f'Nalo{seed} Kiren{i:05d}'


def make_archive(seed, size, entities=150):
    rng=random.Random(seed^0x197B)
    rows=[];names=[]
    for i in range(entities):
        name=make_name(seed,i);names.append(name)
        for field,(descriptor,_,_) in enumerate(FIELDS):
            version=1;value=f'v{rng.getrandbits(32):08x}'
            rows.append(dict(alias=name,field=field,raw=f'{name}: {descriptor} was recorded as {value}.',
                             payload=value,version=version,source=f'signed-simulated:{seed}:{i}:{field}:1'))
            # Corrections maintain the original raw episode and change the selected revision.
            if field==1 and i%5==0:
                v2=f'v{rng.getrandbits(32):08x}'
                rows.append(dict(alias=name,field=field,raw=f'{name}: correction to {descriptor}, now recorded as {v2}.',
                                 payload=v2,version=2,source=f'signed-simulated:{seed}:{i}:{field}:2'))
    while len(rows)<size:
        j=len(rows)
        name=f'Archive{seed}q{j:06d}'
        raw=f'{name}: archival observation about transit details {rng.getrandbits(32):08x}'
        rows.append(dict(alias=name,field=-1,raw=raw,payload='unused',version=1,source='simulated-distractor'))
    if len(rows)>size: raise ValueError('size must exceed target episode count')
    rng.shuffle(rows)
    return rows,names


class Archive:
    def __init__(self,rows):
        t=perf_counter_ns();self.rows=rows;self.by_name=defaultdict(list);self.latest={};self.latest_by_alias=defaultdict(dict)
        for i,r in enumerate(rows):
            self.by_name[r['alias']].append(i)
            key=(r['alias'],r['field'])
            if key not in self.latest or r['version']>rows[self.latest[key]]['version']:
                self.latest[key]=i
                self.latest_by_alias[r['alias']][r['field']]=i
        # Build alias matchers from the raw records, never from test labels.
        self.aliases=tuple(sorted(self.by_name,key=len,reverse=True))
        # Candidate entity resolution uses a cheap first-token posting list.
        self.parts=defaultdict(set)
        for alias in self.aliases:
            for token in alias.casefold().split():self.parts[token].add(alias)
        self.build_ns=perf_counter_ns()-t

    def resolve(self,conversation):
        words=set(re.findall(r'\b[a-z0-9]+\b',conversation.lower()))
        nonempty=[self.parts[w] for w in words if w in self.parts]
        small=[p for p in nonempty if len(p)<=8]
        pool=set().union(*(small if small else nonempty)) if nonempty else set()
        hits=[a for a in pool if re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',conversation,re.I)]
        return hits[0] if len(hits)==1 else None,len(hits)

    def candidates(self,conversation):
        name,n=self.resolve(conversation)
        if name is None:return None,(),n
        # Validate most recent versions without throwing away historical evidence.
        ids=tuple(self.latest_by_alias[name][field] for field in sorted(self.latest_by_alias[name]))
        return name,ids,n

    def direct_scan(self,alias,field):
        best=None;checks=0
        for r in self.rows:
            checks+=1
            if r['alias']==alias and r['field']==field and (best is None or r['version']>best['version']):best=r
        return best,checks


def query_string(name,question,style):
    if style=='ellipsis':return f'Earlier I was discussing {name}. Could you remind me: {question}?'
    if style=='direct':return f'Regarding {name}, {question}?'
    return f'In a previous message we mentioned {name}. I now need to know: {question}?'


def questions(seed,names,phase):
    rng=random.Random(seed ^ (0xAB if phase=='train' else 0xCD if phase=='dev' else 0xEF));out=[]
    for i,name in enumerate(names):
        field=i%6
        desc,train,held=FIELDS[field]
        phrase=(held[0] if phase=='test' and i%3==0 else rng.choice(train))
        style=['direct','ellipsis','indirect'][i%3]
        out.append(dict(text=query_string(name,phrase,style),name=name,field=field,
                        kind='held_paraphrase' if phase=='test' and i%3==0 else 'familiar',
                        question=phrase))
        if phase=='test' and i%4==0:
            out.append(dict(text=query_string(name,FIELDS[6][2][0],'direct'),name=name,field=6,kind='unseen_field',
                            question=FIELDS[6][2][0]))
    if phase=='test':
        for i in range(0,min(len(names)-1,60),6):
            name=names[i];name2=names[i+1]
            out.append(dict(text=f'We discussed {name} and {name2}. {FIELDS[0][1][0]}?',name=None,
                            field=None,kind='ambiguous',question=FIELDS[0][1][0]))
        for i in range(8):
            out.append(dict(text=f'In our earlier conversation with Unknown{i} Nobody, {FIELDS[1][1][0]}?',name=None,
                            field=None,kind='missing',question=FIELDS[1][1][0]))
    rng.shuffle(out);return out


class Dual(nn.Module):
    def __init__(self,dim=VEC_DIM,hidden=32):
        super().__init__()
        self.q=nn.Linear(dim,hidden,bias=False)
        self.d=nn.Linear(dim,hidden,bias=False)
    def forward(self,q,docs):
        a=nn.functional.normalize(self.q(q),dim=-1)
        b=nn.functional.normalize(self.d(docs),dim=-1)
        return (a[:,None,:]*b).sum(-1)*8


def fit_dual(seed,train_rows,dev_rows,archive,epochs=25):
    # Hashing encoder is static; gradients update only learned dual projections.
    hv=HashingVectorizer(n_features=VEC_DIM,alternate_sign=False,analyzer='char',ngram_range=(3,5),norm='l2')
    def encode(cases):
        q=np.asarray(hv.transform([c['text'] for c in cases]).toarray(),dtype='float32')
        docs=[];targets=[]
        for c in cases:
            idx=[archive.latest[(c['name'],f)] for f in range(6)]
            docs.extend([archive.rows[k]['raw'] for k in idx]);targets.append(c['field'])
        d=np.asarray(hv.transform(docs).toarray(),dtype='float32').reshape(len(cases),6,-1)
        return torch.from_numpy(q),torch.from_numpy(d),torch.tensor(targets)
    q,d,y=encode(train_rows);vq,vd,vy=encode(dev_rows)
    torch.manual_seed(seed)
    net=Dual();opt=torch.optim.AdamW(net.parameters(),lr=0.013,weight_decay=0.0001)
    best=-1;keep=None;t=perf_counter_ns()
    for epoch in range(epochs):
        net.train();order=torch.randperm(len(y))
        for part in order.split(48):
            loss=nn.functional.cross_entropy(net(q[part],d[part]),y[part]);opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():correct=int((net(vq,vd).argmax(1)==vy).sum())
        if correct>best:best=correct;keep=(epoch+1,{k:v.detach().clone() for k,v in net.state_dict().items()})
    net.load_state_dict(keep[1]);net.eval()
    return net,hv,dict(train=len(train_rows),dev=len(dev_rows),dev_correct=best,
                       dev_accuracy=best/len(dev_rows),chosen_epoch=keep[0],train_ms=(perf_counter_ns()-t)/1e6,
                       parameters=sum(p.numel() for p in net.parameters()))


def predict_dual(model,hv,text,docs):
    q=torch.from_numpy(np.asarray(hv.transform([text]).toarray(),dtype='float32'))
    d=torch.from_numpy(np.asarray(hv.transform(docs).toarray(),dtype='float32'))
    with torch.inference_mode():return model(q,d[None,:,:])[0].numpy()


def lexical_scores(text,docs,mode):
    # Deliberately strong within-entity word and char TF-IDF query-specific controls.
    analyzer='word' if mode=='word' else 'char'
    ng=(1,2) if mode=='word' else (3,5)
    vec=TfidfVectorizer(analyzer=analyzer,ngram_range=ng,lowercase=True)
    d=vec.fit_transform(docs+[text]);return cosine_similarity(d[-1],d[:-1])[0]



def relation_question(raw,name):
    # Question segmentation is supplied as a task-specific parser, not learned.
    # Strip the name in the surrounding conversation and retain the final clause.
    x=raw.replace(name,'')
    if ':' in x:x=x.rsplit(':',1)[-1]
    elif ',' in x:x=x.rsplit(',',1)[-1]
    return re.sub(r'[^a-z0-9 ]',' ',x.casefold()).strip()


class WitnessReader:
    """Read-only semantic supports: a verified (utterance, raw evidence) pair.

    Uses no stored relation ID. Transfer is from a support utterance to an
    unseen entity via raw evidence text similarity, not schema lookup.
    """
    def __init__(self,rows,supports,shuffle=False):
        self.supports=list(supports)
        if shuffle:
            rng=random.Random(9917)
            evidence=[x[1] for x in self.supports]
            evidence=evidence[2:]+evidence[:2]
            self.supports=[(q,d) for (q,_),d in zip(self.supports,evidence)]
        self.questions=[q for q,_ in self.supports]
        self.target_docs=[d for _,d in self.supports]
        self.vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5))
        self.cached=self.vec.fit_transform(self.questions)

    def propose(self,text,name,docs):
        q=relation_question(text,name)
        sims=cosine_similarity(self.vec.transform([q]),self.cached)[0]
        j=int(sims.argmax());confidence=float(sims[j])
        # Explicit support phrase gate: no fuzzy generalization promotion.
        if confidence<0.84:return None,confidence
        ev=self.target_docs[j]
        b=TfidfVectorizer(analyzer='word',ngram_range=(1,2))
        data=b.fit_transform(docs+[ev])
        ranks=cosine_similarity(data[-1],data[:-1])[0]
        return int(ranks.argmax()),confidence


def build_witnesses(names,index):
    pairs=[]
    for field in range(7):
        for name in names[:2]:
            phrase=FIELDS[field][2][0]
            evidence=index.rows[index.latest[(name,field)]]['raw']
            pairs.append((relation_question(query_string(name,phrase,'direct'),name),evidence))
    return pairs


def overlap_scores(text,docs):
    # Zero-fit, low-cost token-overlap control; does not learn semantic equivalence.
    terms=set(re.findall('[a-z]+',text.lower())) - {'what','which','the','was','is','for','regarding','about','need','know','our','earlier','message','could','you','in','a','we','and'}
    return np.array([len(terms & (set(re.findall('[a-z]+',d.lower())))) for d in docs],dtype=float)

def run_seed(seed,size=8192,train_entities=108,dev_entities=36,test_entities=90,epochs=25):
    torch.set_num_threads(1)
    # All observed entities exist in a single archive. Train and test entity IDs
    # are disjoint; only train/dev questions and witnessed episode links teach.
    count=train_entities+dev_entities+test_entities
    rows,names=make_archive(seed,size,count)
    index=Archive(rows)
    tr=questions(seed,names[:train_entities],'train')
    dv=questions(seed,names[train_entities:train_entities+dev_entities],'dev')
    test=questions(seed,names[train_entities+dev_entities:],'test')
    assert set(x['name'] for x in tr).isdisjoint(x['name'] for x in test if x['name'])
    model,hv,train=fit_dual(seed,tr,dv,index,epochs=epochs)
    modes=('overlap','word','char','frozen','learned','hybrid','witness','shuffled_witness','oracle')
    torch.manual_seed(seed)
    frozen=Dual().eval()
    witnesses=build_witnesses(names[:train_entities],index)
    acquired=WitnessReader(rows,witnesses);sham=WitnessReader(rows,witnesses,shuffle=True)
    summary={m:defaultdict(int) for m in modes};times={m:[] for m in modes};resolve_us=[]
    trace=[]
    t0=perf_counter_ns()
    for case in test:
        start_resolve=perf_counter_ns();name,ids,matched=index.candidates(case['text']);resolve_us.append((perf_counter_ns()-start_resolve)/1000)
        is_unknown=case['name'] is None
        if is_unknown:
            assert not ids,case
            for m in modes:summary[m]['safe_defer']+=1
            trace.append(dict(kind=case['kind'],gold=None,match_count=matched,safe_defer=True))
            continue
        assert name==case['name'] and len(ids)==7,(case,matched,ids)
        docs=[rows[k]['raw'] for k in ids]
        gold=case['field'];gidx=next(i for i,k in enumerate(ids) if rows[k]['field']==gold)
        rankings={};
        for m in modes:
            tick=perf_counter_ns()
            if m=='oracle':
                scores=np.zeros(len(ids));scores[gidx]=1
            elif m in ('word','char'):
                scores=lexical_scores(case['text'],docs,m)
            elif m=='overlap':
                scores=overlap_scores(case['text'],docs)
            elif m=='frozen':
                scores=predict_dual(frozen,hv,case['text'],docs)
            elif m in ('witness','shuffled_witness'):
                rank,score=(acquired if m=='witness' else sham).propose(case['text'],name,docs)
                scores=lexical_scores(case['text'],docs,'word')
                if rank is not None:
                    scores=np.zeros(len(ids));scores[rank]=1
            elif m=='learned':
                scores=predict_dual(model,hv,case['text'],docs)
            else:
                a=predict_dual(model,hv,case['text'],docs);b=lexical_scores(case['text'],docs,'word')
                # Fixed combination, not tuned against held test.
                scores=a+3*b
            order=np.argsort(-scores,kind='stable')
            times[m].append((perf_counter_ns()-tick)/1000)
            rankings[m]=order
            summary[m]['queries']+=1
            summary[m]['top1']+=int(order[0]==gidx)
            for k in (2,4,7):summary[m]['top'+str(k)]+=int(gidx in order[:k])
            summary[m]['candidate_checks']+=len(ids)
            summary[m]['latest_version']+=int(ids[order[0]]==index.latest[(name,rows[ids[order[0]]]['field'])])
        trace.append(dict(kind=case['kind'],text_hash=digest(case['text']),gold=gold,
                          predictions={m:int(rows[ids[rankings[m][0]]]['field']) for m in modes},
                          candidate_count=len(ids),source_revision=int(rows[ids[gidx]]['version'])))
    elapsed_ms=(perf_counter_ns()-t0)/1e6
    # A full raw linear scan is an actual reader control, but its query uses the
    # evaluation-resolved entity and field: this is an oracle scan, not an NLP baseline.
    scan_us=[];scan_checks=[];rng=random.Random(seed^512)
    for c in rng.sample([x for x in test if x['name'] is not None],min(24,len(test))):
        t=perf_counter_ns();r,checks=index.direct_scan(c['name'],c['field'])
        scan_us.append((perf_counter_ns()-t)/1000);scan_checks.append(checks)
        assert r and r['payload']==rows[index.latest[(c['name'],c['field'])]]['payload']
    for m in modes:
        summary[m]=dict(summary[m]);summary[m]['latency_us_median']=round(statistics.median(times[m]),2)
        summary[m]['approx_pipeline_us_median']=round(statistics.median(times[m])+statistics.median(resolve_us),2)
    by_kind={}
    for kind in ('familiar','held_paraphrase','unseen_field'):
        xs=[x for x in trace if x['kind']==kind]
        by_kind[kind]=dict(n=len(xs),**{m:sum(x['predictions'][m]==x['gold'] for x in xs) for m in modes})
    return dict(seed=seed,history=size,training=train,witness_pairs=len(witnesses),entity_count=count,raw_records=size,
                build_ms=round(index.build_ns/1e6,3), resolve_us_median=round(statistics.median(resolve_us),2),query_elapsed_ms=round(elapsed_ms,3),
                reader=summary, by_kind=by_kind,unresolved=sum(x['gold'] is None for x in trace),
                linear_scan_us_median=round(statistics.median(scan_us),2),
                linear_scan_checks_median=statistics.median(scan_checks),
                traced=trace)


def run(seeds=SEEDS,size=8192,epochs=25):
    res=[run_seed(s,size,epochs=epochs) for s in seeds]
    for r in res:print(r['seed'], 'retrieval', {m:(v['top1'],v['queries']) for m,v in r['reader'].items()},
                       'ambiguous+unmatched',r['unresolved'],flush=True)
    agg={}
    for m in ('overlap','word','char','frozen','learned','hybrid','witness','shuffled_witness','oracle'):
        agg[m]={k:sum(r['reader'][m][k] for r in res) for k in ('top1','top2','top4','top7','queries','safe_defer','candidate_checks')}
        agg[m]['median_query_us']=round(statistics.median(r['reader'][m]['latency_us_median'] for r in res),2)
    kinds={k:{m:sum(r['by_kind'][k][m] for r in res) for m in agg} | {'n':sum(r['by_kind'][k]['n'] for r in res)}
           for k in ('familiar','held_paraphrase','unseen_field')}
    return dict(protocol='COG-ATTN-097b synthetic no-supplied-entity retrieval pilot',seeds=list(seeds),
                history=size,aggregate=agg,by_kind=kinds,results=res,
                restrictions=['Artificial templated query language; model has no pretrained semantics',
                              'Resolver uses explicit or contextual full names; open-world coreference is not solved',
                              'Train labels are witnessed query-to-evidence links, synthetic rather than live user feedback',
                              '14 additional held-phrase witness links (two per relation) are introduced before test for the witness and sham arms; these arms are few-shot not zero-shot',
                              'Witness reader includes engineered query clause extraction and an exact phrase confidence gate',
                              'Novel relation exists in every entity archive but was absent from training queries',
                              'All candidate pools have exactly seven fields; cannot infer new latent concepts',
                              'No production ledger, authorization, external actions or actual MultiWOZ evaluation',
                              'Timed methods include feature extraction but not setup/training or entity resolution',
                              'Oracle receives evaluation field and name and is an upper bound not a language model'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='097b-results.json');p.add_argument('--seeds',default=','.join(map(str,SEEDS)))
    p.add_argument('--size',type=int,default=8192);p.add_argument('--epochs',type=int,default=25)
    a=p.parse_args();result=run(tuple(map(int,a.seeds.split(','))),a.size,a.epochs)
    with open(a.out,'w') as f:json.dump(result,f,indent=2,sort_keys=True)
