"""COG-ATTN-097c strict source-first no-label retrieval preflight.

Retrieval only; no entity ID, slot inventory, query-to-document labels, witnessed
training examples, reference plan or supplied candidate set is visible to readers.
Author-written first summaries serve as held-out evaluation queries. Their
associated remainder-of-docstring passages form the raw memory. The evaluator
alone has the source linkage. These are real Python stdlib documentation, not
real dialogue, and this is NOT a complete COG controller evaluation.
"""
from __future__ import annotations
import argparse, ast, collections, hashlib, importlib.util, json, math, os, random, re, statistics, sys, time
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MODULES = ('argparse','subprocess','collections','typing','http.client','email.message',
    'unittest.case','asyncio.base_events','urllib.request','logging','inspect',
    'pydoc','tarfile','zipfile','json.encoder','xml.etree.ElementTree','contextlib','tempfile')
SEED=42097
NEGATIVE_MODULES=('shutil','pickle','statistics','unittest.mock','json.decoder','socket','textwrap')
TOKEN = re.compile(r"(?u)\b[a-zA-Z_][a-zA-Z_0-9]+\b")
def tokens(s): return TOKEN.findall(s.lower())
def sha(text): return hashlib.sha256(text.encode()).hexdigest()

def make_corpus(module_names=MODULES):
    """Extract naturally authored docstring summaries, hiding their source linkage.
    No authored training question or structured field appears in the model's corpus.
    """
    data=[]; origins=[]
    for mod in module_names:
        spec=importlib.util.find_spec(mod)
        if spec is None or not spec.origin or not spec.origin.endswith('.py'):continue
        src=Path(spec.origin).read_text(encoding='utf8');tree=ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef)):continue
            doc=ast.get_docstring(node)
            if not doc:continue
            # Summary is a human-authored docstring opening. The rest is unlabelled memory.
            parts=doc.strip().split('\n\n',1)
            first=parts[0].strip()
            # A description line may wrap; prefer first full sentence or first line.
            first=first.split('\n',1)[0].strip()
            if len(first.split())<6 or len(first.split())>30:continue
            remainder=doc[len(first):].strip()
            if len(remainder.split())<28:continue
            if len(tokens(first))<5:continue
            body=' '.join(remainder.split())
            if len(body)<130:continue
            data.append(dict(query=first,body=body,source=mod,qualified=node.name))
            origins.append((mod,node.name))
    # Remove literal and near-trivial exact-repeat queries; prevents a duplicate
    # authored summary from creating an inherently ambiguous "gold" match.
    counts=collections.Counter(x['query'].casefold() for x in data)
    data=[x for x in data if counts[x['query'].casefold()]==1]
    bodies=set();dedup=[]
    for x in data:
        h=sha(x['body'])
        if h not in bodies:bodies.add(h);dedup.append(x)
    return dedup

class BM25:
    def __init__(self,docs):
        self.t0=time.perf_counter_ns();self.n=len(docs)
        self.length=[];self.post=collections.defaultdict(list)
        for i,s in enumerate(docs):
            tf=collections.Counter(tokens(s));self.length.append(sum(tf.values()))
            for w,count in tf.items():self.post[w].append((i,count))
        self.avg=sum(self.length)/max(self.n,1);self.build_ns=time.perf_counter_ns()-self.t0
    def rank(self,question):
        # Globally searched posting lists; no oracle candidate set or entity parser.
        values=collections.defaultdict(float);k1=1.5;b=.75
        for w in set(tokens(question)):
            hits=self.post.get(w,())
            if not hits:continue
            idf=math.log(1+(self.n-len(hits)+.5)/(len(hits)+.5))
            for i,count in hits:
                denom=count+k1*(1-b+b*self.length[i]/self.avg)
                values[i]+=idf*count*(k1+1)/denom
        if not values:return -1,0.
        return max(values,key=lambda i:(values[i],-i)),max(values.values())

class Tfidf:
    def __init__(self,docs,analyzer='word'):
        t0=time.perf_counter_ns();self.kind=analyzer
        self.vectorizer=TfidfVectorizer(lowercase=True,analyzer=analyzer,
            ngram_range=(1,2) if analyzer=='word' else (3,5),min_df=1,sublinear_tf=True)
        self.mat=self.vectorizer.fit_transform(docs)
        self.build_ns=time.perf_counter_ns()-t0
    def rank(self,question):
        q=self.vectorizer.transform([question]);scores=(q@self.mat.T).toarray()[0]
        j=int(scores.argmax());return j,float(scores[j])

class LSA:
    def __init__(self,docs,rank=64):
        t0=time.perf_counter_ns();self.vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
        mat=self.vectorizer.fit_transform(docs)
        self.svd=TruncatedSVD(n_components=min(rank, min(mat.shape)-1),random_state=SEED)
        self.emb=normalize(self.svd.fit_transform(mat));self.build_ns=time.perf_counter_ns()-t0
    def rank(self,question):
        q=normalize(self.svd.transform(self.vectorizer.transform([question])))
        scores=(q@self.emb.T)[0];j=int(scores.argmax());return j,float(scores[j])

def evaluate(articles=None,out=None):
    corpus_start=time.perf_counter_ns()
    articles=make_corpus() if articles is None else articles
    if len(articles)<100:raise RuntimeError('insufficient authentic passages')
    # Use all article bodies in the archive (visible to learner) and their associated
    # summaries ONLY to hidden evaluator. Test is on authored text, no templates.
    passages=[x['body'] for x in articles]
    negatives=make_corpus(NEGATIVE_MODULES)
    corpus_ms=(time.perf_counter_ns()-corpus_start)/1e6
    # Negative summaries are authored in entirely different source modules;
    # their corresponding documentation bodies are NOT indexed. Models never
    # receive the negative/positive flag, so top-1-only readers cannot defer.
    raw_digest=sha('\n'.join(passages))
    methods={}
    for name,constructor in [('bm25',BM25),('word_tfidf',lambda d:Tfidf(d,'word')),
                             ('char_tfidf',lambda d:Tfidf(d,'char')),
                             ('lsa',LSA)]:
        methods[name]=constructor(passages)
    def sample_queries():
        # all queries fixed ahead of scoring; not selected by model accuracy
        rng=random.Random(SEED);idx=list(range(len(articles)));rng.shuffle(idx)
        return idx
    order=sample_queries();pred=[]
    for i in order:
        q=articles[i]['query']
        row=dict(target=i,query_digest=sha(q),source=articles[i]['source'])
        for name,method in methods.items():
            start=time.perf_counter_ns();predicted,score=method.rank(q)
            row[name]=dict(idx=predicted,score=score,correct=predicted==i,
                           inference_ns=time.perf_counter_ns()-start)
        pred.append(row)
    aggregates={}
    for name,model in methods.items():
        times=[r[name]['inference_ns'] for r in pred]
        correct=sum(r[name]['correct'] for r in pred)
        aggregates[name]=dict(correct=correct,n=len(pred),accuracy=correct/len(pred),
            median_query_ms=round(statistics.median(times)/1e6,4),
            total_query_ms=round(sum(times)/1e6,3),build_ms=round(model.build_ns/1e6,3),
            total_ms=round((sum(times)+model.build_ns)/1e6,3))
    # Entirely disjoint author-written queries with no associated archived body.
    negative_rows=[]
    for item in negatives:
        row={'query_digest':sha(item['query']),'source':item['source']}
        for name,method in methods.items():
            t=time.perf_counter_ns();j,score=method.rank(item['query'])
            row[name]=dict(idx=j,score=score,inference_ns=time.perf_counter_ns()-t,
                           false_answer=j>=0)
        negative_rows.append(row)
    for name in aggregates:
        nr=[r[name] for r in negative_rows]
        aggregates[name]['total_with_corpus_ingest_ms']=round(aggregates[name]['total_ms']+corpus_ms,3)
        aggregates[name]['unanswerable_false_answers']=sum(r['false_answer'] for r in nr)
        aggregates[name]['unanswerable_n']=len(nr)
        aggregates[name]['unanswerable_query_ms']=round(sum(r['inference_ns'] for r in nr)/1e6,3)
    group={}
    for module in MODULES:
        test=[r for r in pred if r['source']==module]
        if len(test)>0:
            group[module]={a:sum(r[a]['correct'] for r in test) for a in methods}
            group[module]['n']=len(test)
    result=dict(status='real-authored-docstring zero-label raw-retrieval preflight, not full COG-ATTN',
        methodology=['All passages globally indexed; no supplied entity IDs, fields, query/evidence pairs or candidate lists',
                     'Queries are already-written docstring first lines, not human-written question answering queries',
                     'Evaluative source linkage is hidden from every retriever; no model trained on it',
                     'Queries and answers originate from same natural documentation object, a limited retrieval proxy',
                     'Negative controls are real author-written summaries from modules withheld entirely from the memory corpus',
                     'These retrieval-only methods have no principled unknown mechanism and return false matches for absent memories',
                     'No authentic dialogue, memory updates, reasoning, action learning, authorization, Transformer parity or semantic generalization qualification'],
        python=sys.version,seed=SEED,n=len(articles),n_unanswerable=len(negatives),corpus_sha256=raw_digest,
        corpus_extract_ms=round(corpus_ms,3),
        source_sha256={mod:hashlib.sha256(Path(importlib.util.find_spec(mod).origin).read_bytes()).hexdigest()
                       for mod in MODULES+NEGATIVE_MODULES},
        modules=dict(collections.Counter(x['source'] for x in articles)),
        results=aggregates,by_module=group,rows=pred,unanswerable=negative_rows)
    if out:Path(out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='097c-results.json')
    args=ap.parse_args();r=evaluate(out=args.out)
    print(json.dumps({k:r[k] for k in ('n','modules','corpus_sha256','results')},indent=2))
