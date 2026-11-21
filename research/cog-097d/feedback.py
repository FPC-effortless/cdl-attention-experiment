"""COG-ATTN-097d2: train from outcome-only feedback on raw documentation.

The hidden evaluator can answer if a chosen passage resolves the observed
request, returning one bit. The learner is never passed the correct passage id.
The candidate proposer is a transparent char TF-IDF policy on visible text.
After feedback training, all model weights freeze before new modules are read.
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, random, statistics, time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from experiment import Encoder,TRAIN_MODULES,TEST_MODULES,DEV_MODULES,UNKNOWN_MODULES,make_corpus,rank_embedding,rank_tfidf,scoring,sha,embed

FRESH_MODULES=('smtplib','imaplib','difflib','cmd','queue','gettext','calendar',
    'ipaddress','ftplib','ntpath','posixpath','logging.handlers','pickletools',
    'fileinput','mailbox','xml.sax.handler','xml.sax.xmlreader','tokenize',
    'codeop','code','numbers','shelve','wsgiref.simple_server','wsgiref.handlers',
    'email.mime.multipart','email.mime.text','pprint','fnmatch','string','enum',
    'runpy','ast','stringprep','unittest.suite')

class World:
    def __init__(self,docs):
        self.docs=docs
        self.tries=0;self.successes=0
    def act(self,request_index,chosen_index):
        # Evaluator code is outside learnable modules, and returns ONLY a bit.
        self.tries+=1
        ok=bool(request_index==chosen_index)
        self.successes+=ok
        return ok

def collect_feedback(training):
    """Outcome-only behavioral acquisition; never emits a gold index."""
    t0=time.perf_counter_ns()
    visible=[r['body'] for r in training]
    reqs=[r['query'] for r in training]
    lexical=TfidfVectorizer(analyzer='char',ngram_range=(3,5),sublinear_tf=True)
    mat=lexical.fit_transform(visible)
    score=(lexical.transform(reqs)@mat.T).toarray()
    world=World(visible)
    observations=[]
    for request_idx,request in enumerate(reqs):
        # This top-3 comes from raw text similarity, not gold labels.
        choices=np.argsort(-score[request_idx],kind='stable')[:3]
        for proposed in choices:
            chosen=int(proposed)
            success=world.act(request_idx,chosen)
            observations.append((request,visible[chosen],success))
            if success:break
    return observations,dict(attempts=world.tries,successes=world.successes,
                             unresolved=len(training)-world.successes,
                             acquisition_queries=len(training),
                             policy_build_and_actions_ns=time.perf_counter_ns()-t0,
                             top3_proposals=sum(min(3,len(training)) for _ in training))

def train_feedback(net,events,seed,mode='true',epochs=8):
    # The learner is shown ONLY raw text pairs and returned 0/1 feedback.
    torch.manual_seed(seed+19)
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.001)
    rng=random.Random(seed+3000)
    labels=[int(r[2]) for r in events]
    if mode=='shuffled':rng.shuffle(labels)
    pairs=[(e[0],e[1],labels[i]) for i,e in enumerate(events)]
    curve=[];steps=0;t0=time.perf_counter_ns()
    for epoch in range(epochs):
        perm=list(range(len(pairs)));random.Random(seed*17+epoch).shuffle(perm)
        batch_loss=[]
        for j in range(0,len(perm),32):
            idx=perm[j:j+32]
            if len(idx)<4:continue
            q=net([pairs[i][0] for i in idx]);d=net([pairs[i][1] for i in idx])
            logits=(q*d).sum(-1)*8.0
            truth=torch.tensor([pairs[i][2] for i in idx],dtype=torch.float32)
            loss=F.binary_cross_entropy_with_logits(logits,truth)
            opt.zero_grad();loss.backward();opt.step()
            batch_loss.append(float(loss.detach()));steps+=1
        curve.append(round(statistics.mean(batch_loss),5))
    return net,dict(epochs=epochs,updates=steps,loss=curve,elapsed_ns=time.perf_counter_ns()-t0)

def fresh_score(seed,output_path,weights_dir=None):
    torch.set_num_threads(1)
    train=make_corpus(TRAIN_MODULES)
    fresh=make_corpus(FRESH_MODULES)
    training_bodies={sha(x['body']) for x in train}
    training_queries={sha(x['query'].casefold()) for x in train}
    fresh=[x for x in fresh if sha(x['body']) not in training_bodies and sha(x['query'].casefold()) not in training_queries]
    assert fresh and len(fresh)>100
    assert not set(FRESH_MODULES)&(set(TRAIN_MODULES)|set(TEST_MODULES)|set(DEV_MODULES)|set(UNKNOWN_MODULES))
    # Input to learner during training: raw observations, one-bit feedback.
    events,env_receipt=collect_feedback(train)
    base=Encoder()
    state=torch.load(Path(__file__).parent/f'model-{seed}.pt',map_location='cpu',weights_only=True)
    assert state['seed']==seed
    base.load_state_dict(state['state']);base.eval()
    before,base_b,base_q=rank_embedding(fresh,base)
    positive=Encoder();positive.load_state_dict(state['state'])
    positive,record=train_feedback(positive,events,seed,'true')
    negative=Encoder();negative.load_state_dict(state['state'])
    negative,shuf_record=train_feedback(negative,events,seed,'shuffled')
    if weights_dir:
        folder=Path(weights_dir);folder.mkdir(exist_ok=True,parents=True)
        torch.save(dict(state=positive.state_dict(),seed=seed,
            feedback_sha256=sha('\n'.join('%s|%s|%s'% (sha(q),sha(d),r) for q,d,r in events))),folder/f'feedback-{seed}.pt')
    learned,l_build,l_query=rank_embedding(fresh,positive)
    control,c_build,c_query=rank_embedding(fresh,negative)
    # Missing evidence control: all scoring arms currently choose an archive id
    # and expose no abstain action. This is a structural false-answer count.
    absent=make_corpus(UNKNOWN_MODULES)
    absent=[z for z in absent if sha(z['body']) not in {sha(x['body']) for x in fresh}]
    char,ch_build,ch_query=rank_tfidf(fresh,'char')
    word,w_build,w_query=rank_tfidf(fresh,'word')
    all_scores={'before_feedback':before,'after_feedback':learned,
                'shuffled_feedback':control,'char_tfidf':char,'word_tfidf':word}
    values={k:scoring(v) for k,v in all_scores.items()}
    for k in values:values[k]['unknown_false_answers']=len(absent)
    for k in values:values[k]['unknown_queries']=len(absent)
    res=dict(name='COG-ATTN-097d2 real-prose / simulated feedback, not full COG',
      seed=seed,counts=dict(train=len(train),fresh_test=len(fresh),feedback_events=len(events),unknown=len(absent)),
      feedback=env_receipt,optimization=record,shuffled_optimization=shuf_record,
      scores=values,
      timings=dict(before_feedback=dict(index_ns=base_b,query_ns=base_q),
        after_feedback=dict(index_ns=l_build,query_ns=l_query),
        shuffled_feedback=dict(index_ns=c_build,query_ns=c_query),
        char_tfidf=dict(index_ns=ch_build,query_ns=ch_query),
        word_tfidf=dict(index_ns=w_build,query_ns=w_query)),
      fresh_modules=FRESH_MODULES,
      source_hashes={x:hashlib.sha256(Path(importlib.util.find_spec(x).origin).read_bytes()).hexdigest()
                     for x in FRESH_MODULES if importlib.util.find_spec(x).origin and importlib.util.find_spec(x).origin.endswith('.py')},
      test_body_digest=sha('\n'.join(x['body'] for x in fresh)),
      feedback_trace=[dict(query=sha(q),document=sha(d),reward=int(r)) for q,d,r in events],
      limitations=['Binary reward is synthesized from docstring author-association; not feedback from real users',
        'Lexical policy is part of acquisition and must be counted; it does not observe gold IDs',
        'No full COG-ATTN controller or executable actions',
        'All models still always answer missing-evidence questions',
        'Frozen evaluation on fresh modules, without test-time gradient updates'])
    if output_path:Path(output_path).write_text(json.dumps(res,sort_keys=True,indent=2)+'\n')
    return res

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=1701)
    parser.add_argument('--out',required=True)
    parser.add_argument('--weights-dir',default=None)
    args=parser.parse_args()
    r=fresh_score(args.seed,args.out,args.weights_dir)
    print(json.dumps(dict(counts=r['counts'],feedback=r['feedback'],optimization=r['optimization'],
       results={k:{'correct':v['correct'],'n':v['n'],'accuracy':v['accuracy']} for k,v in r['scores'].items()}),indent=2))
