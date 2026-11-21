"""Score authentic corrections in a second, previously unused MultiWOZ test shard."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer
from runtime import Runtime


ATTR={'area':'area','type':'type','booktime':'reservation time','pricerange':'price range',
      'bookday':'booking day','bookpeople':'number of people','bookstay':'length of stay',
      'food':'cuisine','leaveat':'departure time','arriveby':'arrival time','destination':'destination',
      'stars':'star rating','day':'day','name':'name','departure':'departure location'}


def mentioned(value, utterance):
    return list(re.finditer(r'(?<!\w)'+re.escape(value)+r'(?!\w)',utterance,re.I))


def cases(path):
    allowed = {
        '89af95d8f596a448e733d59b31be78f1dd1632eddd99d5cb298a3fcb1ac9d185',  # 002 test panel
        'e7ddb563e4da5766ea820cc826dead77e7ca219c19b761e218d62d9c999a252e',  # 003 dev transfer panel
    }
    if hashlib.sha256(path.read_bytes()).hexdigest() not in allowed:
        raise ValueError('unexpected dialogue source')
    for d in json.loads(path.read_text()):
        prev, transcript, user_words = {}, [], []
        for turn in d['turns']:
            new_line=turn['speaker'].lower()+': '+turn['utterance']
            if turn['speaker']=='USER':
                for frame in turn['frames']:
                    for k,values in frame.get('state',{}).get('slot_values',{}).items():
                        old=prev.get(k)
                        prev[k]=values[0] if len(values)==1 else None
                        new=prev[k]
                        if not old or not new or old==new or 'dontcare' in (old,new):
                            continue
                        if not mentioned(old,' '.join(user_words)) or not mentioned(new,turn['utterance']):
                            continue
                        service,field=k.split('-',1)
                        q='\nWhat '+ATTR[field]+' does the user currently want for the '+service+'? The answer is'
                        record=dict(dialogue=d['dialogue_id'],turn=turn['turn_id'],slot=k,old=old,new=new)
                        yield record,'\n'.join(transcript)+q,'\n'.join([*transcript,new_line])+q
                user_words.append(turn['utterance'])
            transcript.append(new_line)


@torch.inference_mode()
def evaluate(a):
    model=Runtime.load(a.base,a.adapter,a.disabled)
    tokenizer=Tokenizer.from_file(str(a.data/'tokenizer.json'))
    receipts=[]
    tasks=[]
    for info,before,after in cases(a.dialogue):
        idx=len(receipts)
        receipts.append(info)
        for condition,prompt in (('before',before),('after',after)):
            context=tokenizer.encode(prompt).ids
            raw=prompt.split('\nWhat ')[0]
            latest={key:max((m.start() for m in mentioned(info[key],raw)),default=-1) for key in ('new','old')}
            info[condition+'_literal_prefers_new']=latest['new']>latest['old']
            for answer in ('new','old'):
                ids=tokenizer.encode(prompt+' '+info[answer]+'.').ids
                assert ids[:len(context)]==context
                tasks.append((idx,condition,answer,len(context),ids))
    tasks.sort(key=lambda t:len(t[-1]))
    for start in range(0,len(tasks),4):
        batch=tasks[start:start+4]
        width=16*((max(len(t[-1])-1 for t in batch)+15)//16)
        x=torch.zeros(len(batch),width,dtype=torch.long)
        for j,(_,_,_,_,ids) in enumerate(batch):
            x[j,:len(ids)-1]=torch.tensor(ids[:-1])
        p,_=model.advance(x)
        for j,(idx,condition,answer,n,ids) in enumerate(batch):
            pred=p[j,n-1:len(ids)-1]
            labels=torch.tensor(ids[n:])
            record=receipts[idx]
            record[f'{condition}_{answer}_logp']=float(pred.gather(-1,labels[:,None]).sum())
            record[f'{condition}_{answer}_exact']=bool((pred.argmax(-1)==labels).all())
    name=a.adapter.stem+('_off' if a.disabled else '')
    (a.out/(name+'.dialogue.json')).write_text(json.dumps(receipts)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('base','adapter','data','dialogue','out'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--disabled',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(2)
    evaluate(a)
