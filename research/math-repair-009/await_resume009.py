"""Wait for verified first-stage completion without restarting active scientific fits."""
import argparse, json, os, time, urllib.request, urllib.error
from pathlib import Path

NATIVE_RUN=46039004295
GPT_RUN=46041424800
SEEDS=(9001,9011,9029)
ARMS=('HH','A0','U','UA','AU')
ROOT='https://api.github.com/repos/FPC-effortless/cdl-attention-experiment/actions/runs/'
KEYS=('id','name','status','conclusion','completed_at')

def request(path):
    req=urllib.request.Request(ROOT+path,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.urlopen(req,timeout=45) as r:
        return json.load(r)

def collect(run,kind):
    result=[]
    for i in range(1,11):
        page=request(f'{run}/{kind}?per_page=100&page={i}')
        result.extend(page[kind])
        if len(result)>=page['total_count']:
            assert len({x['id'] for x in result})==len(result)==page['total_count']
            return result
    raise RuntimeError('Pagination incomplete')

def inspect(run,arms):
    expected={f'stage1 ({a}, {s})' for a in arms for s in SEEDS}
    jobs=collect(run,'jobs')
    selected=[j for j in jobs if j['name'] in expected]
    assert len(selected)==len({j['name'] for j in selected}), 'duplicate stage-one jobs'
    failed=[j for j in selected if j['status']=='completed' and j['conclusion']!='success']
    if failed:
        raise RuntimeError('Original first stage failed; review instead of restarting: '+json.dumps([{k:j.get(k) for k in KEYS} for j in failed]))
    good=[j for j in selected if j['status']=='completed' and j['conclusion']=='success']
    return jobs,good,len(expected)

def await_ready(out):
    assert os.environ['GITHUB_REPOSITORY']=='FPC-effortless/cdl-attention-experiment'
    deadline=time.monotonic()+5*3600
    while True:
        try:
            native,ng,nn=inspect(NATIVE_RUN,ARMS)
            gpt,gg,gn=inspect(GPT_RUN,('gpt',))
            old=request(str(GPT_RUN))
            print(json.dumps({'native_success':len(ng),'native_expected':nn,'gpt_success':len(gg),'gpt_expected':gn,'prior_recovery_status':old['status'],'prior_recovery_conclusion':old['conclusion']}),flush=True)
            if len(ng)==nn and len(gg)==gn and old['status']=='completed':
                later=[j for j in gpt if any(j['name'].startswith(f'stage{i} (') for i in (2,3,4))]
                if any(j['conclusion']=='success' for j in later):
                    raise RuntimeError('Prior recovery advanced successfully; preserve those stages and review before resuming')
                arts=collect(GPT_RUN,'artifacts')
                if any(a['name'].startswith(('stage2-','stage3-','stage4-')) for a in arts):
                    raise RuntimeError('Prior recovery retained later-stage artifacts; review and preserve them before resuming')
                assert old['conclusion'] in ('failure','cancelled','timed_out','action_required'),old['conclusion']
                receipt={'status':'ALL_ORIGINAL_STAGE1_SUCCEEDED_NO_LATER_STAGE_ARTIFACTS','native_run':NATIVE_RUN,'gpt_run':GPT_RUN,'native_jobs':[{k:j.get(k) for k in KEYS} for j in ng],'gpt_jobs':[{k:j.get(k) for k in KEYS} for j in gg],'prior_recovery_conclusion':old['conclusion'],'prior_later_jobs':[{k:j.get(k) for k in KEYS} for j in later]}
                p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(receipt,indent=2));return
            if old['status']=='completed' and len(gg)!=gn:
                raise RuntimeError('Prior GPT run ended without all stage-one jobs; review')
        except (urllib.error.URLError,TimeoutError) as e:
            print('Transient status-read error:',type(e).__name__,flush=True)
        if time.monotonic()>=deadline:
            raise TimeoutError('Prior runs not ready. Retry only this barrier after inspection; preserve training artifacts.')
        time.sleep(60)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();await_ready(a.out)
