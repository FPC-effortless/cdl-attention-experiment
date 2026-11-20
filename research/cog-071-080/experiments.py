"""Run a preregistered panel; keep timings and stable decisions separate."""
from collections import Counter
import hashlib,json,os,sys
from pathlib import Path
from check_boundary import run as boundary
from causal import run as causal
from tracker import run_panel
from plans import economic
from language import Lexicon,outcome_examples,ALIASES,Grounder,render,TARGETS

def shadow():
    model=Grounder(True);lex=Lexicon();ex=outcome_examples();lex.learn(ex)
    docs=[(render(alias,t,l),a) for a,alias in enumerate(ALIASES) for targets in TARGETS for t in targets for l in range(4)]
    p,_=model.score([d[0] for d in docs])
    return dict(held=len(docs),before=sum(int(r.argmax()==a) for r,(_,a) in zip(p,docs)),
                after=sum(int(lex.apply(text,r).argmax()==a) for r,(text,a) in zip(p,docs)),
                origins=lex.provenance,shared_or_conflicting_admitted=sorted(set(lex.bindings)&{'echo','disputed'}))
def main():
    bank=sys.argv[1];seeds=[7103,7109,7121] if bank=='primary' else [8011,8017,8039]
    out=Path(os.getenv('TRAINLAB_OUTPUT_DIR',str(Path(__file__).resolve().parent/'results'/bank)))
    out.mkdir(parents=True,exist_ok=True)
    cases=run_panel(seeds,bank)
    sums=dict(boundary=boundary(),causal={s:causal(s) for s in seeds},plans={s:economic(s) for s in seeds},shadow=shadow())
    def save(name,data):
        (out/name).write_text(json.dumps(data,separators=(',',':'),sort_keys=True)+'\n')
    save('sessions.json',cases);save('subsystems.json',sums)
    costless=[(a['seed'],a['arm'],a['units'],[{k:v for k,v in r.items() if k!='ns'} for r in a['rows']]) for a in cases]
    for report in sums['plans'].values():
        for row in report.values():del row['elapsed_ns']
    stable=hashlib.sha256(json.dumps([costless,sums],sort_keys=True).encode()).hexdigest()
    (out/'stable.sha256').write_text(stable+'\n')
    save('metrics.json',{'unauthorized':sum(r['bad_permission'] for a in cases for r in a['rows'])})
if __name__=='__main__':main()
