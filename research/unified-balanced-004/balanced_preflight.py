"""Engineering-only full-batch shared-model update/replay for balanced knowledge."""
import argparse,hashlib,json,tempfile,math
from pathlib import Path
import torch
from space import WholeCOG, BASE_ARMS
import train as t
from balanced import draw8_balanced, independent_answer
OUT=Path(__file__).resolve().parent
ANCHORS=('NN_native_native','SS_presence_factor','BB_presence_factor','GQA')

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--start',type=int,default=0);p.add_argument('--end',type=int,default=4);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);t.verify(a.data);lock=t.source();t.draw8=draw8_balanced;seed=t.SEEDS[0];rows=[]
    for arm in ANCHORS[a.start:a.end]:
        model=WholeCOG(arm,seed);opt=t.optimizer(model);raw,starts,g4,g8,s4,s8,regime=t.init_batches(a.data,seed,'random_full')
        data1=t.batch(raw,starts,0,g4,g8,s4,s8,regime);data2=t.batch(raw,starts,1,g4,g8,s4,s8,regime)
        assert all(torch.equal(independent_answer(data[3]),data[4]) for data in (data1,data2))
        first=t.update(model,opt,data1,0)
        with tempfile.TemporaryDirectory() as td:
            ckpt=Path(td)/'step1.pt';torch.save(dict(weights=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),ckpt)
            cp=torch.load(ckpt,map_location='cpu',weights_only=True)
            second=t.update(model,opt,data2,1)
            replay=WholeCOG(arm,seed);ro=t.optimizer(replay);replay.load_state_dict(cp['weights']);ro.load_state_dict(cp['optimizer']);torch.set_rng_state(cp['rng']);replayed=t.update(replay,ro,data2,1)
            assert second==replayed and t.same(model.state_dict(),replay.state_dict()) and t.same(opt.state_dict(),ro.state_dict())
        assert all(torch.isfinite(v).all() for v in model.parameters())
        row=dict(arm=arm,seed=seed,parameters=t.count(model),first_update=first,second_update=second,exact_next_update=True,independent_answer_parity=True,real_targets_per_update=12288,cognitive_examples_per_update=16,scientific_fit=False)
        print(json.dumps(row),flush=True);rows.append(row)
    (OUT/f'balanced_preflight_{a.start}_{a.end}.json').write_text(json.dumps(dict(status='ENGINEERING_ONLY',scientific_fits=0,source=lock,rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
