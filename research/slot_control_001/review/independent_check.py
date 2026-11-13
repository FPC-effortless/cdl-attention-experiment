"""Fresh-session model/recipe check only: no fit and no stress scoring."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/workspace/scratch/a74a27d116f3/recovered/experiments/csm_address_002')
sys.path.insert(0,str(ROOT))
from model import SlotStore
sys.path.insert(0,str(SOURCE))
import model_adapter as adapter

def check(test,message):
    if not test:raise RuntimeError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    torch.set_num_threads(1);torch.manual_seed(912)
    m=SlotStore();m.eval();checks=[]
    check(sum(p.numel() for p in m.parameters())==18254,'parameter count');checks.append('parameter_count')
    # Explicit selected replacement preserves all other records exactly.
    with torch.no_grad():
        m.destination.weight.zero_();m.destination.bias.fill_(-100);m.destination.bias[3]=100
        m.value_edit.weight.zero_();m.value_edit.bias.copy_(torch.tensor([-100.,100.]))
    state=m.init_state(2,'cpu');state[0].uniform_(-.7,.7);old=state[0].clone()
    _,new=m.step(torch.tensor([2,5]),state)
    for k in [0,1,2,4,5,6,7]:check(torch.equal(old[:,k],new[0][:,k]),'collateral write')
    check(not torch.equal(old[:,3],new[0][:,3]),'selected slot failed to change')
    checks.append('selected_write_frame')
    # Native source has no labels or routing overrides in its forward signatures.
    import inspect
    check(set(inspect.signature(m.sequence).parameters)=={'ids','collect','all_outputs'},'sequence labels')
    check(set(inspect.signature(m.step).parameters)=={'ids','state','collect'},'step labels')
    checks.append('input_signature_no_semantic_metadata')
    # Reconstruct the historical objective independently on synthetic tensors.
    batchsize,length,slots=2,5,8
    logits=torch.randn(batchsize,19,requires_grad=True)
    capture={'destination':torch.randn(batchsize,length,slots,requires_grad=True),
             'value_edit':torch.randn(batchsize,length,2,requires_grad=True),
             'write_gate':torch.randn(batchsize,length,1,requires_grad=True)}
    batch={'answers':torch.tensor([11,17]),'dest':torch.tensor([[-100,0,-100,-100,1],[-100,2,-100,-100,3]]),
           'edit':torch.tensor([[0,1,0,0,0],[0,1,0,0,0]])}
    total,_=adapter.objective('supervised',logits,capture,batch)
    f=torch.nn.functional
    expected=f.cross_entropy(logits,batch['answers'])+f.cross_entropy(capture['destination'].reshape(-1,8),batch['dest'].reshape(-1))+f.cross_entropy(capture['value_edit'].reshape(-1,2),batch['edit'].reshape(-1))+.25*f.binary_cross_entropy_with_logits(capture['write_gate'].squeeze(-1)[batch['edit'].bool()],torch.ones(2))
    check(torch.equal(total,expected),'objective mismatch');checks.append('same_objective_weights_and_information')
    bank=json.loads((SOURCE/'data/panels.json').read_text())['train'];check(len(bank)==2048,'TRAIN size')
    schedules={}
    for seed in [101,202,303]:
        schedule=torch.randint(len(bank),(256,32),generator=torch.Generator().manual_seed(seed+900000))
        h=hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
        receipt=json.loads((SOURCE/'runs'/f'supervised_{seed}'/'complete.json').read_text())
        check(h==receipt['schedule_sha256'],'historical schedule mismatch')
        check(receipt['updates']==256 and receipt['answer_exposures']==8192,'historical exposure mismatch')
        schedules[str(seed)]=h
    checks.append('all_three_original_schedules_and_exposures')
    # Test uninstrumented CSM vs instrumented source on random tokens, no checkpoint.
    torch.manual_seed(17)
    original=adapter.RecordRewrite(adapter.Config(vocab_size=19)).eval()
    wrapped=adapter.InstrumentedRecord(adapter.Config(vocab_size=19)).eval();wrapped.load_state_dict(original.state_dict())
    ids=torch.randint(19,(2,11));state=original.init_state(2,'cpu');step_logits=[]
    with torch.no_grad():
        for token in ids.unbind(1):out,state=original.step(token,state);step_logits.append(out)
        reference,_=wrapped.sequence(ids,all_outputs=True)
    check(torch.equal(torch.stack(step_logits,1),reference),'uninstrumented original mismatch')
    checks.append('exact_original_uninstrumented_vs_instrumented_logits')
    frozen=json.loads((ROOT/'freeze.json').read_text())
    for name,h in frozen['files'].items():check(sha(ROOT/name)==h,'changed frozen source')
    return {'status':'PASS','checks':checks,'schedule_hashes':schedules,'freeze_sha256':sha(ROOT/'freeze.json'),
            'reviewed_files':{p.name:sha(p) for p in ROOT.iterdir() if p.suffix in {'.py','.md'}},
            'torch':torch.__version__,'fit_steps':0,'stress_scoring':False,
            'scope':'random/synthetic small model checks and public TRAIN schedule verification'}

if __name__=='__main__':print(json.dumps(run(),indent=2,sort_keys=True))
