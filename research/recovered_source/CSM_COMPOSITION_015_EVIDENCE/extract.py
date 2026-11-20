"""Evidence extraction only; does not run counterfactuals or models."""
import json,hashlib
from pathlib import Path
from collections import Counter
B=Path(__file__).resolve().parent;ROOT=B.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 paths=sorted((ROOT/'csm-pointer-014/runs/20261116T095637-ffd1770b3d33/output/retained').glob('*.json'))
 assert len(paths)==15
 binding={};rows=[];endpoint_counts={};panels=Counter()
 for p in paths:
  d=json.loads(p.read_text());binding[str(p.relative_to(ROOT))]=sha(p);n=0
  for panel, result in d['dev'].items():
   if panel.endswith('_renamed_EH'):continue
   assert len(result['records'])==64
   for record in result['records']:
    assert not record['case_id'].endswith('/renamed_EH')
    rows.append(dict(arm=d['arm'],seed=d['seed'],panel=panel,record=record,source_record_file=str(p.relative_to(ROOT)),label_space='first-occurrence-interned-slots',identity_panel='original_A-D'))
    n+=1;panels[panel]+=1
  assert n==2048;endpoint_counts[f"{d['arm']}/{d['seed']}"]=n
 assert len(rows)==30720 and len(endpoint_counts)==15
 for rel in ['csm-pointer-014/INDEPENDENT_AUDIT.json','csm-pointer-014/source/canonical_counts.json','csm-pointer-014/source/canonical_data.json','csm-attribution-013/source/reference_executor.py']:
  p=ROOT/rel;binding[rel]=sha(p)
 (B/'records.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
 (B/'SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
 counts=dict(histories=len(rows),endpoints=15,histories_per_endpoint=endpoint_counts,panels=dict(panels),renamed_cases_excluded=30720,modes=9,counterfactual_histories_planned=len(rows)*9,no_neural_reexecution=True,no_fit=True,executed_diagnostic=False,records_sha256=sha(B/'records.json'),source_binding_sha256=sha(B/'SOURCE_BINDING.json'),field_count_definition='Field mismatches are counted only when opcode is correct; simultaneous source/destination mismatches may overlap. Wrong opcode, missing and false actions are separate. These counts do not sum to disjoint errors.')
 (B/'EXTRACTION_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps({k:counts[k] for k in ('histories','endpoints','executed_diagnostic','records_sha256')}))
if __name__=='__main__':main()
