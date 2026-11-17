"""Generate external labels; raw input is retained untouched; no model fits."""
import hashlib,json
from pathlib import Path
import torch
from canonical import convert
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1)
 raw_path=ROOT/'data.json';before=sha(raw_path);raw=json.loads(raw_path.read_text());out={};checks=0
 for split,panels in raw.items():
  out[split]={}
  for key,cases in panels.items():
   assert len({len(c['text']) for c in cases})==1
   cooked=convert(cases);out[split][key]=cooked
   for a,b in zip(cases,cooked):
    assert b['label_space']=='first-occurrence-interned-slots'
    for field in ('text','program','query_targets','allquery_targets','query_positions','final_query_positions','ingest_boundary','identity_symbols'):
     assert a[field]==b[field]
    assert b['logical_events']==a['events'] and b['logical_final_state']==a['final_state']
    for e,f in zip(a['events'],b['events']):
     for field in ('value','answer','op','pos','field_masks'):assert e[field]==f[field]
    # Independently permute final record rows and reference targets using final table.
    mapping=[b['canonical_table_final'][ord(sym)-65] for sym in a['identity_symbols']]
    assert sorted(mapping)==[0,1,2,3]
    expected=[None]*4
    for logical,(kind,value) in enumerate(a['final_state']):
     expected[mapping[logical]]=[kind,mapping[value] if kind==2 else value]
    assert expected==b['final_state']
    # Payload multiset preserved exactly under row permutation, including absences.
    assert sorted(v for k,v in a['final_state'] if k==1)==sorted(v for k,v in b['final_state'] if k==1)
    checks+=1
 paired=0
 for key,cases in out['dev'].items():
  if key.endswith('_renamed_EH'):continue
  renamed=out['dev'][key+'_renamed_EH']
  for a,b in zip(cases,renamed):
   assert a['events']==b['events'] and a['final_state']==b['final_state']
   assert a['query_targets']==b['query_targets']
   assert b['paired_original_case_id']==a['case_id']
   paired+=1
 assert sha(raw_path)==before
 dest=ROOT/'canonical_data.json';dest.write_text(json.dumps(out,separators=(',',':'))+'\n')
 counts=dict(label_space='first-occurrence-interned-slots',program_label_space='logical-identity-indices',canonical_table_final_schema='26 entries indexed by uppercase ASCII A-Z; -1 uninterned, otherwise slot0..3',cases_checked=checks,paired_renamings_checked=paired,raw_sha256=before,canonical_sha256=sha(dest),canonical_source_sha256=sha(ROOT/'canonical.py'),lexer_sha256=sha(ROOT/'lexer.py'),generator_sha256=sha(Path(__file__)),raw_unchanged=True,tests=['values and query targets unchanged','logical event/state copies exact','final record permutation and reference remapping independently recomputed','all original/renamed canonical events and states identical'],no_model_fits=True)
 (ROOT/'canonical_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
