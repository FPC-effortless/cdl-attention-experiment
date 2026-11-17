"""Independent literal surface-to-role checker for this constructed grammar.
Not a learned parser and never a model inference component. Deliberately does
not import generator templates; detects swapped roles or wrong rendered values.
"""
import re
N=r'[A-Za-z]+';V=r'[A-Z2-9]{3}'
PATTERNS={
 'set':[
  rf'The current code for (?P<dst>{N}) is (?P<value>{V})\.',
  rf'Update (?P<dst>{N}): the code is now (?P<value>{V})\.',
  rf'(?P<dst>{N}) now uses code (?P<value>{V})\.',
  rf"Replace (?P<dst>{N})'s code with (?P<value>{V})\.",
  rf'For (?P<dst>{N}), the newly assigned code is (?P<value>{V})\.'],
 'copy':[
  rf"Save a snapshot of (?P<src>{N})'s code for (?P<dst>{N})\.",
  rf"Give (?P<dst>{N}) a copy of (?P<src>{N})'s current code\.",
  rf"Record (?P<src>{N})'s code as an independent snapshot for (?P<dst>{N})\.",
  rf'Make (?P<dst>{N}) retain the value (?P<src>{N}) has at this moment\.'],
 'link':[
  rf"Keep (?P<dst>{N})'s code linked to (?P<src>{N})\.",
  rf"Let (?P<dst>{N}) always use (?P<src>{N})'s current code\.",
  rf"Have (?P<dst>{N}) follow any changes to (?P<src>{N})'s code\.",
  rf"The code of (?P<dst>{N}) should mirror (?P<src>{N})'s live code\."],
 'delete':[
  rf'Remove the code for (?P<dst>{N})\.',rf'(?P<dst>{N}) no longer has a code\.',
  rf"Erase (?P<dst>{N})'s assigned code\.",rf'Clear the code belonging to (?P<dst>{N})\.'],
 'query':[
  rf'What code does (?P<dst>{N}) currently have\?',rf'Report the present code for (?P<dst>{N})\.',
  rf'Which code is assigned to (?P<dst>{N}) now\?',rf"State (?P<dst>{N})'s code as it currently stands\."]}
def decode(text,op):
 matches=[m.groupdict() for p in PATTERNS[op] if (m:=re.fullmatch(p,text.strip()))]
 assert len(matches)==1,(op,text,matches)
 return matches[0]
def audit_surface(record):
 for op in record['operations']:
  d=decode(record['text'][op['start']:op['end']],op['op'])
  for k,v in d.items():assert v==op[k],(k,v,op[k])
 for a in record['answers']:
  q=record['text'][a['question_start']:a['start']];assert q.endswith('\nAnswer: ')
  assert decode(q[:-len('\nAnswer: ')],'query')['dst']==a['identity']
