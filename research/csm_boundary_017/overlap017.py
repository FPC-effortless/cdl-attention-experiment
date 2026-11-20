"""Text and structural overlap diagnostic; matching templates are disclosed, not independence."""
import json,hashlib

def signature(case):
 return tuple((o['op'],o['dst'],o['src'] if o['op'] in (2,3) else 0,o['value'] if o['op']==1 else 0) for o in case['program'] if not o.get('ignored'))
def audit(data,earlier):
 train=[c for cs in data['train'].values() for c in cs];dev=[c for cs in data['dev'].values() for c in cs]
 ts={signature(c) for c in train};tt={c['text'] for c in train};et={c['text'] for c in earlier};es={signature(c) for c in earlier if 'program' in c}
 return {'train_histories':len(train),'dev_histories':len(dev),'train_unique_semantic_programs':len(ts),'dev_text_matches_train':sum(c['text'] in tt for c in dev),'dev_program_matches_train':sum(signature(c) in ts for c in dev),'all_text_matches_earlier':sum(c['text'] in et for c in train+dev),'all_program_matches_earlier':sum(signature(c) in es for c in train+dev),'scope':'Program equality excludes ignored commands and formatting, retains identities/payloads. This is not semantic independence; missing016 overlap cannot be certified.'}
