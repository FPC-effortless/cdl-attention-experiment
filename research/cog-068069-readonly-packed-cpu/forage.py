from forge_data.core import canonical
from forge_data.diagnostics import diagnose
import hashlib
def emit_view(out,name,records,split,fields):
 raw={'schema':'forge.diagnostic-input.v1','feature_fields':list(fields),'group_fields':['source_group'],'rows':[{'id':x['id'],'split':split,'groups':{'source_group':x.get('group',x['id'])},'features':{k:x[k] for k in fields},**{}} for x in records]}
 p=out/(name+'-forge-source.json');p.write_bytes(canonical(raw));digest=hashlib.sha256(p.read_bytes()).hexdigest()
 (out/(name+'-forge-receipt.json')).write_bytes(canonical(diagnose(p,sha256=digest,max_rows=1000,max_findings=0)))
 # Worker rebinds required features against the Forge envelope; contextual
 # request metadata is pinned on the same immutable file closure.
 (out/(name+'.json')).write_bytes(canonical(records))
