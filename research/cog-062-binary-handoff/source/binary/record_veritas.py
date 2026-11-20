"""Native Veritas envelope for separate aggregate rescore, always NOT_QUALIFIED."""
import hashlib,json
from datetime import datetime,UTC
from investigation_world.evidence import EvidenceRecord,serialize_evidence_record
from trainlab.runner import verify_run
from research_lab.evidence import put_node,link

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(work,source,freeze,scored,pred,reviewer,c):
    if not all(verify_run(x)['valid'] for x in (scored,pred)):raise ValueError('invalid run receipt')
    result=scored/'output/scores.json'
    if json.loads(result.read_bytes())['qualification']!='NOT_QUALIFIED':
        raise ValueError('scientific promotion forbidden')
    artifacts=(('scores',result),('fit-primary',work/'fit-primary/RECEIPT.json'),('fit-replay',work/'fit-replay/RECEIPT.json'),('inference',work/'predict/RECEIPT.json'),('score-receipt',work/'score/RECEIPT.json'),('forge',source/'data/forge_receipt.json'))
    record=EvidenceRecord(evidence_type='research.reranker_diagnostic',outcome='OBSERVED',visibility='operator_private',claim='B-informed binary top-two reranker diagnostic on historically exposed MultiWOZ 2.2, NOT_QUALIFIED',
       subjects=[dict(kind='experiment',subject_id='cog062-binary',content_sha256=freeze)],
       producer=dict(producer_id='cog062-separate-scorer',producer_version='1',content_sha256=sha(source/'score.py')),
       policy=dict(policy_id='cog062-diagnostic',policy_version='1',content_sha256=sha(source/'PROTOCOL.md')),
       artifacts=[dict(artifact_id=n,role='score' if n=='scores' else 'audit',content_sha256=sha(p)) for n,p in artifacts],
       observed_at=datetime.now(UTC),provenance={'prescore_reviewer':reviewer,'scientific_qualification':'NOT_QUALIFIED'})
    data=serialize_evidence_record(record);EvidenceRecord.model_validate_json(data)
    dest=work/'VERITAS_EVIDENCE.json'
    with dest.open('xb') as f:f.write(data)
    put_node(c,'cog062-binary.veritas','evidence',{'evidence_id':record.evidence_id,'sha256':sha(dest),'qualification':'NOT_QUALIFIED'})
    link(c,'cog062-binary.score','supports','cog062-binary.veritas')
