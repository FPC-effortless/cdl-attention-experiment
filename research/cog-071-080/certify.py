"""Bind the public cycle corpus and its reproduction receipts to Veritas evidence."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from investigation_world.evidence.models import (EvidenceRecord,EvidenceArtifactRef,EvidenceProducerRef,
    EvidenceSubjectRef,EvidenceOutcome,EvidenceVisibility)

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.name not in ('SHA256SUMS','veritas-evidence.json'))
    manifest=ROOT/'SHA256SUMS'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in files))
    rec=EvidenceRecord(evidence_type='runtime.adaptive_commitment',outcome=EvidenceOutcome.OBSERVED,
        visibility=EvidenceVisibility.PUBLIC,
        claim='Ten bounded qualification attempts with distinct authority, semantic and economic outcomes.',
        subjects=(EvidenceSubjectRef(kind='protocol',subject_id='cog-071-080',content_sha256=sha(ROOT/'PROTOCOL.md')),),
        producer=EvidenceProducerRef(producer_id='graded-commitment-audit',producer_version='1',content_sha256=sha(ROOT/'audit.py')),
        artifacts=(EvidenceArtifactRef(artifact_id='cycle-manifest',content_sha256=sha(manifest)),),
        observed_at=datetime.now(timezone.utc),provenance={'lab':'Veritas Research Lab OS and TrainLab',
        'replay':'separate hosted CPU, shared source','scope':'single investigator, simulated authority'})
    (ROOT/'veritas-evidence.json').write_text(rec.model_dump_json(indent=2)+'\n')
if __name__=='__main__':main()
