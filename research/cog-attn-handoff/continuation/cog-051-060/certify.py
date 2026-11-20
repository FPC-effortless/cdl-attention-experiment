"""Bind the final replay and study sources to a Veritas shared-evidence record."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from investigation_world.evidence.models import (EvidenceRecord,EvidenceArtifactRef,EvidenceProducerRef,EvidenceSubjectRef,EvidenceOutcome,EvidenceVisibility)
root=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    inputs=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('SHA256SUMS','veritas-evidence.json')]
    manifest=root/'SHA256SUMS'
    manifest.write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in sorted(inputs)))
    record=EvidenceRecord(evidence_type='runtime.grounded_acquisition',outcome=EvidenceOutcome.OBSERVED,
        visibility=EvidenceVisibility.PUBLIC,claim='Bounded acquisition experiments and held authentic diagnostics with open qualification gates.',
        subjects=(EvidenceSubjectRef(kind='protocol',subject_id='cog-051-060',content_sha256=sha(root/'PROTOCOL.md')),),
        producer=EvidenceProducerRef(producer_id='grounded-cycle-audit',producer_version='1',content_sha256=sha(root/'audit.py')),
        artifacts=(EvidenceArtifactRef(artifact_id='study-manifest',content_sha256=sha(manifest)),),
        observed_at=datetime.now(timezone.utc),provenance={'replay':'same-host TrainLab and separate GitHub-hosted scoped replay','review':'single research operator'})
    (root/'veritas-evidence.json').write_text(record.model_dump_json(indent=2))
if __name__=='__main__':main()
