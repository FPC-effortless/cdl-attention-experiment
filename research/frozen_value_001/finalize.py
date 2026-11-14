"""Recover reporting only after successful fixed inference; no model execution."""
from run import *
from research_lab.ecosystem import _runtime

def finalize():
    check()
    paths=list((ROOT/'runs').iterdir());require(len(paths)==1,'unexpected run count')
    path=paths[0];receipt=read_json(path/'run.json')
    require(receipt['status']=='succeeded' and verify_run(path)['valid'],'invalid inference receipt')
    file=path/'output/predictions.json'
    require(file.stat().st_size<8*1024*1024,'unexpected artifact size')
    predictions=json.loads(file.read_text())
    result=summarize(predictions);require(result==read_json(path/'output/summary.json'),'aggregate mismatch')
    write_json(ROOT/'RESULT.json',result)
    manifest=check()['manifest_sha256'];lab=Controller(ROOT/'research_lab');import_forge(lab,ROOT/'dataset',manifest);import_trainlab(lab,path)
    ev=EvidenceRecord(evidence_type='research.frozen_value_probe',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,
        claim='Frozen-checkpoint value discrimination across controlled history contexts, not architecture qualification.',
        subjects=(EvidenceSubjectRef(kind='dataset',subject_id='frozen-value-grid',content_sha256=manifest),),
        producer=EvidenceProducerRef(producer_id='veritas-lifecycle-semantics',producer_version='1',content_sha256=_runtime()['source_trees']['investigation_world']),
        artifacts=(EvidenceArtifactRef(artifact_id='frozen-value-predictions',content_sha256=digest(path/'output/predictions.json')),),
        observed_at=datetime.fromisoformat(receipt['finished_at']),provenance={'scope':'GENERATED_FROZEN_MODEL_DIAGNOSTIC','model_fits':0})
    write_json(ROOT/'evidence.json',ev.model_dump(mode='json'));put_node(lab,'frozen-value-001','finding',result)
    put_node(lab,'frozen-value-evidence-001','evidence',ev.model_dump(mode='json'))
    link(lab,'frozen-value-001','derived_from','frozen-value-evidence-001');link(lab,'frozen-value-001','derived_from','forge.'+manifest[:24]);link(lab,'frozen-value-001','derived_from','trainlab.'+receipt['id'])
    write_json(ROOT/'knowledge.json',retrieve(lab))
    return {'predictions':len(predictions),'model_fits':0,'run_id':receipt['id'],'receipt_sha256':digest(path/'run.json')}

if __name__=='__main__':print(json.dumps(finalize(),indent=2))
