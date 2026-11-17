"""Draft private Kaggle transport entry; one admitted native segment per invocation.
Never reads account credentials, launches another job, or resumes without Library receipt.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,sys,zipfile

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def copy_bundle(source,dest,manifest_sha):
 source=Path(source).resolve();dest=Path(dest).resolve();manifest=source/'BUNDLE_MANIFEST.json'
 if not manifest.is_file():
  found=list(source.rglob('BUNDLE_MANIFEST.json'))
  if len(found)!=1:raise ValueError('expected one exact private input bundle manifest in Kaggle extracted tree')
  manifest=found[0];source=manifest.parent.resolve()
 if sha(manifest)!=manifest_sha:raise ValueError('outer input bundle identity mismatch')
 files=read(manifest)['files']
 if dest.exists() and any(dest.iterdir()):raise ValueError('refuse to overwrite an existing work directory')
 dest.mkdir(parents=True,exist_ok=True)
 for n,h in files.items():
  p=Path(n);raw=source/p;src=raw.resolve()
  if any(q.is_symlink() for q in [raw,*raw.parents] if q.is_relative_to(source)):raise ValueError('symlink bundle input prohibited')
  if p.is_absolute() or '..' in p.parts or not src.is_relative_to(source) or not src.is_file() or src.is_symlink():raise ValueError('unsafe bundle entry')
  if p.name in ('kaggle.json','.env') or any(x in p.parts for x in ('.git','.aws')):raise ValueError('credential-like bundle entry prohibited')
  if sha(src)!=h:raise ValueError('bundle content hash mismatch '+n)
  out=dest/p;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out)
 shutil.copyfile(manifest,dest/'BUNDLE_MANIFEST.json');return files

def retain_failure(work,error,output_root=None):
 work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
 output_root=Path(output_root).resolve() if output_root else work.parent;output_root.mkdir(parents=True,exist_ok=True)
 receipt=output_root/'TRANSPORT_FAILURE.json';archive=output_root/'FAILED_NATIVE_ATTEMPT.zip'
 if archive.exists() or receipt.exists():raise ValueError('existing failed attempt evidence must not be overwritten')
 write(receipt,{'status':'failed','error':repr(error),'scope':'No continuation permitted; native partial outputs retained when present'})
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  z.write(receipt,receipt.name)
  for base,directory in ((work,'runs'),(work,'research_lab_020'),(output_root,'uncommitted_runs')):
   for p in (base/directory).rglob('*'):
    if p.is_file():z.write(p,p.relative_to(base))
 return archive

def run(source,work,manifest_sha,config_name,output_root=None):
 work=Path(work).resolve();output_root=Path(output_root).resolve() if output_root else work.parent;output_root.mkdir(parents=True,exist_ok=True);copy_bundle(source,work,manifest_sha)
 vendor=work/'ecosystem_vendor';vm=read(vendor/'VENDOR_MANIFEST.json')
 for n,h in vm['files'].items():
  p=(vendor/n).resolve()
  if Path(n).is_absolute() or not p.is_relative_to(vendor) or not p.is_file() or sha(p)!=h:raise ValueError('vendored ecosystem integrity mismatch')
 sys.path[:0]=[str(work),str(vendor)]
 os.environ['PYTHONPATH']=os.pathsep.join([str(vendor),str(work),os.environ.get('PYTHONPATH','')]);os.chdir(work)
 import ecosystem020
 from trainlab.runner import run_experiment,verify_run
 from research_lab.core import Controller
 from research_lab.adapters import import_trainlab
 from research_lab.evidence import put_node,link
 from investigation_world.evidence import EvidenceRecord,EvidenceOutcome,EvidenceVisibility,EvidenceProducerRef,EvidenceSubjectRef,EvidenceArtifactRef
 from datetime import datetime
 ecosystem020.check_runtime(work);recipe=ecosystem020.prepare(work,config_name)
 record=run_experiment(recipe,output_root/'uncommitted_runs');directory=output_root/'uncommitted_runs'/record['id'];verified=verify_run(directory)
 if not verified['valid']:raise ValueError('native receipt integrity invalid; retain work directory for investigation')
 lab=Controller(work/'research_lab_020');import_trainlab(lab,directory)
 ev=EvidenceRecord(evidence_type='woyengi020.segment_execution',outcome=EvidenceOutcome.OBSERVED,visibility=EvidenceVisibility.OPERATOR_PRIVATE,claim='Native bounded training execution; not scientific qualification or independent compute attestation',subjects=(EvidenceSubjectRef(kind='run',subject_id=record['id'],content_sha256=sha(directory/'run.json')),),producer=EvidenceProducerRef(producer_id='woyengi020-native-ecosystem',producer_version='draft1',content_sha256=sha(work/'ecosystem020.py')),artifacts=(EvidenceArtifactRef(artifact_id='trainlab_receipt',role='execution',content_sha256=sha(directory/'run.json'),media_type='application/json'),),observed_at=datetime.fromisoformat(record['finished_at']),provenance={'scope':'MIXED_AUTHENTIC_TRAIN_VALIDATION_AND_CONSTRUCTED_PRIVATE','status':record['status']})
 write(directory/'veritas_evidence.json',ev.model_dump(mode='json'));node='woyengi020.'+record['id'];put_node(lab,node,'finding',ev.model_dump(mode='json'));link(lab,node,'derived_from','trainlab.'+record['id'])
 archive=output_root/('native_'+record['id']+'.zip');pins={}
 if archive.exists():raise ValueError('refuse to overwrite existing native output archive')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in directory.rglob('*'):
   if p.is_file():n=str(p.relative_to(directory));pins[n]=sha(p);z.write(p,n)
  z.writestr('ARCHIVE_MANIFEST.json',json.dumps(pins,sort_keys=True,indent=2))
 with zipfile.ZipFile(archive) as z:
  if z.testzip() is not None:raise ValueError('native output ZIP CRC failed')
  for n,h in pins.items():
   if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('native output ZIP content mismatch')
 summary={'status':record['status'],'run_id':record['id'],'archive':archive.name,'archive_sha256':sha(archive),'bundle_manifest_sha256':manifest_sha,'run_receipt_sha256':sha(directory/'run.json'),'scope':'Kaggle output retained; NOT a verified Library backup. No automatic continuation.'}
 cp=directory/'output/checkpoint.pt'
 if cp.is_file():summary['checkpoint_sha256']=sha(cp)
 write(output_root/'RECOVERY_RECEIPT.json',summary)
 if record['status']!='succeeded':raise RuntimeError('native run failed; output archive retained, no continuation')
 # All run bytes are in the CRC/hash-verified archive. Avoid exporting hundreds of duplicate files.
 shutil.rmtree(directory)
 if not any((output_root/'uncommitted_runs').iterdir()):(output_root/'uncommitted_runs').rmdir()
 return summary

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--input-root',required=True);p.add_argument('--work',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--config',required=True);a=p.parse_args()
 try:print(json.dumps(run(a.input_root,a.work,a.manifest_sha256,a.config,a.output_root)))
 except Exception as error:
  retain_failure(a.work,error,a.output_root)
  raise
