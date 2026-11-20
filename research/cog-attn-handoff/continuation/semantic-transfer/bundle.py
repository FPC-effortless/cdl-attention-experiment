"""Package the experiment with one copy of each byte-identical encoder file."""
import hashlib
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODELS=(
    ROOT/"recipe/minilm/model.safetensors",
    ROOT/"recipe/nli/model_quint8_avx2.onnx",
)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


canonical={p.name:(f"{ROOT.name}/{p.relative_to(ROOT)}",sha256(p)) for p in MODELS}


def include(info):
    if "__pycache__" in Path(info.name).parts:
        return None
    if Path(info.name).name in canonical and info.isfile():
        original,digest=canonical[Path(info.name).name]
        if info.name==original:
            return None
        path=ROOT.parent/info.name
        if sha256(path)!=digest:
            raise RuntimeError(f"snapshot weights changed: {info.name}")
        info.type=tarfile.LNKTYPE
        info.linkname=original
        info.size=0
    return info


target=ROOT.parent/"semantic-transfer-evidence.tar.gz"
with tarfile.open(target,"w:gz",compresslevel=3) as archive:
    for path in MODELS:
        archive.add(path,arcname=f"{ROOT.name}/{path.relative_to(ROOT)}")
    archive.add(ROOT,arcname=ROOT.name,filter=include)
with tarfile.open(target,"r:gz") as archive:
    for filename,digest in canonical.values():
        data=archive.extractfile(filename)
        if data is None or hashlib.file_digest(data,"sha256").hexdigest()!=digest:
            raise RuntimeError(f"archive weight mismatch: {filename}")
