"""Create the evidence archive with verified links to duplicate frozen inputs."""
import hashlib
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
path=ROOT.parent/"cog-047-050-evidence.tar.gz"
files={
    "minilm":ROOT/"recipe/minilm/model.safetensors",
    "nli":ROOT/"recipe/nli/model_quint8_avx2.onnx",
    **{s:ROOT/f"recipe/multiwoz/{s}/dialogues_001.json" for s in ("train","dev","test")},
}


def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def arcname(file):
    return f"{ROOT.name}/{file.relative_to(ROOT)}"


expected={name:(arcname(file),digest(file)) for name,file in files.items()}


def include(info):
    pieces=Path(info.name).parts
    if "__pycache__" in pieces:
        return None
    leaf=pieces[-1]
    key=("minilm" if leaf=="model.safetensors" else
         "nli" if leaf=="model_quint8_avx2.onnx" else
         pieces[-2] if leaf=="dialogues_001.json" else None)
    if key and info.isfile():
        original,sha=expected[key]
        if info.name==original:
            return None
        if digest(ROOT.parent/info.name)!=sha:
            raise ValueError(info.name)
        info.type=tarfile.LNKTYPE
        info.linkname=original
        info.size=0
    return info


with tarfile.open(path,"w:gz",compresslevel=3) as archive:
    for file in files.values():
        archive.add(file,arcname=arcname(file))
    archive.add(ROOT,arcname=ROOT.name,filter=include)
with tarfile.open(path,"r:gz") as archive:
    for name,sha in expected.values():
        handle=archive.extractfile(name)
        assert handle is not None and hashlib.file_digest(handle,"sha256").hexdigest()==sha
