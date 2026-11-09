"""Fresh structural proof build, declared-theorem axiom audit and negative controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lean",default="lean")
    ap.add_argument("--explicit-path-launcher")
    ap.add_argument("--distribution-archive")
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    build=root/"verification"/"lean_build"
    build.mkdir(parents=True,exist_ok=True)
    cmd=([args.explicit_path_launcher] if args.explicit_path_launcher else [])+[args.lean]
    env=os.environ.copy()
    env["LEAN_PATH"]=str(build)
    receipts=[]

    def invoke(extra):
        result=subprocess.run(cmd+extra,cwd=root,env=env,text=True,capture_output=True)
        receipts.append({"command":cmd+extra,"returncode":result.returncode,
                         "stdout":result.stdout,"stderr":result.stderr})
        return result

    version=invoke(["--version"])
    assert version.returncode==0,version.stderr
    source=root/"CSMRegulation.lean"
    # Anchor at declaration lines; prose comments also contain the word theorem.
    names=re.findall(r"(?m)^(?:@\[simp\]\s+)?theorem\s+(\w+)",source.read_text())
    assert len(names)==len(set(names)) and names
    compiled=invoke(["--trust=0","-o",str(build/"CSMRegulation.olean"),str(source)])
    assert compiled.returncode==0,compiled.stdout+compiled.stderr
    audit=build/"Audit.lean"
    audit.write_text("import CSMRegulation\n"+"\n".join(
        "#print axioms CSMRegulation."+name for name in names)+"\n")
    audited=invoke(["--trust=0",str(audit)])
    assert audited.returncode==0,audited.stdout+audited.stderr
    (root/"verification"/"axioms.log").write_text(audited.stdout+audited.stderr)
    dependencies=set()
    for group in re.findall(r"depends on axioms:\s*\[([^\]]*)\]",audited.stdout,re.S):
        dependencies.update(x.strip() for x in group.split(",") if x.strip())
    allowed={"propext","Quot.sound","Classical.choice"}
    assert dependencies <= allowed,dependencies
    assert "sorryAx" not in audited.stdout and "Lean.ofReduceBool" not in audited.stdout
    # Every requested name must actually appear in the checker output.
    assert all("CSMRegulation."+name in audited.stdout for name in names)
    negatives=[]
    for name,body in [("false_arithmetic","import Std\nexample : (0 : Nat) = 1 := by decide\n"),
                      ("unjustified_equality","import Std\nexample (a b : Nat) : a = b := by rfl\n")]:
        path=build/(name+".lean");path.write_text(body)
        result=invoke(["--trust=0",str(path)])
        assert result.returncode!=0 and "error:" in result.stdout+result.stderr
        negatives.append({"name":name,"rejected":True})
    hashes={"CSMRegulation.lean":digest(source),"verify_lean.py":digest(__file__)}
    lp=Path(args.lean)
    if lp.exists():
        hashes["official_lean_executable"]=digest(lp)
        shared=lp.parent.parent/"lib"/"lean"/"libleanshared.so"
        if shared.exists():hashes["official_shared_library"]=digest(shared)
    if args.explicit_path_launcher:
        hashes["explicit_path_launcher"]=digest(args.explicit_path_launcher)
        hashes["explicit_path_launcher_source"]=digest(root/"tooling"/"lean_explicit_path.c")
    if args.distribution_archive:hashes["official_distribution_archive"]=digest(args.distribution_archive)
    result={"status":"PASS","compiler":version.stdout.strip(),"trust_level":0,
            "declared_theorems_audited":len(names),"theorems":names,
            "axiom_dependencies":sorted(dependencies),"negative_controls":negatives,
            "hashes":hashes,"adapted_application_path_launcher":bool(args.explicit_path_launcher),
            "scope":"Structural pure-transition and conditional affine-algebra theorems. No formal proof of real-valued stability, IEEE refinement, training or model quality.",
            "commands":receipts}
    (root/"verification"/"lean.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="commands"},indent=2))


if __name__=="__main__":main()
