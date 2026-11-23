"""Exposed fixed-window language regression for the completed 002 campaign.

Reuse the exact 001 token windows and evaluation implementation. The targeted
repair's nondegradation reference is each seed's own fused001 parent. Original
full Connected COG remains a separate stronger reference and recovery gate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from common import HERE,PARENT,SEEDS,ARMS,parent_checkpoint,sha256,write_json
import numpy as np
import torch
from fusion.model import FusionModel,NATIVE_SHA256
from fusion.preservation import (fixed_windows,evaluate_windows,VALIDATION_SHA256,
                                 WINDOW_SEED,
                                 WINDOW_COUNT,TARGETS_PER_WINDOW)


def declaration(validation:Path,native:Path):
    if sha256(validation)!=VALIDATION_SHA256 or sha256(native)!=NATIVE_SHA256:
        raise ValueError("inherited language inputs failed identity checks")
    tokens=np.memmap(validation,dtype="<u2",mode="r")
    starts,windows=fixed_windows(tokens)
    prior=json.loads((PARENT/"evidence/LANGUAGE_PRESERVATION.json").read_text())
    window_hash=hashlib.sha256(windows.numpy().astype("<u2").tobytes()).hexdigest()
    if (starts!=prior["validation"]["start_positions"]
            or window_hash!=prior["validation"]["window_tokens_sha256"]):
        raise ValueError("002 regression windows do not reproduce001")
    return {"study":"COG-FUSION-002","scope":"exposed001 language regression windows, not a new held language panel",
        "validation_sha256":VALIDATION_SHA256,"native_checkpoint_sha256":NATIVE_SHA256,
        "window_seed":WINDOW_SEED,"window_count":WINDOW_COUNT,"targets_per_window":TARGETS_PER_WINDOW,
        "start_positions":starts,"window_tokens_sha256":window_hash,"total_targets":8192,
        "state_policy":"reset independently per window for every model","batch_size":8,
        "parent_nondegradation_max_relative_nll_increase":.05,
        "original_full_recovery_max_relative_nll_increase":.10,
        "source_sha256":sha256(Path(__file__)),
        "inherited_source_sha256":{name:sha256(PARENT/"fusion"/name) for name in ("preservation.py","model.py")}}


def verified_endpoints(runs):
    entries=[]
    for seed in SEEDS:
        for arm in ARMS:
            path=runs/f"{arm}_seed{seed}"/"final.pt"
            result=json.loads(path.with_name("RESULT.json").read_text())
            if (result.get("status")!="completed" or result.get("seed")!=seed
                    or result.get("arm")!=arm or result.get("steps")!=1200
                    or result.get("parent_sha256")!=sha256(parent_checkpoint(seed))
                    or result.get("checkpoint_sha256")!=sha256(path)):
                raise ValueError("fit incomplete or not the locked paired endpoint: "+str(path))
            for relative,expected in result["source_sha256"].items():
                if sha256(HERE.parent/relative)!=expected:
                    raise ValueError("a fitted source changed: "+relative)
            entries.append((arm,seed,path,result))
    return entries


def compare(metrics,reference,allowed):
    change=metrics["mean_nll"]/reference["mean_nll"]-1
    return {"relative_nll_change":change,"allowed_relative_nll_increase":allowed,
            "character_perplexity_ratio":math.exp(metrics["mean_nll"]-reference["mean_nll"]),
            "passed":change<=allowed}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--declare",action="store_true")
    parser.add_argument("--runs",type=Path,default=HERE/"evidence/runs")
    parser.add_argument("--validation",type=Path,default=PARENT/"evidence/reproduction_inputs/val.bin")
    parser.add_argument("--native",type=Path,default=PARENT/"evidence/reproduction_inputs/COG_CONNECTED_1M.pt")
    parser.add_argument("--output",type=Path,default=HERE/"evidence/LANGUAGE_PRESERVATION.json")
    args=parser.parse_args();torch.set_num_threads(1)
    protocol=declaration(args.validation,args.native)
    protocol_path=HERE/"evidence/LANGUAGE_PROTOCOL.json"
    if args.declare:
        if protocol_path.exists():raise FileExistsError("language regression protocol already declared")
        write_json(protocol_path,protocol);return
    if not protocol_path.is_file() or json.loads(protocol_path.read_text())!=protocol:
        raise ValueError("prospective language regression declaration missing or changed")
    if args.output.exists():raise FileExistsError("refusing to overwrite language regression scores")
    # No model is scored until all six fixed fitting endpoints are admitted.
    entries=verified_endpoints(args.runs)
    tokens=np.memmap(args.validation,dtype="<u2",mode="r")
    _,windows=fixed_windows(tokens)
    original=FusionModel.from_native(args.native,seed=17).eval()
    native_metadata=torch.load(args.native,map_location="cpu",weights_only=True)
    full=evaluate_windows(original,windows,mode="full_original_native",
                           native_feedback_enabled=native_metadata["feedback_enabled"])
    local=evaluate_windows(original,windows,mode="local_only")
    parents={}
    for seed in SEEDS:
        path=parent_checkpoint(seed)
        model=FusionModel.from_checkpoint(path).eval()
        metrics=evaluate_windows(model,windows,mode="local_only")
        parents[str(seed)]={"checkpoint_sha256":sha256(path),"model_identity":model.model_identity(),
                            "metrics":metrics,"original_full_gap":compare(metrics,full,.10)}
    results={"schema":"cog-fusion-002-language-preservation-v1","protocol":protocol,
        "protocol_sha256":sha256(protocol_path),"references":{"original_connected_cog":full,
            "original_local_only":local,"paired_fused001_parents":parents},"runs":[]}
    for arm,seed,path,fit in entries:
        model=FusionModel.from_checkpoint(path).eval()
        if (model.model_identity()!=fit["final_model_identity"] or model.chars!=original.chars
                or model.native.config!=original.native.config or not model.memory_enabled):
            raise ValueError("endpoint interpretation differs from the admitted candidate")
        metrics=evaluate_windows(model,windows,mode="local_only")
        row={"arm":arm,"seed":seed,"checkpoint_sha256":sha256(path),"model_identity":model.model_identity(),
             "metrics":metrics,"paired_parent_nondegradation":compare(metrics,parents[str(seed)]["metrics"],.05),
             "original_full_recovery":compare(metrics,full,.10)}
        results["runs"].append(row)
        print(json.dumps({"arm":arm,"seed":seed,"nll":metrics["mean_nll"],"ppl":metrics["character_perplexity"],
             "parent_nondegradation":row["paired_parent_nondegradation"]["passed"],
             "original_full_recovery":row["original_full_recovery"]["passed"]}),flush=True)
    results["all_continuous_parent_nondegradation"]=all(
        r["paired_parent_nondegradation"]["passed"] for r in results["runs"] if r["arm"]=="continuous")
    results["all_continuous_original_full_recovery"]=all(
        r["original_full_recovery"]["passed"] for r in results["runs"] if r["arm"]=="continuous")
    write_json(args.output,results)


if __name__=="__main__":main()
