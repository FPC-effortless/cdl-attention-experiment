"""LRR-006: five frozen LRR-005 models; energy vs scalar selective answer.

Never train the historical hard router or choose thresholds using TEST.
All failed seeds, logits, risk curves and calibration receipts are preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .lrr005_register import CausalRoleRegister
from .lrr006_energy import (
    build_rows, fit_energy, score_rows, dev_threshold, eval_scored,
    descriptive_risk_curve,
)

SOURCE_SHA = {
    41: "be56f93e5991cea72a2489c37d056015540bbeef4ca4aa4aa6fab7790b5fd676",
    42: "fa13b60b9c340468e9e01bac7de96edb49b420f8e7061b773a71df9dc04be4a5",
    43: "118258c03b88465a057613696b85c4643fed7e0700404726004762cac1ce9331",
    44: "4e855e0cd23c605449bc26fc8ab5a9937627c8660629d54f39cfe23919e32117",
    45: "a8a5567b34240b1e229f6c5c9fd4cb69de933294650b3e39d307f32e5005a044",
}


def run(checkpoint_dir, seeds=(41,42,43,44,45), lengths=(7,32,128),
        train_docs=256, dev_docs=128, test_docs=64, steps=600,
        save_heads=None):
    torch.set_num_threads(1)
    output = {
        "experiment": "LRR-006 train conditional EBM to select one hard-indexed candidate",
        "status": "not an LM attention replacement; energy calibration has no guarantee",
        "source_artifact": "LRR-005/13466926475",
        "source_scientific_sha": "1d114fbc28c6b5b37084f3afbbd32a08676f7eb4",
        "torch_version": torch.__version__, "python": platform.python_version(),
        "frozen_router_parameters": 4748,
        "head_training_steps": steps, "train_histories_per_seed":train_docs,
        "dev_histories_per_seed":dev_docs,
        "test_histories_per_seed_cell":test_docs,
        "seeds":list(seeds),"lengths":list(lengths),"training":[],"results":[],
    }
    if save_heads:
        Path(save_heads).mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        began=time.perf_counter()
        source=Path(checkpoint_dir)/f"lrr005_register_local_credit_{seed}.pt"
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != SOURCE_SHA[seed]:
            raise ValueError(f"frozen source checksum mismatch seed={seed}")
        router=CausalRoleRegister()
        router.load_state_dict(torch.load(source,weights_only=True,map_location="cpu"),
                               strict=True)
        router.eval()
        if sum(x.numel() for x in router.parameters())!=4748:
            raise AssertionError("modified source architecture")
        for p in router.parameters():
            p.requires_grad_(False)
        training=build_rows(router,seed,split="train",length=7,
                            n=train_docs,layout="regular")
        correct=sum(r["correct"] for r in training if r["feature"] is not None)
        incorrect=sum(not r["correct"] for r in training if r["feature"] is not None)
        unavailable=sum(r["feature"] is None for r in training)
        if not (correct and incorrect):
            raise AssertionError("energy training positive/negative missing")
        head,training_meta=fit_energy(training,seed,steps=steps)
        if save_heads:
            torch.save(head.state_dict(),Path(save_heads)/f"lrr006_energy_{seed}.pt")
        dev=build_rows(router,seed,split="dev",length=32,
                       n=dev_docs,layout="regular")
        e_dev=dev_threshold(score_rows(dev,head=head),max_absent_fpr=.10)
        s_dev=dev_threshold(score_rows(dev,scalar=True),max_absent_fpr=.10)
        output["training"].append({
            "seed":seed,"frozen_source_sha256":digest,
            "train_correct_candidates":correct,
            "train_invalid_candidates":incorrect,
            "train_no_candidate_cases":unavailable,
            **training_meta, "dev_energy":e_dev,"dev_scalar":s_dev,
            "total_head_train_and_feature_wall_seconds":time.perf_counter()-began,
        })
        for n in lengths:
            for layout in ("regular","shift","novel_fillers"):
                cases=build_rows(router,seed,split="test",length=n,
                                n=test_docs,layout=layout)
                es=score_rows(cases,head=head)
                ss=score_rows(cases,scalar=True)
                raw=eval_scored(ss,float("-inf"))
                energy=eval_scored(es,e_dev["threshold"])
                scalar=eval_scored(ss,s_dev["threshold"])
                energy_zero=eval_scored(es,0.)
                if energy["positive"]["correct_coverage"] > raw["positive"]["correct_coverage"] + 1e-12:
                    raise AssertionError("energy has incorrectly changed hard-index recall")
                output["results"].append({
                    "seed":seed,"facts":n,"layout":layout,
                    "raw_hard_index":raw,
                    "energy_dev_calibrated":energy,
                    "scalar_dev_calibrated":scalar,
                    "energy_zero_threshold":energy_zero,
                    "energy_risk_curve":descriptive_risk_curve(es) if n==32 and layout=="regular" else [],
                    "scalar_risk_curve":descriptive_risk_curve(ss) if n==32 and layout=="regular" else [],
                })
    main=[r for r in output["results"] if r["facts"]==32 and r["layout"]=="regular"]
    if len(main)!=len(seeds):
        raise AssertionError("missing primary seed evidence")
    differences=[]
    for row in main:
        a=row["energy_dev_calibrated"]["positive"]["correct_coverage"]
        b=row["scalar_dev_calibrated"]["positive"]["correct_coverage"]
        differences.append(a-b)
    output["decision"]={
        "raw_retrieval_mean":statistics.mean(r["raw_hard_index"]["positive"]["correct_coverage"] for r in main),
        "energy_correct_coverage":statistics.mean(r["energy_dev_calibrated"]["positive"]["correct_coverage"] for r in main),
        "scalar_correct_coverage":statistics.mean(r["scalar_dev_calibrated"]["positive"]["correct_coverage"] for r in main),
        "paired_energy_minus_scalar":differences,
        "energy_known_absent_false_answer_rate":statistics.mean(r["energy_dev_calibrated"]["known_absent"]["accept_rate"] for r in main),
        "scalar_known_absent_false_answer_rate":statistics.mean(r["scalar_dev_calibrated"]["known_absent"]["accept_rate"] for r in main),
        "energy_unknown_false_answer_rate":statistics.mean(r["energy_dev_calibrated"]["unknown_relation"]["accept_rate"] for r in main),
        "energy_benefit_gate":sum(
            (r["energy_dev_calibrated"]["positive"]["correct_coverage"] -
             r["scalar_dev_calibrated"]["positive"]["correct_coverage"] >= .10)
            and r["energy_dev_calibrated"]["known_absent"]["accept_rate"]<=.10
            for r in main)>=4,
        "always_reject_false_answer_rate":0.,
        "always_reject_correct_coverage":0.,
        "no_universal_abstention_guarantee":True,
        "no_hard_retrieval_improvement_claim":True,
    }
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--seeds",default="41,42,43,44,45")
    p.add_argument("--lengths",default="7,32,128")
    p.add_argument("--train-docs",type=int,default=256)
    p.add_argument("--dev-docs",type=int,default=128)
    p.add_argument("--test-docs",type=int,default=64)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--head-dir",type=Path,default=Path("/tmp/lrr006-heads"))
    p.add_argument("--output",type=Path,default=Path("/tmp/lrr006-results.json"))
    args=p.parse_args()
    report=run(args.checkpoint_dir,tuple(int(x) for x in args.seeds.split(",")),
               tuple(int(x) for x in args.lengths.split(",")),args.train_docs,
               args.dev_docs,args.test_docs,args.steps,args.head_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "decision":report["decision"],
        "training":report["training"],
        "evaluated_cells":len(report["results"]),
        "full_output":str(args.output),
    },indent=2))


if __name__=="__main__":
    main()
