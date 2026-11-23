"""Live persistent-session evaluation after all 002 endpoints are complete.

No reference operation is supplied to the model or executor. Expected operations
and records below are a separate scorer. Pending/source-erasure injections have
supplied typed scopes and are explicitly runtime controls, not learned parsing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from common import HERE, PARENT, ARMS, SEEDS, parent_checkpoint, sha256, write_json
import torch
from fusion.data import ENTITIES, RELATIONS, UNKNOWN
from fusion.interaction import run_checkpoint as exposed_001_interaction
from fusion.model import FusionModel, state_bytes
from fusion.session import PREFIX, Session


def new_scenario():
    """Frozen new history with asks after observations 2,4,6 before extensions."""
    return [
        {"kind":"observe","id":"initial-owner","text":"Faye owns the vault.","expected_op":[1,"vault","owner","Faye"]},
        {"kind":"observe","id":"initial-link","text":"The gem is in the vault.","expected_op":[1,"gem","container","vault"]},
        {"kind":"ask","id":"first-question","text":"Who owns the container holding the gem?","expected":"Faye."},
        {"kind":"observe","id":"other-owner","text":"Hadi owns the box.","expected_op":[1,"box","owner","Hadi"]},
        {"kind":"observe","id":"relevant-correction","text":"The vault belongs to Eren.","expected_op":[1,"vault","owner","Eren"]},
        {"kind":"ask","id":"after-correction","text":"Who owns what contains the gem?","expected":"Eren."},
        {"kind":"observe","id":"unrelated-crate","text":"Gita owns the crate.","expected_op":[1,"crate","owner","Gita"]},
        {"kind":"observe","id":"unrelated-box","text":"The box belongs to Cora.","expected_op":[1,"box","owner","Cora"]},
        {"kind":"ask","id":"after-unrelated","text":"Who is the owner of the container with the gem?","expected":"Eren."},
        {"kind":"observe","id":"bridge-change","text":"The gem is in the box.","expected_op":[1,"gem","container","box"]},
        {"kind":"observe","id":"off-route-chest","text":"Davi owns the chest.","expected_op":[1,"chest","owner","Davi"]},
        {"kind":"ask","id":"after-process-restart","text":"Who owns the container holding the gem?","expected":"Cora.","new_process":True},
        {"kind":"observe","id":"cancelled-transfer","text":"A transfer of the box to Hadi was cancelled.","expected_op":[0,None,None,None]},
        {"kind":"ask","id":"after-cancellation","text":"Who owns what contains the gem?","expected":"Cora."},
        {"kind":"pending","id":"pending-box-owner","key":["world","box","owner"]},
        {"kind":"restart","id":"restart-pending"},
        {"kind":"ask","id":"pending-question","text":"Who owns the container holding the gem?","expected":"pending","policy":True},
        {"kind":"ask","id":"unrelated-during-pending","text":"Who owns the vault?","expected":"Eren."},
        {"kind":"observe","id":"pending-retry","text":"The box belongs to Bela.","expected_op":[1,"box","owner","Bela"],"clear_pending":["pending-box-owner"]},
        {"kind":"ask","id":"after-pending-retry","text":"Who owns the container holding the gem?","expected":"Bela."},
        {"kind":"observe","id":"retract-owner","text":"The owner of the box is unknown.","expected_op":[2,"box","owner",None]},
        {"kind":"ask","id":"after-retraction","text":"Who owns what contains the gem?","expected":"unknown."},
        {"kind":"observe","id":"reinstated-owner","text":"Hadi owns the box.","expected_op":[1,"box","owner","Hadi"]},
        {"kind":"ask","id":"after-reinstatement","text":"Who owns what contains the gem?","expected":"Hadi."},
        {"kind":"erase_source","id":"erase-reinstatement","source":"reinstated-owner","expected_removed":["world","box","owner"]},
        {"kind":"ask","id":"after-source-erasure","text":"Who owns the container holding the gem?","expected":"unknown."},
        {"kind":"replay","id":"first-question","text":"Who owns the container holding the gem?"},
    ]


def state_digest(state):
    result = hashlib.sha256()
    def visit(value, name):
        if isinstance(value, torch.Tensor):
            result.update(name.encode()); result.update(str(tuple(value.shape)).encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value): visit(value[key], name+"."+key)
        else:
            for i, child in enumerate(value): visit(child, name+"."+str(i))
    visit(state,"state")
    return result.hexdigest()


def session_summary(session):
    snapshot = session.store.snapshot_state()
    return {"state_sha256":state_digest(session.state),"counters":dict(session.counters),
            "rng_sha256":hashlib.sha256(session.generator.get_state().numpy().tobytes()).hexdigest(),
            "revision":snapshot["revision"],"neural_blob_sha256":snapshot["neural_digest"],
            "tensor_bytes":state_bytes(session.state)}


def records_json(records):
    return [[list(key),value] for key,value in sorted(records.items())]


def operation_correct(predicted, expected):
    op, entity, relation, value = expected
    if predicted[0] != op: return False
    if op == 0: return True
    if predicted[1:3] != [ENTITIES.index(entity),RELATIONS.index(relation)]: return False
    return op != 1 or predicted[3] == ENTITIES.index(value)


def declared_protocol():
    return {"study":"COG-FUSION-002","scenario":"new-schedule-familiar-wording",
            "scope":"schedule/history shift; first asks after observations2/4/6; no learned-only threshold chosen after scoring",
            "typed_controls":"pending key and source erasure supplied; no gold writes or reads supplied",
            "steps":new_scenario(),"source_sha256":sha256(Path(__file__)),
            "immutable_runtime_sha256":{name:sha256(PARENT/"fusion"/name) for name in (
                "session.py","store.py","engine.py","model.py","interaction.py")}}


def run_new_scenario(checkpoint:Path,directory:Path):
    directory.mkdir(parents=True,exist_ok=True)
    database=directory/"session.sqlite"
    if database.exists(): raise FileExistsError("never overwrite a scored runtime fixture")
    session=Session.from_checkpoint(checkpoint,database,seed=200991)
    expected_records={}; transcript=[]; original_replies={}; consumed_inputs=0; emitted=0; prefixes=0
    updates=0; observations=0; questions=0; process_checks=[]; tensor_sizes=[]
    completed=True
    for step in new_scenario():
        row=dict(step)
        try:
            kind=step["kind"]
            if kind=="observe":
                actual=session.observe(step["text"],step["id"],clear_pending=step.get("clear_pending",()))
                consumed_inputs+=len(step["text"])+1; observations+=1; updates+=int(actual["record_updated"])
                op,entity,relation,value=step["expected_op"]
                if op==1: expected_records[("world",entity,relation)]=value
                elif op==2: expected_records.pop(("world",entity,relation),None)
                observed=session.store.current_records()
                row.update(result=actual,operation_correct=operation_correct(actual["predicted_operation"],step["expected_op"]),
                    full_current_ledger_correct=observed==expected_records,
                    expected_records=records_json(expected_records),actual_records=records_json(observed))
            elif kind=="ask":
                if step.get("new_process"):
                    before=session_summary(session); session.close()
                    command=[sys.executable,str(Path(__file__)),"--worker","--checkpoint",str(checkpoint),
                             "--database",str(database),"--question",step["text"],"--request-id",step["id"]]
                    process=subprocess.run(command,capture_output=True,text=True,check=True)
                    worker=json.loads(process.stdout)
                    actual=worker["result"]
                    session=Session.from_checkpoint(checkpoint,database,seed=987654)
                    after=session_summary(session)
                    check={"before_worker_restore_exact":before==worker["before"],
                           "after_worker_restore_exact":after==worker["after"],
                           "fresh_process":True,"worker_stderr":process.stderr}
                    process_checks.append(check); row["restart_check"]=check
                else:
                    actual=session.ask(step["text"],step["id"])
                consumed_inputs+=len(step["text"])+1; questions+=1
                if actual["generation_mode"]=="neural":
                    emitted+=len(actual["text"]); prefixes+=len(PREFIX)
                row.update(result=actual,correct=actual["text"]==step["expected"])
                original_replies[step["id"]]=actual
            elif kind=="pending":
                row["result"]=session.store.mark_pending(step["id"],key=tuple(step["key"]))
            elif kind=="restart":
                before=session_summary(session); session.close()
                session=Session.from_checkpoint(checkpoint,database,seed=987654)
                after=session_summary(session)
                row["restart_check"]={"restore_exact":before==after,"fresh_process":False}
                process_checks.append(row["restart_check"])
            elif kind=="erase_source":
                row["result"]=session.store.invalidate_source(step["source"],request_id=step["id"])
                expected_records.pop(tuple(step["expected_removed"]),None)
                observed=session.store.current_records()
                row.update(full_current_ledger_correct=observed==expected_records,
                           expected_records=records_json(expected_records),actual_records=records_json(observed))
            elif kind=="replay":
                before=session_summary(session)
                actual=session.ask(step["text"],step["id"])
                after=session_summary(session)
                row.update(result=actual,replay_preserves_current_continuation=before==after,
                           exact_historical_reply=actual["text"]==original_replies[step["id"]]["text"] and actual["replayed"])
            else: raise ValueError("unknown declared operation")
            row["state_after"]=session_summary(session)
            tensor_sizes.append(state_bytes(session.state))
            transcript.append(row)
        except Exception as exc:
            row["error"]={"type":type(exc).__name__,"message":str(exc)}
            transcript.append(row); completed=False; break
    declared=new_scenario()
    expected_accounting={"input_tokens":consumed_inputs,"emitted_tokens":emitted,"control_tokens":prefixes,
                         "requests":observations+questions,"observations":observations,"questions":questions,
                         "record_updates":updates,"historical_tokens_replayed":0}
    questions_expected=[r for r in declared if r["kind"]=="ask" and not r.get("policy")]
    question_rows=[r for r in transcript if r["kind"]=="ask" and not r.get("policy")]
    policies=[r for r in transcript if r.get("policy")]
    selected_pending=[r for r in policies if r.get("result",{}).get("evidence_status")=="pending"]
    observation_rows=[r for r in transcript if r["kind"]=="observe"]
    replay_rows=[r for r in transcript if r["kind"]=="replay"]
    snapshot=session.store.snapshot_state()
    result={"checkpoint_sha256":sha256(checkpoint),"completed":completed,"transcript":transcript,
        "learned_questions_correct":sum(bool(r.get("correct")) for r in question_rows),
        "learned_question_count":len(questions_expected),
        "learned_questions_executed":len(question_rows),
        "operations_correct":sum(bool(r.get("operation_correct")) for r in observation_rows),
        "operation_count":sum(r["kind"]=="observe" for r in declared),
        "all_observation_ledgers_correct":completed and all(r.get("full_current_ledger_correct",False) for r in observation_rows),
        "all_learned_questions_correct":completed and all(r.get("correct",False) for r in question_rows),
        "expected_pending_behavior_correct":completed and bool(policies) and all(r.get("correct",False) for r in policies),
        "pending_policy_if_scope_selected":(all(r["result"]["generation_mode"]=="runtime_policy" and r["result"]["text"]=="pending" for r in selected_pending) if selected_pending else None),
        "restart_checks":process_checks,
        "all_restarts_exact":completed and bool(process_checks) and all(all(value for key,value in check.items() if key.endswith("exact")) for check in process_checks),
        "historical_replay_correct":completed and len(replay_rows)==1 and all(r.get("replay_preserves_current_continuation") and r.get("exact_historical_reply") for r in replay_rows),
        "token_accounting_correct":session.counters==expected_accounting,
        "expected_accounting":expected_accounting,"observed_accounting":dict(session.counters),
        "neural_tensor_bytes_min":min(tensor_sizes) if tensor_sizes else None,
        "neural_tensor_bytes_max":max(tensor_sizes) if tensor_sizes else None,
        "snapshot_blob_bytes":len(snapshot["neural_state"]),"database_bytes":database.stat().st_size,
        "scope":"predeclared unfamiliar schedule and entity history using familiar wording; supplied runtime pending/erasure controls"}
    result["complete_pipeline_pass"]=(completed and result["all_observation_ledgers_correct"]
        and result["operations_correct"]==result["operation_count"] and result["all_learned_questions_correct"]
        and result["expected_pending_behavior_correct"] and result["all_restarts_exact"]
        and result["historical_replay_correct"] and result["token_accounting_correct"])
    session.close(); return result


def completed_checkpoint(arm,seed,runs):
    path=runs/f"{arm}_seed{seed}"/"final.pt"
    result=json.loads(path.with_name("RESULT.json").read_text())
    if (result.get("status")!="completed" or result.get("arm")!=arm or result.get("seed")!=seed
            or sha256(path)!=result.get("checkpoint_sha256")):
        raise ValueError("incomplete or incompatible training checkpoint")
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--declare",action="store_true")
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--checkpoint",type=Path); parser.add_argument("--database",type=Path)
    parser.add_argument("--question");parser.add_argument("--request-id")
    parser.add_argument("--runs",type=Path,default=HERE/"evidence/runs")
    parser.add_argument("--output",type=Path,default=HERE/"evidence/RUNTIME_RESULTS.json")
    args=parser.parse_args(); torch.set_num_threads(1)
    if args.worker:
        with Session.from_checkpoint(args.checkpoint,args.database,seed=998877) as session:
            before=session_summary(session); result=session.ask(args.question,args.request_id)
            print(json.dumps({"before":before,"result":result,"after":session_summary(session)}))
        return
    protocol_path=HERE/"evidence/RUNTIME_PROTOCOL.json"
    if args.declare:
        if protocol_path.exists():raise FileExistsError("runtime protocol already declared")
        write_json(protocol_path,declared_protocol());return
    if not protocol_path.is_file() or json.loads(protocol_path.read_text())!=declared_protocol():
        raise RuntimeError("prospective runtime declaration is missing or changed")
    if args.output.exists():raise FileExistsError("refusing to overwrite scored runtime results")
    # Resolve all new endpoints before even evaluating exposed parent checkpoints.
    fits=[(arm,seed,completed_checkpoint(arm,seed,args.runs)) for seed in SEEDS for arm in ARMS]
    fits=[("parent_001",seed,parent_checkpoint(seed)) for seed in SEEDS]+fits
    results={"schema":"cog-fusion-002-runtime-v1","protocol_sha256":sha256(protocol_path),"runs":[]}
    for arm,seed,checkpoint in fits:
        root=args.output.parent/"runtime"/f"{arm}_seed{seed}"
        old=exposed_001_interaction(checkpoint,root/"exposed_001")
        new=run_new_scenario(checkpoint,root/"new_schedule")
        result={"arm":arm,"seed":seed,"checkpoint_sha256":sha256(checkpoint),
                "exposed_001_regression":old,"new_schedule":new}
        write_json(root/"RESULT.json",result);results["runs"].append(result)
        print(json.dumps({"arm":arm,"seed":seed,"old_correct":old["learned_questions_correct"],
              "old_total":old["learned_question_count"],"new_correct":new["learned_questions_correct"],
              "new_total":new["learned_question_count"],"new_pipeline":new["complete_pipeline_pass"],
              "accounting":new["token_accounting_correct"]}),flush=True)
    write_json(args.output,results)


if __name__=="__main__": main()
