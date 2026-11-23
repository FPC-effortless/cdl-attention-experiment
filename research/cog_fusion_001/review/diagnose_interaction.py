"""Post-exposure, no-fit state interventions on one observed interaction failure.

This script diagnoses the exposed correction sentence. It does not add examples
to training, select a checkpoint, change held-panel scores or qualify a repair.
Every candidate writer sees only the sentence plus the selected continuation;
expected labels are used afterward for diagnostic comparison.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fusion.data import ENTITIES, RELATIONS, UNKNOWN
from fusion.engine import evidence_tensors, execute, greedy_generate, predicted_operation, strict_read
from fusion.model import FusionModel, clone_state
from fusion.session import PREFIX, Session


TARGET = "The chest belongs to Bela."
EXPECTED = (1, 8, 0, 1)
SEEDS = (17, 29, 41)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(value, path="state"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from tensors(value[key], path + "." + key)
    else:
        for index, item in enumerate(value):
            yield from tensors(item, path + "." + str(index))


def state_identity(state):
    digest = hashlib.sha256()
    norms = {}
    for name, tensor in tensors(state):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        norms[name] = float(tensor.double().norm())
    return {"tensor_sha256": digest.hexdigest(), "tensor_l2_norms": norms}


def max_state_difference(first, second):
    a, b = dict(tensors(first)), dict(tensors(second))
    if a.keys() != b.keys():
        raise AssertionError("state geometry changed")
    return max(float((a[key] - b[key]).abs().max()) for key in a)


def named_operation(operation):
    op, entity, relation, value = operation
    return {"operation": ("hold", "set", "delete")[op], "entity": ENTITIES[entity],
            "relation": RELATIONS[relation], "value": "unknown" if value == UNKNOWN else ENTITIES[value]}


@torch.inference_mode()
def parse(model, state):
    inputs, lengths = model.encode_texts([TARGET + "\n"])
    feature, semantics, _ = model.encode_event(inputs, lengths, clone_state(state))
    operation = predicted_operation(semantics, 0)
    probabilities = semantics["entity"][0].softmax(-1)
    top = probabilities.topk(5)
    return {"predicted_operation": list(operation), "named_prediction": named_operation(operation),
            "exact_operation": operation == EXPECTED, "entity_correct": operation[1] == EXPECTED[1],
            "entity_top_five": [{"entity": ENTITIES[index], "probability": probability}
                                for probability, index in zip(top.values.tolist(), top.indices.tolist())],
            "chest_probability": float(probabilities[8]), "box_probability": float(probabilities[10]),
            "box_minus_chest_logit": float(semantics["entity"][0, 10] - semantics["entity"][0, 8]),
            "semantic_logits": {key: value[0].tolist() for key, value in semantics.items()},
            "feature": feature[0].tolist()}


@torch.inference_mode()
def replay_prefix(model, prefix, *, answer_mode):
    """Replay actual text through declared interfaces without supervision.

    answer_mode=full uses the model's real generation. prefix_only consumes
    the runtime Answer prefix but zero generated characters. no_generation
    consumes questions but no decoding tokens. observations_only skips asks.
    """
    if answer_mode not in ("full", "prefix_only", "no_generation", "observations_only", "neutral_output"):
        raise ValueError("unknown diagnostic replay mode")
    state = model.init_state(1)
    ledger, trace = {}, []
    for record in prefix:
        if record["type"] == "ask" and answer_mode == "observations_only":
            continue
        text = record["input"]
        tokens, lengths = model.encode_texts([text + "\n"])
        _, semantics, state = model.encode_event(tokens, lengths, state)
        if record["type"] == "observe":
            operation = predicted_operation(semantics, 0)
            execute(ledger, operation)
            trace.append({"kind": "observation", "text": text, "operation": list(operation),
                          "encoded_characters": int(lengths.sum())})
        elif record["type"] == "ask":
            entity = int(semantics["query_entity"][0].argmax())
            plan = int(semantics["query_plan"][0].argmax())
            resolution = strict_read(ledger, entity, plan)
            answer = ""
            if answer_mode == "full":
                generated, state = greedy_generate(model, state, [resolution])
                answer = generated[0]
            elif answer_mode in ("prefix_only", "neutral_output"):
                values, masks = evidence_tensors([resolution])
                decode_text = PREFIX
                if answer_mode == "neutral_output":
                    # Length comes from the exposed model-generated reply, never
                    # the target answer label. This is a declared diagnostic input.
                    answer = " " * len(record["result"]["text"])
                    decode_text += answer
                decode_tokens, _ = model.encode_texts([decode_text])
                _, state = model.decode(decode_tokens, state, values, masks)
            trace.append({"kind": "question", "text": text, "predicted_entity": entity,
                          "predicted_plan": plan, "selected_values": resolution.values,
                          "consumed_output_text": answer, "consumed_answer_prefix": answer_mode != "no_generation",
                          "encoded_characters": int(lengths.sum()),
                          "decoder_characters": 0 if answer_mode == "no_generation" else len(PREFIX) + len(answer)})
        else:
            raise ValueError("unexpected interaction operation in the exposed prefix")
    return state, trace


@torch.no_grad()
def diagnose_seed(seed, original, scratch):
    checkpoint = HERE / "evidence" / "runs" / f"memory_seed{seed}" / "final.pt"
    if sha(checkpoint) != original["checkpoint_sha256"]:
        raise ValueError("diagnostic checkpoint differs from the exposed interaction")
    model = FusionModel.from_checkpoint(checkpoint).eval()
    correction_index = next(index for index, record in enumerate(original["transcript"])
                            if record.get("type") == "observe" and record.get("input") == TARGET)
    prefix = original["transcript"][:correction_index]
    reproduced = []
    with Session(model, scratch / f"seed{seed}.sqlite", scope="world", seed=0, max_chars=12) as session:
        for record in prefix:
            request_id = record["result"]["request_id"]
            actual = (session.observe(record["input"], request_id) if record["type"] == "observe"
                      else session.ask(record["input"], request_id, temperature=0))
            relevant = "predicted_operation" if record["type"] == "observe" else "text"
            if actual[relevant] != record["result"][relevant]:
                raise AssertionError("could not reproduce the original exposed interaction prefix")
            reproduced.append({"type": record["type"], "input": record["input"],
                               "observed": actual[relevant], "matches_recorded": True})
        full_state = clone_state(session.state)
        original_parse = parse(model, full_state)
        correction_result = session.observe(TARGET, "diagnostic-correction")
        if correction_result["predicted_operation"] != original_parse["predicted_operation"]:
            raise AssertionError("direct semantic probe differs from actual Session execution")
        committed_records = [[list(key), value] for key, value in sorted(session.store.current_records().items())]

    manual_full, manual_trace = replay_prefix(model, prefix, answer_mode="full")
    manual_difference = max_state_difference(full_state, manual_full)
    if manual_difference != 0:
        raise AssertionError("manual full replay did not exactly reproduce Session continuation tensors")
    fresh = model.init_state(1)
    states = {
        "reset_all": (fresh, "All continuation tensors start from the model's initial state."),
        "full_interactive": (full_state, "Exact original Session prefix, including generated answer characters."),
    }
    replay_traces = {"full": manual_trace}
    for mode in ("observations_only", "no_generation", "prefix_only", "neutral_output"):
        state, trace = replay_prefix(model, prefix, answer_mode=mode)
        descriptions = {
            "observations_only": "Same observations, skipping questions and all output processing.",
            "no_generation": "Same observations and questions; skip Answer prefixes and generated characters.",
            "prefix_only": "Same observations and questions; consume Answer prefixes but zero generated characters.",
            "neutral_output": "Same observations and questions; replace generated characters by equal-length spaces.",
        }
        states[mode] = (state, descriptions[mode])
        replay_traces[mode] = trace

    decoder_zero = clone_state(full_state)
    decoder_zero["decoder"] = torch.zeros_like(decoder_zero["decoder"])
    states["clear_decoder_only"] = (decoder_zero, "Zero only the answer decoder controller; retain all local backbone state.")
    backbone_zero = clone_state(full_state)
    backbone_zero["native"]["layers"] = clone_state(fresh)["native"]["layers"]
    states["clear_backbone_layers_only"] = (backbone_zero, "Zero only all six local hidden and previous-input tensors; retain decoder/event/legacy state.")
    hidden_zero = clone_state(full_state)
    previous_zero = clone_state(full_state)
    for layer in hidden_zero["native"]["layers"]:
        layer["hidden"] = torch.zeros_like(layer["hidden"])
    for layer in previous_zero["native"]["layers"]:
        layer["previous"] = torch.zeros_like(layer["previous"])
    states["clear_hidden_only"] = (hidden_zero, "Zero only the local recurrent hidden states; retain previous-input features.")
    states["clear_previous_only"] = (previous_zero, "Zero only previous-input feature tensors; retain recurrent hidden states.")
    event_zero = clone_state(full_state)
    event_zero["event"] = torch.zeros_like(event_zero["event"])
    states["clear_event_only"] = (event_zero, "Zero only the prior event feature; the event encoder already resets its GRU for the new event.")

    results = {}
    for name, (state, intervention) in states.items():
        results[name] = {"intervention": intervention, "state_before": state_identity(state),
                         "maximum_state_difference_from_full": max_state_difference(state, full_state),
                         **parse(model, state)}
    full_logits = results["full_interactive"]["semantic_logits"]
    logits_equal = lambda condition: results[condition]["semantic_logits"] == full_logits
    causal = {"isolated_parse_correct": results["reset_all"]["exact_operation"],
              "full_interactive_parse_correct": results["full_interactive"]["exact_operation"],
              "backbone_reset_restores_exact_operation": results["clear_backbone_layers_only"]["exact_operation"],
              "decoder_reset_restores_exact_operation": results["clear_decoder_only"]["exact_operation"],
              "decoder_reset_leaves_all_semantic_logits_identical": logits_equal("clear_decoder_only"),
              "event_reset_leaves_all_semantic_logits_identical": logits_equal("clear_event_only"),
              "removing_just_generated_characters_restores_exact_operation": results["prefix_only"]["exact_operation"],
              "removing_all_generation_restores_exact_operation": results["no_generation"]["exact_operation"],
              "equal_length_neutral_output_restores_exact_operation": results["neutral_output"]["exact_operation"]}
    return {"seed": seed, "checkpoint_sha256": sha(checkpoint), "model_identity": model.model_identity(),
            "recorded_prefix_reproduced": reproduced, "manual_full_state_max_absolute_difference": manual_difference,
            "session_correction_result": correction_result, "records_after_wrong_key_commit": committed_records,
            "conditions": results, "replay_traces": replay_traces, "causal_comparisons": causal}


def language_reference_gap():
    path = HERE / "evidence" / "LANGUAGE_PRESERVATION.json"
    data = json.loads(path.read_text())
    full = data["references"]["original_connected_cog"]
    local = data["references"]["parent_local_only"]
    rows = []
    for row in data["runs"]:
        if row["training"]["arm"] != "memory":
            continue
        metrics = row["metrics"]
        rows.append({"name": row["name"], "mean_nll": metrics["mean_nll"],
                     "character_perplexity": metrics["character_perplexity"],
                     "nll_change_vs_full_original": metrics["mean_nll"] / full["mean_nll"] - 1,
                     "perplexity_change_vs_full_original": metrics["character_perplexity"] / full["character_perplexity"] - 1,
                     "nll_change_vs_local_parent": metrics["mean_nll"] / local["mean_nll"] - 1,
                     "declared_local_only_preservation_gate_passed": row["preservation_gate"]["passed"]})
    return {"source_sha256": sha(path), "full_original_nll": full["mean_nll"],
            "full_original_character_perplexity": full["character_perplexity"],
            "local_only_parent_nll": local["mean_nll"],
            "local_only_parent_character_perplexity": local["character_perplexity"], "memory_runs": rows,
            "interpretation": "The declared local-only preservation gate passes, but the fused local path remains worse than the full original native model. The two baselines must remain distinct."}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = HERE / "evidence" / "INTERACTION_RESULTS.json"
    original = json.loads(source.read_text())
    with tempfile.TemporaryDirectory(prefix="interaction-diagnostic-") as directory:
        rows = [diagnose_seed(seed, original[str(seed)], Path(directory)) for seed in SEEDS]
    conditions = rows[0]["conditions"].keys()
    summary = {condition: {"exact_operations": sum(row["conditions"][condition]["exact_operation"] for row in rows),
                           "entity_correct": sum(row["conditions"][condition]["entity_correct"] for row in rows),
                           "total_seeds": len(rows),
                           "predicted_entities": [row["conditions"][condition]["named_prediction"]["entity"] for row in rows]}
               for condition in conditions}
    code_paths = [Path(__file__), HERE / "fusion/model.py", HERE / "fusion/engine.py", HERE / "fusion/session.py"]
    output = {"schema": "cog-fusion-interaction-diagnostic-v1", "time_utc": datetime.now(timezone.utc).isoformat(),
              "status": "post-exposure no-fit causal diagnostic on one correction sentence",
              "target_sentence": TARGET, "expected_operation": list(EXPECTED),
              "expected_named_operation": named_operation(EXPECTED), "source_interaction_sha256": sha(source),
              "source_sha256": {str(path.relative_to(HERE)): sha(path) for path in code_paths},
              "runtime": {"torch": torch.__version__, "threads": 1, "device": "cpu"},
              "summary": summary, "seeds": rows, "language_reference_gap": language_reference_gap(),
              "limits": ["One already exposed sentence and interaction prefix across three trained seeds; not a new held-out qualification panel",
                         "No weights are fitted or changed; resets are diagnostic interventions, not an admitted repair",
                         "Deleting output tokens changes context length as well as content; prefix-only and equal-length neutral controls separate these effects partially",
                         "The decoder controller is separate from local backbone state; answer tokens still pass through the backbone",
                         "No serving-performance measurements are made while other research jobs run"]}
    destination = HERE / "evidence" / "INTERACTION_DIAGNOSTIC.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "summary": summary,
                      "language_reference_gap": output["language_reference_gap"]}, indent=2))


if __name__ == "__main__":
    main()
