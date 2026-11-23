"""Pipeline tests with a decoder whose next write requires an emitted period."""
import copy

import pytest
import torch

from evaluate import candidate_inputs, evaluate_panel, rollout_batch
from fusion.engine import pad_texts


class OutputSensitiveFixture:
    chars = tuple("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    names = ("Amara.", "Bela.", "Cora.", "Davi.")

    def eval(self):
        return self

    def init_state(self, batch):
        return {"periods": torch.zeros(batch, 1, dtype=torch.long),
                "answer_index": torch.zeros(batch, 1, dtype=torch.long)}

    def encode_texts(self, texts):
        return pad_texts(texts, self.chars)

    def encode_event(self, tokens, lengths, state):
        texts = ["".join(self.chars[token] for token in tokens[index, :int(length)].tolist()).strip()
                 for index, length in enumerate(lengths)]
        labels = {"operation": [], "entity": [], "relation": [], "value": [],
                  "query_entity": [], "query_plan": []}
        for index, text in enumerate(texts):
            value = {"Amara owns the chest.": 0, "Bela owns the chest.": 1,
                     "Cora owns the chest.": 2, "Davi owns the chest.": 3}.get(text, 0)
            if text in ("Bela owns the chest.", "Davi owns the chest.") and state["periods"][index, 0] < 1:
                value -= 1  # Controlled failure if the earlier emitted period was not carried.
            neutral = text == "The clock ticks."
            for key, label in ("operation", 0 if neutral else 1), ("entity", 8), ("relation", 0), ("value", value), ("query_entity", 8), ("query_plan", 0):
                labels[key].append(label)
        widths = {"operation": 3, "entity": 16, "relation": 2, "value": 17,
                  "query_entity": 16, "query_plan": 3}
        logits = {}
        for key, width in widths.items():
            logits[key] = torch.full((len(texts), width), -10.)
            logits[key][torch.arange(len(texts)), torch.tensor(labels[key])] = 10.
        return torch.zeros(len(texts), 64), logits, {key: value.clone() for key, value in state.items()}

    def decode(self, tokens, state, values, masks, *, memory_enabled=None):
        state = {key: value.clone() for key, value in state.items()}
        logits = torch.full((*tokens.shape, len(self.chars)), -100.)
        prefix = tokens.size(1) == len("\nAnswer: ")
        for row in range(tokens.size(0)):
            if prefix:
                state["answer_index"][row, 0] = 0
            for position, token in enumerate(tokens[row].tolist()):
                char = self.chars[token]
                if not prefix:
                    state["answer_index"][row, 0] += 1
                    if char == ".":
                        state["periods"][row, 0] += 1
                answer = self.names[int(values[row, 0])]
                at = int(state["answer_index"][row, 0])
                nxt = answer[at] if at < len(answer) else "X"
                logits[row, position, self.chars.index(nxt)] = 100.
        return logits, state


def fixture_family(with_hold=False):
    worlds = []
    for first, second in ((0, 1), (2, 3)):
        names = ("Amara", "Bela", "Cora", "Davi")
        steps = []
        for turn, value in enumerate((first, second)):
            if turn and with_hold:
                steps.append({"type": "observe", "text": "The clock ticks.", "operation": 0,
                              "entity": 0, "relation": 0, "value": 16,
                              "post_event_ledger": [[8, 0, first]]})
            steps.append({"type": "observe", "text": names[value] + " owns the chest.",
                          "operation": 1, "entity": 8, "relation": 0, "value": value,
                          "post_event_ledger": [[8, 0, value]]})
            steps.append({"type": "ask", "text": "Who owns the chest?", "query_entity": 8, "query_plan": 0,
                          "answer_text": names[value] + ".", "supports": [[8, 0]], "evidence_values": [value]})
        worlds.append({"steps": steps})
    return {"family_id": "fixture", "kind": "correction", "wording": "test", "worlds": worlds}


def test_actual_generated_terminators_are_carried_into_later_writes():
    family = fixture_family()
    model = OutputSensitiveFixture()
    result = evaluate_panel(model, [family], batch_families=1)
    assert result["summary"]["answers_exact"] == 4
    assert result["summary"]["whole_session_pairs_exact"] == 1
    assert result["summary"]["meaningful_post_answer_operations_exact"] == 2
    assert result["summary"]["rates"]["query_support_joint_accuracy"] == 1
    reset = evaluate_panel(model, [family], reset_after_answer=True)
    assert reset["summary"]["answers_exact"] == 2
    assert reset["summary"]["whole_session_pairs_exact"] == 0
    assert reset["summary"]["meaningful_post_answer_operations_exact"] == 0
    # Wrong values at correctly selected keys must not pass the support gate.
    assert reset["summary"]["support_keys_exact"] == 4
    assert reset["summary"]["support_exact"] == 2


def test_candidate_cannot_receive_labels_and_poisoned_labels_do_not_change_generation():
    family = fixture_family()
    inputs = [candidate_inputs(world) for world in family["worlds"]]
    assert all(set(step) == {"type", "text"} for world in inputs for step in world)
    contaminated = copy.deepcopy(inputs)
    contaminated[0][0]["operation"] = 1
    with pytest.raises(ValueError, match="only text"):
        rollout_batch(OutputSensitiveFixture(), contaminated)
    original = evaluate_panel(OutputSensitiveFixture(), [family])
    poisoned = copy.deepcopy(family)
    for world in poisoned["worlds"]:
        for step in world["steps"]:
            if step["type"] == "ask":
                step["answer_text"] = "Never supply this target to the model."
    altered = evaluate_panel(OutputSensitiveFixture(), [poisoned])
    outputs = lambda result: [step["generated"] for row in result["rows"] for step in row["steps"] if step["type"] == "ask"]
    assert outputs(original) == outputs(altered)
    assert altered["summary"]["answers_exact"] == 0


def test_batched_and_individual_rollout_consume_identical_output_histories():
    family = fixture_family()
    inputs = [candidate_inputs(world) for world in family["worlds"]]
    batched = rollout_batch(OutputSensitiveFixture(), inputs)
    for row, world in enumerate(inputs):
        single = rollout_batch(OutputSensitiveFixture(), [world])
        assert batched["traces"][row] == single["traces"][0]
        for key in batched["final_state"]:
            assert torch.equal(batched["final_state"][key][row:row+1], single["final_state"][key])
    assert batched["final_state"]["periods"].tolist() == [[2], [2]]


def test_hold_events_cannot_inflate_meaningful_postanswer_operation_gate():
    result = evaluate_panel(OutputSensitiveFixture(), [fixture_family(with_hold=True)])
    summary = result["summary"]
    assert summary["post_answer_operations_total"] == 4
    assert summary["meaningful_post_answer_operations_total"] == 2
    assert summary["immediate_post_answer_operations_total"] == 2
    assert summary["meaningful_immediate_post_answer_operations_total"] == 0
    assert summary["rates"]["meaningful_immediate_post_answer_operation_accuracy"] is None
