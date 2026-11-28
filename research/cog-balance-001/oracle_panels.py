"""Input-only grammar and constant-zero baselines on the same BALANCE dev panels."""
import argparse
import json
import re
from pathlib import Path

import torch

from panels import evaluate


def grammar_answer(prompt):
    """Decode the visible claims and the question, including the held grammar."""
    if prompt.endswith(b"="):
        history, question = prompt[:-1].split(b"#")
        facts = dict(re.findall(rb"!([a-d])([0-3])", history))
        (subject,) = re.findall(rb"[a-d]", question)
        primitives = re.findall(rb"[+0-]", question)
        assert len(facts) == 4 and len(primitives) == 2
        value = int(facts[subject])
        for primitive in primitives:
            if primitive == b"+":
                value = (value + 1) % 4
            elif primitive == b"-":
                value = (value - 1) % 4
        return ord(str(value))

    statements = prompt.split(b". ")
    query = statements.pop()
    subject = re.search(rb"for ([A-Z][a-z]+)\?$", query).group(1)
    facts = {}
    for clause in statements:
        # Reliable source cues come from the textual specification.
        m = re.fullmatch(rb"The log lists (\w+) at ([0-3])", clause)
        if m is None:
            m = re.fullmatch(rb"(\w+) has code ([0-3]) on file", clause)
        if m is not None:
            name, value = m.groups()
            facts[name] = value
            continue
        m = re.fullmatch(rb"The record assigns ([0-3]) to (\w+)", clause)
        if m is not None:
            value, name = m.groups()
            facts[name] = value
    if len(facts) != 4:
        raise ValueError("Expected four source-qualified facts")
    return ord(facts[subject])


class DecisionRule(torch.nn.Module):
    def __init__(self, use_grammar):
        super().__init__()
        self.use_grammar = use_grammar
        self.device_marker = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, inputs):
        logits = torch.zeros((*inputs.shape, 288), device=inputs.device)
        for i, row in enumerate(inputs.tolist()):
            prompt = bytes(row).rstrip(b"\x00")
            answer = grammar_answer(prompt) if self.use_grammar else ord("0")
            logits[i, len(prompt) - 1, answer] = 1.0
        return logits


def audit(worlds=64, seed=610003, learned_metrics=None):
    """The oracle has hand-written knowledge of all templates and operator rules."""
    outputs = {}
    for rule in ("grammar", "constant_0"):
        raw = evaluate(DecisionRule(rule == "grammar"), seed, worlds=worlds)
        fields = ("sha256", "worlds", "answers", "correct", "both_correct", "answers_changed")
        outputs[rule] = {panel: {k: row[k] for k in fields} for panel, row in raw.items()}
    for panel, row in outputs["grammar"].items():
        assert row["both_correct"] == worlds and row["correct"] == 2 * worlds, panel
        assert row["sha256"] == outputs["constant_0"][panel]["sha256"], panel
        if learned_metrics is not None:
            assert row["sha256"] == learned_metrics["readouts"]["matched_data"]["panels"][panel]["sha256"], panel
    return {"seed": seed, "worlds_per_panel": worlds, "checked_learned_panel_hashes": learned_metrics is not None,
            "rules": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-metrics", type=Path)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--seed", type=int, default=610003)
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = None if args.learned_metrics is None else json.loads(args.learned_metrics.read_text())
    print(json.dumps(audit(args.worlds, args.seed, reference), indent=2))
