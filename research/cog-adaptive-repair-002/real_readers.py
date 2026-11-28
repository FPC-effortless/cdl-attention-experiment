"""Common raw-byte passage evaluation for the BALANCE and repaired checkpoints."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cog-balance-001"))
from panels import _batch_bytes
from pointer_control import natural


class Reader:
    """All 288 vocabulary entries compete for the next byte on the unmodified prompt."""
    def __init__(self, model=None, fixed=None):
        self.model = model
        self.fixed = fixed
        self.digest = hashlib.sha256()
        self.predictions = []

    @torch.inference_mode()
    def select(self, prompts):
        for prompt in prompts:
            self.digest.update(len(prompt).to_bytes(4, "little"))
            self.digest.update(prompt)
        if self.fixed is None:
            logits = self.model(_batch_bytes(prompts))
            if isinstance(logits, tuple):
                logits = logits[0]
            scores = logits[range(len(prompts)), [len(x) - 1 for x in prompts]]
            if not torch.isfinite(scores).all():
                raise ValueError("nonfinite answer logits")
            prediction = scores.argmax(dim=-1).tolist()
        else:
            prediction = [self.fixed] * len(prompts)
        self.predictions.append(prediction)
        return prediction, [-1] * len(prompts)


def evaluate_reader(model, corpus, fixed=None):
    reader = Reader(model, fixed)
    rows = natural(reader, corpus)
    for row, raw in zip(rows, reader.predictions):
        row.pop("selected_support", None)
        row["predicted_byte_ids"] = raw
    return {
        "prompt_sha256": reader.digest.hexdigest(),
        "answers_correct": sum(r["answers_correct"] for r in rows),
        "answers": sum(r["questions"] for r in rows),
        "passages": len(rows),
        "original_query_pair": sum(r["original_query_pair"] for r in rows),
        "all_six_correct": sum(r["all_six_correct"] for r in rows),
        "passage_records": rows,
    }


def load_model(arm, seed, weights, tagger_path=None):
    from scale import build_model
    if arm == "R":
        from repair_model import build_repair_model
        model = build_repair_model(seed)
    else:
        model = build_model("C" if arm == "structured" else arm, seed)
    model.load_state_dict(torch.load(weights,weights_only=True))
    if arm == "structured":
        from structured_memory import StructuredCOG, Tagger
        trained = torch.load(tagger_path,weights_only=True)
        tagger = Tagger({w.encode("ascii"):i for w,i in trained["vocab"].items()})
        tagger.load_state_dict(trained["weights"])
        model = StructuredCOG(model,tagger)
    return model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("C","GQA","GRU","R","H","structured","constant"))
    parser.add_argument("--seed", type=int, default=9731)
    parser.add_argument("--fixed", type=int, choices=range(4), default=0)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--tagger", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    a=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if a.arm == "constant":
        result=evaluate_reader(None,a.corpus,fixed=48+a.fixed)
    else:
        if not a.weights:
            parser.error("Trained model requires --weights")
        if a.arm == "structured" and a.tagger is None:
            parser.error("Structured model requires --tagger")
        result=evaluate_reader(load_model(a.arm,a.seed,a.weights,a.tagger),a.corpus)
        if a.arm == "structured":
            with a.tagger.open("rb") as stream:
                result["tagger_sha256"]=hashlib.file_digest(stream,"sha256").hexdigest()
        with a.weights.open("rb") as stream:
            result["weights_sha256"]=hashlib.file_digest(stream,"sha256").hexdigest()
    result["arm"], result["seed"] = a.arm, a.seed
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
