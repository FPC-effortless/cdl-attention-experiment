"""Run the trained COG record reader on one UTF-8 prompt at a time."""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for directory in (HERE / "source", HERE.parent / "cog-balance-001", HERE.parent / "unified-search-002"):
    if directory.is_dir():
        sys.path.insert(0, str(directory))
from integrated import WholeCOG
from structured_memory import StructuredCOG, Tagger


def load_reader(root, variant):
    if variant == "copy":
        from repair_model import build_repair_model
        reader = build_repair_model(9731)
        reader.load_state_dict(torch.load(root / "copy_R_trace_pair_9731.pt", weights_only=True))
        return reader.eval()
    parent = WholeCOG("SS_native_native", 9731)
    parent.load_state_dict(torch.load(root / "slow_state_C_9731.pt", weights_only=True))
    weights = "narrow" if variant == "guarded" else variant
    payload = torch.load(root / f"reader_{weights}_9731.pt", weights_only=True)
    tagger = Tagger({word.encode("ascii"): index for word, index in payload["vocab"].items()})
    tagger.load_state_dict(payload["weights"])
    if variant == "guarded":
        from schema_guard import GuardedStructuredCOG
        return GuardedStructuredCOG(parent.eval(), tagger.eval()).eval()
    return StructuredCOG(parent.eval(), tagger.eval()).eval()


@torch.inference_mode()
def answer(model, prompt):
    raw = prompt.rstrip().encode("utf-8")
    if not raw:
        raise ValueError("A non-empty prompt is required")
    from structured_memory import StreamingStore
    read = None
    if hasattr(model, "tagger"):
        from schema_guard import GuardedStructuredCOG, GuardedStore
        store = GuardedStore(model.tagger) if isinstance(model, GuardedStructuredCOG) else StreamingStore(model.tagger)
        for byte in raw:
            read = store.step(byte)
    state = None
    ids = torch.tensor([list(raw)], dtype=torch.long)
    for block in ids.split(128, dim=1):
        logits, state = model(block, state, carry=True)
    token = int(logits[0, -1].argmax())
    if read is not None and token != read:
        raise ArithmeticError("streaming receipt disagrees with fused output")
    source = "keyed memory" if read is not None else "base language model"
    if not hasattr(model, "tagger"):
        source = "neural copy model"
    return {"predicted_id": token, "printable_ascii": chr(token) if 32 <= token <= 126 else None,
            "source": source,
            "memory_value": chr(read) if read is not None else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Statements and question in one string")
    parser.add_argument("--variant", choices=("guarded", "narrow", "diverse", "copy"), default="guarded")
    parser.add_argument("--checkpoint-dir", type=Path, default=HERE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(answer(load_reader(args.checkpoint_dir, args.variant), args.prompt)))
