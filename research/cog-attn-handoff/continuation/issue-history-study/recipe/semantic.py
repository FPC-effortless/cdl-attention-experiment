"""Replay the previously pinned semantic verification checkpoint on authentic comments."""
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / "dataset.json").read_text())
    source = ROOT / "nli"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source / "model_quint8_avx2.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    inputs = {item.name for item in session.get_inputs()}
    scores = np.empty((len(rows), 3), np.float32)
    start = perf_counter()
    sentences = [row["text"][:2400] for row in rows]
    hypotheses = [
        "The author is announcing a decision to reopen this issue." if row["closed"]
        else "The author is announcing a decision to close this issue."
        for row in rows
    ]
    ordered = sorted(range(len(rows)), key=lambda i: len(sentences[i]))
    for offset in range(0, len(rows), 24):
        ids = ordered[offset:offset + 24]
        encoded = tokenizer([sentences[i] for i in ids], [hypotheses[i] for i in ids],
                            padding=True, truncation="only_first", max_length=128,
                            return_tensors="np")
        scores[ids] = softmax(session.run(None, {k: v for k, v in encoded.items() if k in inputs})[0], axis=1)
        if offset % 480 == 0:
            print(f"Scored {offset + len(ids)}/{len(rows)} comment hypotheses", flush=True)
    out = Path(os.environ.get("TRAINLAB_OUTPUT_DIR", ROOT))
    np.save(out / "semantic.npy", scores)
    timing = {"rows": len(rows), "inference_seconds": perf_counter() - start, "max_tokens": 128,
              "onnx_threads": options.intra_op_num_threads}
    (out / "semantic-timing.json").write_text(json.dumps(timing))
    (out / "metrics.json").write_text(json.dumps({"rows": len(rows)}))


if __name__ == "__main__":
    main()
