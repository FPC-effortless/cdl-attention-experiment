"""Check what information reaches the protected-write controller at the fact digit."""
import sys
from pathlib import Path

from trace_fault import ids, json, loadstudy, torch


@torch.inference_mode()
def audit(model, template, keys):
    prefix = "00000"
    texts = [prefix + template.format(key=k, digit=d) for d in range(4) for k in keys]
    tokens = ids(texts)
    _, _, (role, key, value) = model.parent.local(tokens, model.parent.core.emb)
    groups = []
    for digit in range(4):
        batch = slice(4*digit, 4*digit+4)
        arrays = [head[batch,-1] for head in (role, key, value)]
        groups.append({"digit": digit, "windows": [s[-5:] for s in texts[batch]],
                       "max_controller_logit_change_across_entities": max(float((a-a[:1]).abs().max()) for a in arrays),
                       "roles": arrays[0].argmax(-1).tolist(), "slots": arrays[1].argmax(-1).tolist()})
    return groups


if __name__ == "__main__":
    torch.set_num_threads(1)
    model, arm, scale, seed, sha = loadstudy(sys.argv[1])
    names = ["Ada", "Ben", "Cia", "Dan"]
    out = {k:audit(model, pattern, keys) for k, pattern, keys in [
        ("symbol", "!{key}{digit}", list("abcd")),
        ("log", "The log lists {key} at {digit}", names),
        ("rumor", "A rumor puts {key} at {digit}", names),
        ("file", "{key} has code {digit}", names)]}
    Path(sys.argv[2]).write_text(json.dumps(dict(seed=seed, checkpoint_sha256=sha, **out), indent=2))
