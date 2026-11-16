"""New 017 reference guard. Pure decision contract, not trained policy."""
import math
TERMINALS = frozenset((ord('.'), ord('?')))
def select_operation(raw_logits, byte, guarded=True):
    if len(raw_logits) != 6 or not all(math.isfinite(v) for v in raw_logits):
        raise ValueError('Exactly six finite operation logits required')
    if guarded and byte not in TERMINALS:
        return 0
    return max(range(6), key=lambda i: raw_logits[i])

def guard_torch(raw_operation_logits, bytes_, guarded=True):
    """Outputs effective logits. Keep raw logits for all training losses/audits."""
    import torch
    if raw_operation_logits.shape[-1] != 6 or raw_operation_logits.shape[:-1] != bytes_.shape:
        raise ValueError('shape mismatch')
    if not bool(torch.isfinite(raw_operation_logits).all()):
        raise ValueError('nonfinite raw logits')
    if not guarded:
        return raw_operation_logits
    terminal = (bytes_ == ord('.')) | (bytes_ == ord('?'))
    no_op = torch.full_like(raw_operation_logits, -float('inf'))
    no_op[..., 0] = 0
    return torch.where(terminal[..., None], raw_operation_logits, no_op)
