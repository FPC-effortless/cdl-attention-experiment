# Next phase — recurrent-depth memory reasoning

CASM-U v0 showed that changing the Q/K target alone is insufficient. The next experiment changes the model's effective computation while holding parameter count fixed.

## Hypothesis

A single weighted memory read is structurally inadequate for tasks that require composition, such as multi-hop reachability or repeated state updates. Reusing the same memory read/update block several times should provide an iterative latent computation path without adding parameters.

For working state `z_r` and memory candidates `M`:

```
z_0 = local_hidden
for r = 0 .. R-1:
    a_r = softmax(q(z_r) k(M)^T)
    m_r = sum_i a_ri v(M_i)
    z_{r+1} = z_r + gated(m_r) + shared_ffn(z_r + gated(m_r))
```

The same Q/K, value projection, memory gate, and FFN are reused at every recurrent step.

## Controlled variants

1. `qk-1step`: the current ordinary Q/K memory model, one retrieve/update step.
2. `qk-3step`: identical parameters and initialization, three shared retrieve/update steps.
3. `set-utility-3step`: the same three-step recurrent model, with training-only set-conditioned utility supervision for Q/K.

## Set-conditioned utility

For an answer position, first execute the full recurrent memory loop. Then remove candidate memory `i` from **all recurrent steps** and recompute the final answer NLL.

```
U_i = NLL(answer | all memories except i) - NLL(answer | all memories)
```

This is a leave-one-memory-out marginal contribution conditioned on the rest of the set. It captures redundancy and interaction better than CASM-U v0's isolated single-memory injection.

The counterfactual evaluator is detached. Q/K scores at each recurrent step are trained toward `softmax(U / tau)`. Runtime inference still uses ordinary Q/K only.

## Primary gates

- corrected six-task hard answer NLL;
- true autoregressive exact solve rate;
- balanced graph accuracy by class, not aggregate alone;
- state/associative performance as context grows;
- inference recurrence sweep (1, 3, 5 steps) to test whether additional test-time compute helps or overthinks;
- parameter count must remain identical across variants.

A lower teacher-forced NLL without improved generated answers is not considered a reasoning win.
