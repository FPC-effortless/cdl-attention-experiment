# GEN-001: Can the original 1M language model generate the visible fact?

We used the unchanged 993,574-parameter TinyShakespeare COG checkpoint and fine-tuned two copies with different training data order. Both trained copies and the original produced characters from all 65 original choices, up to the first period or eight characters. No answer choices, categorical replacement head, lookup, output trimming or forced spelling. Exact answers include the full four-letter word and period.

**0/2 adapted runs met all preregistered gates.** This is a small, immediate synthetic fact task; it is not a test of archival memory or general language modeling.

| Held result | Original | Adapted 1 | Adapted 2 |
|---|---:|---:|---:|
| familiar: exact answers | 0/256 | 254/256 | 254/256 |
| familiar: both paired answers exact | 0/128 | 126/128 | 126/128 |
| novel: exact answers | 0/256 | 251/256 | 248/256 |
| novel: both paired answers exact | 0/128 | 123/128 | 121/128 |
| unrelated: exact answers | 0/256 | 111/256 | 100/256 |
| unrelated: both paired answers exact | 0/128 | 38/128 | 33/128 |
| no fact: exact answers | 0/256 | 256/256 | 256/256 |
| no fact: both paired answers exact | 0/128 | 128/128 | 128/128 |
| distractor: exact answers | 0/256 | 77/256 | 84/256 |
| distractor: both paired answers exact | 0/128 | 22/128 | 27/128 |

**The failure was subject binding, not formatting.** Both adapted runs generated well-formed output on every held row, but answered for the wrong subject with a concrete task value in 145/256 and 156/256 prompts. In 144 and 156 of those prompts, the output copied the value of the subject in the prompt. By contrast, both returned the taught missing answer on all 256 prompts where the subject matched but the relation did not. The positive one-fact panels alone cannot establish binding: simply copying the last value gets all of them right. On the longer two-fact diagnostic, the model was exact on only 25/128 and 24/128 when the relevant fact came first.

The held novel prompt recombined fact and question wording seen separately during adaptation with the same prompt length. In the two missing-fact controls, the taught target was the literal string ‘none.’, not a general abstention mechanism. The two-fact distractor view uses a longer prompt than TRAIN and is descriptive only. The exact literal parser is a fixture check, not a model output; last-visible-word copying is a named shortcut control, not a learned baseline.
