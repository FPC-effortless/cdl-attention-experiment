# COG Adaptive Repair 002

The revision generator previously repeated the target person in every rumor. A question-blind rule that read the repeated person’s official entry scored 64/64 on each original English panel. The new English training conditions give every person a rumor and an official entry. Identical-history pairs require different answers when the named person changes. All denominators below are both-correct pairs unless labeled otherwise.

## Whole-model continuations

Each row uses three seeds and 64 shared evaluation worlds per seed (192 model–world evaluations, 64 unique worlds). C is the slow-state parent, H the competitive geometric model, and R the new bounded contextual byte-copy head. Every seed received 256 optimizer updates on 2,048 question streams and 32,768 WikiText replay targets. Final-only, trace, paired and trace+paired conditions redistribute the cognitive loss on identical data within a generator.

| Model | Data | Supervision | Symbolic retrieval /192 | Balanced English revision /192 | Same history, new person /192 | New names and wording /192 | Mean dev BPB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| C | balanced | final | 1 | 2 | 0 | 0 | 3.6567 |
| C | balanced | pair | 1 | 2 | 0 | 0 | 3.6735 |
| C | balanced | trace | 3 | 2 | 3 | 0 | 3.7552 |
| C | balanced | trace_pair | 13 | 3 | 4 | 3 | 3.7210 |
| C | original | final | 1 | 1 | 0 | 0 | 3.6680 |
| C | original | trace_pair | 14 | 8 | 1 | 3 | 3.6967 |
| GQA | balanced | final | 0 | 0 | 0 | 0 | 3.7761 |
| GQA | balanced | trace_pair | 11 | 3 | 0 | 1 | 3.8042 |
| GRU | balanced | final | 0 | 0 | 0 | 0 | 3.3587 |
| GRU | balanced | trace_pair | 19 | 0 | 5 | 0 | 3.4144 |
| H | balanced | trace_pair | 15 | 8 | 1 | 1 | 3.7943 |
| R | balanced | final | 53 | 36 | 0 | 0 | 3.7415 |
| R | balanced | trace_pair | 48 | 34 | 0 | 2 | 3.7417 |

R’s additional read can change the answer. On the independent symbolic addressing test R with combined supervision scored 75/384 on value edits, 30/384 on fact permutations, and 3/384 on query swaps. Suppressing its read gave zero on all three. A question-blind rule reading key a scored 26/128 on single-fact edits. Useful question-conditioned binding is still unresolved for this neural design.

## COG with typed memory

A separately supervised 8,627-parameter parser classifies completed English clauses and labels subject and value. The frozen C backbone handles language modeling. Confident recognized writes store an exact byte under a name in 32 slots. At a recognized read the stored value overrides the LM’s top prediction. All three parser seeds used the same frozen C checkpoint. The generic word tokenizer, explicit key/value store, and privileged role and action annotations are architectural and supervision differences from the whole-model comparison.

The narrow parser learned two English forms. Each of three parser seeds got 64/64 on familiar revisions, revisions with new names, same-history different-subject queries and both irrelevant-edit tests. Disabling the reader gave 0/64 on the familiar revision and different-subject tests. The narrow and broader 18-family versions both got 0/64 on the earlier held wording. Across 32,768 WikiText byte targets per parser seed, zero reader overrides gave the same 4.2313 BPB as the frozen parent. An exact FIFO check retained the latest 32 of 40 names and the narrow reader abstained for all eight evicted names.

## Sealed transfer and the released interface

Eight wording families were sealed before scoring the three new 18-family taggers. Each complete world requires correct original, changed evidence, and changed-person answers. The same 512 scenarios appeared with familiar and unfamiliar names for 1,024 renderings. Narrow whole-wrapper results by seed were 0, 0 and 0 complete. The 18-family wrappers solved 0, 0 and 4. Across six reader seeds, every one of the 117 incorrect direct memory answers matched the named person’s weak claim. The frozen base alone solved zero complete worlds.

The portable default therefore enforces a narrow, explicit input contract derived from the training signatures. Both writes and reads must agree with those forms and with the parser. Unsupported completed clauses clear the typed memory, so later questions cannot use a stale value. The raw experimental variants remain available. In a post-hoc check the guarded reader abstained from memory on all 3,072 sealed prompts at each seed. That is a rejection result. Under fallback, the frozen language model can still predict a wrong byte.

## Natural text and control readers

A trained extractive pointer with a 49-byte window reached 128/128 per seed on the familiar balanced English edits and question swaps. With only 21 bytes and no explicit trigram matching, the swap scores were 69/128, 60/128 and 51/128. On eleven natural passages with two manually written questions and two edited versions, none of the seed-9731 whole-model readers solved all six answers for a passage. The highest individual-answer result among these continuations was 29/66 for R; a question-blind digit baseline reached 33/66.

With extra real-text training, the extractive control learned questions generated from lexical anchors in Wikipedia passages. On 175 held training-partition passages free of shared 32-byte substrings with the control training windows, it solved 73, 58 and 79 four-answer suites across seeds. Matched template-only continuations solved zero. On the 11 manually questioned dev passages, it solved 1, 0 and 0 six-answer suites.

The pointer cannot generate an arithmetic answer outside its input. The old held cancellation task can be solved by copying the requested digit. Neither result measures broad compositional reasoning.

The frozen weights, evaluation panel and training digests, empirical baselines and per-seed readouts identify each comparison. The checkpoint package includes reproducible inference for the supported English reader and a neural R comparison.
