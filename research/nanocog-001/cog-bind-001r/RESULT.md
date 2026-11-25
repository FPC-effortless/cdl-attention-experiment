# COG-BIND-001R: a separate item reader catches known-form errors but does not transfer reliably

## Result
On held passages with familiar distractor wording, the learned agreement check blocked **all 226** wrong-item answers that the existing evidence check would have emitted from the correct record. It preserved **all 9,248** exactly correct answers already accepted. On the novel distractor wording, it blocked **6,533 of 6,598** wrong-item answers, but also blocked **1,354 of 2,321** correct ones and still agreed with the wrong item in **65** cases. The learned veto is not reliable on novel forms.

On every valid held row, the frozen router selected the right record, including under novel wording. The severe drop is in extracting the item from that record, not finding the record. There are also evidence-check errors on invalid inputs that an item-agreement guard cannot generally resolve.

## Design and scope
Freeze the three diverse 001O routes and their respective learned 001Q structured gates. Fit one independent 5,505-parameter token-span reader per seed, from raw text in a single clause, with no question, native logits, true record or test span as reader inputs. TRAIN provides span labels for 62,976 physical transfer appearances, deduplicated to 3,044 distinct clauses, with 16 passes. The actor independently scores every actual clause so the sealed run can diagnose both known and selected records later; the deployed-style decision consults only the frozen selected clause. Agreement can only veto an earlier accepted answer, not choose a new record, change the item, or emit a previously refused answer.

There are 256/64/128 disjoint TRAIN/DEV/HELD families, 82 rows per family. Each held view contains 3,328 valid and 7,168 invalid rows per seed. Each studied clause with a distractor includes two eligible item tokens. Swapping their payload and locative roles keeps the exact tokens and supporting slot and changes the valid answer. First-item and last-item readers get half of the held correct-witness answers by design. Both candidate items follow the same article; the original data draft with a simpler article shortcut was held before model use. The matched clean view removes the distractor without changing the task. These are controlled locative constructions over the same known vocabulary, not unconstrained natural language or novel item identities. Some old individual alias links are reused; canonical transfer/event combinations, actor rows and Q links are fresh per the audit.

## Held results across three checkpoints
Each view reports 9,984 valid and 21,504 invalid model–row observations over 128 distinct families repeated across seeds.

| View | Native exact before Q | Q exact accepted → learned-veto exact accepted | Q wrong-item emissions → after veto | Q invalid emissions → after veto |
| --- | ---: | ---: | ---: | ---: |
| Clean | 9,984 | 9,914 → 9,914 | 0 → 0 | 64 → 64 |
| Familiar distractor | 9,728 | 9,248 → 9,248 | 226 → 0 | 209 → 191 |
| Novel distractor | 2,638 | 2,321 → 967 | 6,598 → 65 | 420 → 45 |

The learned reader alone, scored at the actual selected (also correct on these valid cases) witness, was correct on **9,984/9,984** familiar-distractor rows and **7,030/9,984** novel-distractor rows. In the latter view it flipped the answer as required on both sides of **3,309/4,992** valid pairs; the native reader did so on **259/4,992**. Despite this diagnostic gain, using agreement rather than substitution left just **47** complete valid pairs accepted and exact, down from **189** with Q alone. The study does not authorize replacing native answers with the learned reader.

## Simpler controls and decision
The exact actor-derived locative rule excludes the item immediately following “by a”; within this restricted grammar it reads every valid held item correctly. It is an engineered grammar reference, not evidence of general language understanding. It removed all 6,598 wrong-item emissions under novel wording without losing any of the 2,321 correct accepted cases, but still admitted 106 invalid cases. The DEV-selected item-confidence rule blocked just 346 novel wrong items while losing 325 correct accepted ones. This helps separate genuine payload extraction from generic refusal. Low residual error counts from a veto are not enough if correct coverage drops.

Do not treat learned agreement as the default on unseen wording. The evidence points to teaching the item reader the actual payload role across unfamiliar constructions and testing it separately from agreement; keep the exact grammar rule as an explicit control and the frozen record selector as a diagnostic. The independent oracle comparison should establish span and pair transfer before another whole-system score. The original evidence gate still makes invalid admissions even with the exact local grammar rule. Extra reader latency was only measured in the offline all-clause evaluation, not production selected-clause throughput. No architecture-wide or deployment claim.

Three span fits and 12 sealed prediction bundles covering 110,208 prompt rows; independent raw recount. Frozen O/Q weights were unchanged and no experimental reruns occurred.

Data pin `9e16cefce2ac86b9527fccde87be610822693988c34249e6f75acef4ef988601`; score `cc872a25f6100479a2d7356af7081a54d1fbcc2589b6d59d3d80723e87f39b2f`.
