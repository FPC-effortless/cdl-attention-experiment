# COG-BIND-001E: broader wording training transfers, but does not qualify
## Decision
Jointly training the same encoder, record selector and reader on varied wording
substantially improved two previously unused synthetic fact/question forms.
Strict family accuracy went from 4/768 to 267/768 on HELDX (0.5%→34.8%) and
from 21/768 to 432/768 on HELDY (2.7%→56.3%). Both gains occurred in all
three matched seeds. Familiar strict success fell from 634/768 to 596/768
(82.6%→77.6%). Neither arm qualified in any seed. This is a win for the
training regimen on this synthetic task, not an attention-replacement win.

## Matched comparison
Both arms use the same exact B pooled implementation, with 29,352 learned
scalars each, and no inherited model weights. Three paired initializations,
the same examples/labels/schedule, the same 1024 updates and 81,920 total
actor row exposures per fit, equal padded tensor shapes. The diverse arm
replaces half the baseline phrasing with matched alternate wording. Both
train selector and decoder with the same TRAIN-only support and answer losses.
No evaluation feedback, checkpoint selection or tuning. Loss over the last
32 updates averaged 0.00165–0.06608 for baseline and 0.03399–0.05446 for
diverse; the arms were not both broadly robust on CAL.

Before fitting, v1 was rejected because the baseline had not seen some
evaluation words. V2 gives both arms the same 24-token neutral TRAIN-only
filler in the same slot; independent scheduled exposure checks confirmed
all 84 evaluation words appeared in every arm and seed. The diverse arm
received 0.80–0.84% more nonpadding input tokens. Exact wording patterns,
word frequencies and roles differ, so this is a broader wording exposure
package, not word order alone or a comparison at equal token count.

## Held results
Strict family success requires all 16 target views to select the unique
correct record and emit the unique correct one of eight values. The
known-source value column uses the exact support only in offline scoring.
Columns are baseline / diverse. One entry summarizes 768 model–family
evaluations (256 paired groups repeated across three seeds), not 768
independent semantic cases. The fresh rectangles and exact prompts exclude
all B/C/D ancestors, although all 144 held entity–relation keys appeared
individually in ancestor roles. BASE, HELDX and HELDY are paired renderings.
Their two new fact/query pairs were not used for fitting.

| Wording | Strict native joint | Strict correct record selected | Strict value with correct record |
| --- | ---: | ---: | ---: |
| BASE, familiar | 634 / 596 | 635 / 596 | 767 / 768 |
| HELDX, unseen template | 4 / 267 | 83 / 451 | 37 / 440 |
| HELDY, unseen template | 21 / 432 | 66 / 451 | 323 / 736 |

| Seed | BASE | HELDX | HELDY |
| --- | ---: | ---: | ---: |
| 7371 | 189 / 188 | 3 / 145 | 10 / 127 |
| 7471 | 233 / 226 | 0 / 90 | 9 / 204 |
| 7571 | 212 / 182 | 1 / 32 | 2 / 101 |

On matched strict targets, diverse vs baseline counts of rescues/losses
were 83/121 on BASE, 265/2 on HELDX, and 419/8 on HELDY. No seed met
all predefined held criteria (95% joint target rows, 90% complete target
families, 95% joint anchor rows in each wording/value stratum). A separate
unfamiliar CAL form remained weak: 1/192→6/192 strict joint families.
All 30 packets are sealed and model-free scorers reconstruct the raw ranking
and eight-way decisions with exact tie rejection. The exact typed reference
resolves every candidate/truth pair; that is a fixture check, not learned.

## What this changes
HELDY's known-source value was 736/768 (95.8%) under diverse training but its
joint value was only 432/768 (56.3%); selection is a major limit on this form.
In HELDX the known-source value was only 440/768, so there is also a separate
encoding/readout problem. Both facts and questions change in these template
pairs: the study does not isolate one wording edit at a time. Do not promote
the architecture, scale context, or treat the exposed templates as new tests.

The next fair comparison is a capable Transformer control and the candidate
at the same parameter budget, training exposure and task contract on new
evaluation groups/wording. Use separate selection and value diagnostics and
measure speed only when competence is comparable, with matched hardware and
complete request costs. There is no Transformer or inference-efficiency
measurement in this run.
