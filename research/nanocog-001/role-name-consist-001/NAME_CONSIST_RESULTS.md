# ROLE-NAME-CONSIST-001: matched-name margin regularization did not repair binding

Both arms trained the same 552-parameter adapter on exact matching
sequences from original TRAIN. A weight of 0.25 on bounded signed
answer-margin consistency was compared against weight 0, with answer
CE and the assignment foil objective identical. Both performed the same
forwards, used the same balanced-per-seed remap schedules and original
frozen backbone, and saw no held prompt or labels during fits.

| Adapter seed | Ordinary strict worlds, 0→.25 / 960 | All-eight, 0→.25 / 120 | Fragile second target, 0→.25 / 96 |
|---|---:|---:|---:|
| 1771 | 866→869 | 75→76 | 59→59 |
| 1871 | 862→864 | 75→76 | 56→56 |
| 1971 | 863→862 | 76→75 | 64→64 |

The frozen native ordinary reference was 840/960 worlds and 67/120
complete families. Native first/second targets: 473/480 and 367/480.
The weight-0.25 arm improved second but lost 34, 32 and 40 first-clause
answers. No seed in either arm cleared ≥912/960 and ≥108/120;
the first-clause guard and fragile target improvement criterion
also failed. No architecture or learned repair is promoted.

The enriched secondary panel confirmed persistent target fragility.
Native focal-second: 7/96 fragile and 96/96 robust. Both trained arms
improved fragile accuracy compared with frozen native, but the 0.25
loss changed **none of 2,304 deterministic top-1 choices** relative
to its matched zero-weight arms on this panel, nor their focal-second
block scores. Robust focal-second stayed 96/96. This isolates the
null benefit of the tested objective coefficient, not the independent
effect of the new paired schedule. Late mean TRAIN loss on the
regularizer itself also barely changed in the final 128 updates; that
observation neither proves why the loss failed nor excludes other
forms of relational supervision. Full-vocabulary and two-color
correctness agreed throughout.

The new ordinary and enriched panels are joint-combination disjoint
from 928 prior families and each other. This remains one familiar
synthetic grammar, 16 names and one known weak backbone. No
general-language, memory, or inference cost claim. The next step is a
stagewise frozen information audit, not another consistency weight.
A probe's recoverability will be separated from causal use.

Six exact TrainLab fits, 7,680 updates, 12 adapter-only checkpoints.
Twenty sealed prompt-only actors, 17,856 full-vocabulary prediction
rows. Independent recount reconciled all published metrics, confidence
intervals and decisions. No protected TEST, MAIN, paid compute, or
full-backbone fit. Native process times are not serving benchmarks.

- FIT ledger: `290ae7e56763ccba99d36fe3513fd2c0f34a3e98eebac58f33547c1b17aee68a`
- FIT seal: `da7e78286d8f77f684140a979a6f4b7dac5a6b34b010b500840f86a918065e10`
- Actor ledger: `814c6cb79cc7b30e0d5f4f18a6c1623494e1e9a6d04532b9ee955f4eb34f9c52`
- Actor seal: `ee4be7073c5686829970394f93c2e283d8fde2a6d36e91a9e3b1a9eca34bd4a9`
- Score aggregate: `cd22f5a5a899066db1e7cd13b56ec5f974d666e130060d0b5dd06008e59c6861`
- Independent actor audit: `192156461a0ed0ecd1562c24ca25e82a36854991c03e3fb4b1608d3219e8421b`
