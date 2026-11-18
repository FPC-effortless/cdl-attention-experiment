# CSM-SKILL-002 — Goal-directed record assignment

Compare two constructive planners on all 256 mappings from four initially distinct records to four output slots. The 24 permutations preserve all inputs. The other 232 mappings repeat at least one value. Both planners see the entire goal, execute the same primitives, and must clear scratch. The control writes a final value whenever overwriting the destination preserves every still-needed value. When a cycle blocks progress it saves a value in scratch. The treatment may also invoke the admitted SKILL-001 SWAP when two misplaced records can both reach their final values in one call. Count planner dispatches and expanded primitive operations separately. Exact final state on every mapping qualifies the cost comparison. Wall time, learning the controller, and larger machines remain open.

| Goal family | Goals passed per arm | Primitive planner dispatches / primitives | SWAP planner dispatches / primitives |
| --- | ---: | ---: | ---: |
| Permutations | 24/24 | 121 / 121 | 88 / 124 |
| Repeated output values | 232/232 | 720 / 720 | 612 / 864 |

Both planners solve every goal without search. SWAP saves 141 dispatches while adding 147 primitive operations overall. One reason is that SWAP preserves both source values even when the goal permits discarding one. These are totals over the exhaustive goal roster, including the identity mapping. They exclude controller comparisons, acquisition and admission costs. A timing study should include controller and executor costs together, and compare a controller that checks whether the source values still need preserving before it selects SWAP.

Replay from this directory with Python 3.12 and its standard library.

```sh
python run.py > /tmp/csm-skill002.csv
diff -u scores.csv /tmp/csm-skill002.csv
```
