# COG-ATTN-087–090: graded puzzle curriculum, Novice → Frontier

**State:** benchmark environment extension and reference validation; **NO neural COG-ATTN policy was trained or scored** in this cycle. The live COG-066 option-admission negative result from 086 is preserved separately. This program adds a staged curriculum and deliberately separates *difficulty specifications*, *measured reference complexity*, and *actual agent success*.

## Ten levels

| Level | Name | Maze dimension | Sudoku | Verified 8-puzzle optimal depth | Hanoi discs / minimum moves | Restricted chess forced mate |
| --- | --- | --- | --- | ---: | --- | --- |
| 1 | Novice | 7×7 | 4×4, 4 blanks | 2 | 2 / 3 | 1 White move |
| 2 | Easy | 9×9 | 4×4, 7 blanks | 4 | 3 / 7 | 1 White move |
| 3 | Foundation | 11×11 | 4×4, 9 blanks | 6 | 4 / 15 | 1 White move |
| 4 | Intermediate | 15×15 | 9×9, 35 blanks | 8 | 5 / 31 | 2 White moves |
| 5 | Challenging | 21×21 | 9×9, 42 blanks | 12 | 7 / 127 | 2 White moves |
| 6 | Advanced | 31×31 | 9×9, 47 blanks | 16 | 9 / 511 | 2 White moves |
| 7 | Expert | 45×45 | 9×9, 50 blanks | 20 | 12 / 4,095 | 3 White moves |
| 8 | Master | 65×65, shifted | 9×9, 53 blanks | 24 | 15 / 32,767 | 3 White moves |
| 9 | Extreme | 95×95, shifted | 9×9, 55 blanks | 28 | 18 / 262,143 | 4 White moves |
| 10 | Frontier | 151×151, shifted | 9×9, target 58 blanks | 31 | 22 / 4,194,303 | 4 White moves |

Levels 8–10 also feature **15-puzzle** scrambles of 24/48/96 legal walk steps. The 15-puzzle has a reversible solution witness, but **shortest solution depth is unknown**; scramble length must not be marketed as certified optimum. Sudoku's requested blanks may be unattainable under the bounded generator. Actual blanks and decision counts are included in the hidden reference metrics.

Chess is *restricted to legal K+Q vs K and K+R vs K* (no pawns, castling, promotions, real middlegame tactics, etc.). Depth 1–4 is independently certified by an adversarial minimax search of every legal Black reply, not full general chess. Full chess extreme qualifications require an external chess-rules engine and rated tactical suites and are **not implemented** here.

## Data and scoring

- **210** generated public challenges across 4 seeded cohort streams: development 8701/8707 and sealed reserved 9701/9707.
- Six families: `chess` 40, `sudoku` 40, `maze` 40, `sliding3` 38, `hanoi` 40, `sliding4` 12.
- 31-move 8-puzzle has **exactly two** canonical-goal states across the 181,440 reachable states. This benchmark assigns one to development and one to reserved. Two attempted extras are **excluded**, not silently duplicated.
- Hanoi is intentionally canonical and repeats across seeds. This is a procedure/scale test; its repeated states **must not** be counted as novel generalization evidence.
- All other input fingerprints are unique across both cohorts. The reserved panel cannot be used for acquisition or calibration.
- `results/cases.jsonl` holds **non-oracle** task observations and setup. `results/reference_metrics.jsonl` contains reference measurements for evaluation only and **must not** be supplied to an agent. `score_ladder.py` accepts independent action submissions and does not load reference metrics.
- Chess submissions at depth >1 must supply a **contingent state→move strategy**, not merely an attractive first move. The scorer verifies *all* Black replies recursively.
- The offline JSONL scorer intentionally **does not qualify** million-move Hanoi trajectories; these require a future streaming-action evaluator. A symbolic lower bound is not an executed solution.
- For mazes at levels 8–10, reference data includes an obstacle change that invalidates the original route. The scorer can compare a separately submitted changed-world route, but this is a two-phase static replan check, **not a fully interactive partial-observation agent**.

## Reproduce

From a checkout containing the existing `research/cog-081-086/puzzles.py` in its parent directory, with Python 3.12 (no third-party dependencies):

```bash
cd research/cog-081-086/extreme
python run_ladder.py --output results
python -m unittest test_ladder -v
sha256sum -c CASE_SHA256SUMS
```

To score predictions (JSONL containing IDs and proposed `actions` arrays; chess can use `strategy` mapping FENs to triples):

```bash
python score_ladder.py --cases results/cases.jsonl \
  --submissions /path/to/agent_predictions.jsonl --out /tmp/puzzle_score.json
```

### Research gates still open

The current reference metrics certify the generator and validators, **not architecture performance**. The next experimental run should fit a COG-ATTN policy through observation/action/reward, test on level-sealed puzzles, and compare to trained and untrained matched-parameter alternatives, literal tool/search controls, random controls and the inherited COG-066 option baseline. Charge generation separately from inference; measure full inference, exploration and verifier/recovery costs, success at each level, illegal actions, branches, generalization, and admission/rejection of inappropriate skills. Report abstention/timeout at Frontier rather than fabricating success.
