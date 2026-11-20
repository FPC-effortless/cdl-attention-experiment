import json
from collections import defaultdict
from pathlib import Path

records = json.loads(Path("evidence/results.json").read_text())
assert len(records) == 24
groups = defaultdict(list)
for row in records:
    groups[row["entities"], row["mode"]].append(row)

def ratio(rows, cohort, num, den):
    return sum(r["cohorts"][cohort][num] for r in rows) / sum(
        r["cohorts"][cohort][den] for r in rows)

def passed(row):
    tests = row["cohorts"]
    if not all(tests[c]["state_correct"] / tests[c]["n"] >= threshold
               for c, threshold in [("familiar", .95), ("held_pair", .85), ("recombine", .85)]):
        return False
    if row["mode"] in ("exhaustive", "top8"):
        return True
    return (row["training"]["checks"] / row["training"]["examples"] <= 40
            and all(tests[c]["changed_shortlist_compatible"] / tests[c]["changed"] >= .95
                    for c in ("familiar", "held_pair", "recombine")))

print("| Entities | Training | Familiar | Held pairs | Recombined | Checks/example | Seconds/seed | Seeds passing applicable gates |")
print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for (n, mode), rows in groups.items():
    accuracy = [ratio(rows, c, "state_correct", "n") for c in ("familiar", "held_pair", "recombine")]
    checks = sum(r["training"]["checks"] for r in rows) / sum(r["training"]["examples"] for r in rows)
    seconds = sum(r["training"]["seconds"] for r in rows) / len(rows)
    fields = " | ".join(f"{x:.2%}" for x in accuracy)
    print(f"| {n} | {mode} | {fields} | {checks:.2f} | {seconds:.2f} | {sum(map(passed, rows))}/3 |")

print("\n| Entities | Training | Pre-fallback support | Post-fallback support | Rescued among initial misses | Novel verb state accuracy | Novel alias state accuracy |")
print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
for (n, mode), rows in groups.items():
    totals = {key: sum(r["training"][key] for r in rows)
              for key in ("examples", "initial_supported", "supported", "rescued")}
    missed = totals["examples"] - totals["initial_supported"]
    rescued = f'{totals["rescued"] / missed:.2%}' if missed else "—"
    novel = [ratio(rows, c, "state_correct", "n") for c in ("novel_verb", "novel_alias")]
    print(f'| {n} | {mode} | {totals["initial_supported"]/totals["examples"]:.2%} | '
          f'{totals["supported"]/totals["examples"]:.2%} | {rescued} | {novel[0]:.2%} | {novel[1]:.2%} |')
