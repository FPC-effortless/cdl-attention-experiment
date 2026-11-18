"""Build and reconcile the claim-specific TrainLab adequacy contract for SKILL-006."""
import csv
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

from trainlab.execution_adequacy import validate_execution
from trainlab.research_adequacy import assess, recipe_identity
from trainlab.runner import digest, read_json, write_json


def development_summary(archive):
    with ZipFile(archive) as bundle:
        panel, = [p for p in bundle.namelist() if p.endswith("/output/panel.csv")]
        with bundle.open(panel) as raw:
            rows = list(csv.DictReader(TextIOWrapper(raw, encoding="utf-8")))
    goals = {row["goal"] for row in rows}
    if len(goals) != len(rows):
        raise ValueError("duplicate four-record development goal")
    arms = ("primitive", "swap", "rotate")
    best_fixed = min(sum(int(row[arm]) for row in rows) for arm in arms)
    oracle = sum(min(int(row[arm]) for arm in arms) for row in rows)
    wins = {
        arm: sum(int(row[arm]) < min(int(row[a]) for a in arms if a != arm) for row in rows)
        for arm in ("swap", "rotate")
    }
    return {"dev_goals": len(goals), "dev_headroom": best_fixed - oracle,
            "dev_swap_wins": wins["swap"], "dev_rotate_wins": wins["rotate"]}


def prepare(stage, recipe, review, model_name="frozen.json"):
    """Freeze and preflight a research recipe after review and development-only fitting."""
    stage = Path(stage)
    if recipe.get("execution_purpose") != "research":
        raise ValueError("SKILL-006 requires a claim-bearing research recipe")
    inputs = recipe["inputs"]
    for required in (model_name, "development.zip", "research.py", "admission.py"):
        if required not in inputs:
            raise ValueError(f"missing declared input {required}")
    summary = dict(read_json(stage / model_name)["observations"])
    summary.update(development_summary(stage / "development.zip"))
    write_json(stage / "preflight.json", summary)
    recipe = dict(recipe, inputs=[*inputs, "preflight.json", "adequacy.json"])
    if "observations.json" not in recipe["artifacts"]:
        recipe["artifacts"] = [*recipe["artifacts"], "observations.json"]

    def criterion(field, operator, threshold, rationale):
        return dict(field=field, operator=operator, threshold=threshold, rationale=rationale)

    cap = [criterion("features", "eq", 4, "Use the frozen four structural summaries."),
           criterion("tree_depth", "le", 3, "Bound split depth to the reviewed controller capacity."),
           criterion("tree_nodes", "le", 15, "A depth-three binary tree contains at most fifteen nodes.")]
    training = [criterion("fit_calls", "eq", 1, "One prescribed deterministic fit at each replay."),
                criterion("train_goals", "eq", 3408, "Exhaust three-, four- and five-record training assignments.")]
    regime = [criterion("arms", "eq", 3, "Supply costs of every candidate controller to the learner."),
              criterion("training_labels", "eq", 10224, "Observe one target per training assignment and arm.")]
    discriminating = [criterion("dev_goals", "eq", 256, "Exhaust the separate four-record development panel."),
                      criterion("dev_headroom", "ge", 1, "Permit an observed gain over the best fixed arm."),
                      criterion("dev_swap_wins", "ge", 1, "SWAP must uniquely win a development case."),
                      criterion("dev_rotate_wins", "ge", 1, "ROTATE3 must uniquely win a development case.")]
    descriptions = {
        "model_size": ("Depth-three, four-feature cost-regression tree.",
                       "Evaluate transfer of a fixed compact policy with a controlled capacity.",
                       "Inputs and depth were engineered from prior development studies.", cap),
        "training_duration": ("One fit with full supervised controller costs on sizes 3 through 5.",
                              "The deterministic learner consumes the entire defined training population.",
                              "No iterative optimizer or convergence claim.", training),
        "datasets": ("Synthetic assignment goals with distinct initial values and empty scratch.",
                     "Both procedures have strict wins on the frozen diagnostic panel.",
                     "Six-record goals were previously used for a different controller comparison.", discriminating),
        "training_regime": ("Multioutput cost regression against exactly three candidate controllers.",
                            "Each controller receives the same initial state, goal and primitive fallback.",
                            "Cost excludes binding enumeration and routing work.", regime),
    }
    dimensions = {}
    for name, (configuration, justification, limitations, criteria) in descriptions.items():
        dimensions[name] = dict(configuration=configuration, justification=justification, limitations=limitations,
                                criteria=[dict(c, id=c["field"], evidence="preflight.json") for c in criteria])
    execution = dict(model_size=cap, training_duration=training, training_regime=regime,
                     datasets=[criterion("heldout_goals", "eq", 46656, "Evaluate the complete six-record roster.")])
    contract = dict(schema="trainlab.research-adequacy.v1", claim_kind="fixed_budget_comparison",
                    claim="Does the frozen two-skill selector reduce weighted action cost below every fixed controller?",
                    scope="Complete six-record assignment roster and the reviewed three-controller action semantics.",
                    excluded_claims="Latency, external replication, general algorithm discovery and broad transfer.",
                    recipe_sha256=recipe_identity(recipe),
                    inputs={name: digest(stage / name) for name in inputs + ["preflight.json"]},
                    review=dict(review, criteria_frozen_before_evaluation=True), dimensions=dimensions,
                    execution=dict(artifact="observations.json", dimensions=execution))
    write_json(stage / "adequacy.json", contract)
    recipe["research_adequacy"] = dict(file="adequacy.json", sha256=digest(stage / "adequacy.json"))
    write_json(stage / "recipe.json", recipe)
    hashes = {file: digest(stage / file) for file in recipe["inputs"]}
    report = assess(recipe, stage, hashes)
    if report["status"] != "DECLARED_CRITERIA_MET":
        raise ValueError(f"SKILL-006 preflight {report['status']}: {report['checks']}")
    validate_execution(contract, recipe)
    return stage / "recipe.json"


def observe(output, producer, observations):
    """Persist measured worker counters and bind them to the full admitted input set."""
    recipe = read_json(Path("recipe.json"))
    hashes = {name: digest(Path(name)) for name in recipe["inputs"]}
    receipt = dict(schema="trainlab.execution-adequacy-observations.v1",
                   recipe_sha256=recipe_identity(recipe), inputs=hashes,
                   producer=dict(file=producer, sha256=hashes[producer]), observations=observations)
    write_json(Path(output) / "observations.json", receipt)
