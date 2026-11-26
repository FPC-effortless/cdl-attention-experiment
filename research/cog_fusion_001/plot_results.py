"""Publication-style summary from measured qualification receipts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
E = HERE / "evidence"
SEEDS = (17, 29, 41)


def read(path):
    return json.loads(path.read_text())


def main():
    ev = {f"{arm}_{s}": read(E / "evaluation" / f"{arm}_seed{s}" / "RESULT.json")
          for arm in ("memory", "no_memory") for s in SEEDS}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#344256", "text.color": "#172536",
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.5))
    blue, orange, grey = "#126F86", "#D97434", "#718095"
    ax = axs[0, 0]
    x = np.arange(3)
    for arm, color, label, offset in (("memory", blue, "Fused memory", -.13),
                                     ("no_memory", orange, "Trained memory-off", .13)):
        y = np.array([[100 * ev[f"{arm}_{s}"]["panels"][p]["modes"]["strict"]["exact_answers"] / n
                       for p, n in (("test", 256), ("fresh", 256), ("long", 64))] for s in SEEDS])
        ax.bar(x + offset, y.mean(0), width=.24, color=color, alpha=.85, label=label)
        for row in y:
            ax.plot(x + offset, row, "o", ms=4, color="#172536", alpha=.65)
    ax.set_xticks(x, ["Familiar\n256 answers", "New wording\n256 answers", "64 filler events\n64 answers"])
    ax.set_ylim(0, 108); ax.set_ylabel("Generated-answer exact match (%)")
    ax.set_title("A  The familiar gain does not generalize", loc="left", pad=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.55, 1.03), fontsize=9)
    ax.axhline(95, color="#AAA", ls=":", lw=1)

    ax = axs[0, 1]
    conditions = [("Trained memory", "memory", "strict"),
                  ("Acute removal", "memory", "acute_memory_off"),
                  ("Trained memory-off", "no_memory", "strict")]
    for i, (label, arm, mode) in enumerate(conditions):
        values = [100 * ev[f"{arm}_{s}"]["panels"]["test"]["modes"][mode]["exact_answers"] / 256 for s in SEEDS]
        ax.bar(i, np.mean(values), color=(blue if i == 0 else orange), width=.6, alpha=.85)
        ax.plot([i] * 3, values, "o", color="#172536", ms=4)
        ax.text(i, np.mean(values) + 3, f"{np.mean(values):.1f}%", ha="center", fontsize=10)
    ax.set_xticks(range(3), ["Trained\nmemory", "Acute evidence\nremoval", "Trained\nmemory-off"])
    ax.set_ylim(0, 115); ax.set_ylabel("Familiar exact match (%)")
    ax.set_title("B  The answer path uses the records", loc="left", pad=12)

    ax = axs[1, 0]
    language = read(E / "LANGUAGE_PRESERVATION.json")
    original = language["references"]["original_connected_cog"]["mean_nll"]
    local = language["references"]["parent_local_only"]["mean_nll"]
    runs = language["runs"]
    if isinstance(runs, dict):
        fitted = [r["metrics"]["mean_nll"] for name, r in runs.items() if name.startswith("memory_")]
    else:
        fitted = [r["metrics"]["mean_nll"] for r in runs if r.get("fit", r.get("name", "")).startswith("memory_")]
    if len(fitted) != 3:
        # Values are keyed as fit in the current receipt; fail rather than plot
        # an accidental empty mean or select results by performance.
        raise ValueError("could not resolve the three declared memory runs")
    vals = [original, local, np.mean(fitted)]
    ax.bar(range(3), vals, color=[blue, grey, orange], width=.6, alpha=.9)
    for i, v in enumerate(vals):
        ax.text(i, v + .06, f"{v:.3f}", ha="center")
    ax.plot([2] * 3, fitted, "o", color="#172536", ms=4)
    ax.set_xticks(range(3), ["Original full\nCOG", "Local-only\nparent", "Fused memory\nthree seeds"])
    ax.set_ylim(0, 2.95); ax.set_ylabel("Validation NLL, nats/character (lower is better)")
    ax.set_title("C  Original language quality is reduced", loc="left", pad=12)
    ax.text(.5, .94, "Same 8,192 character targets", transform=ax.transAxes, ha="center", color=grey, fontsize=9)

    ax = axs[1, 1]
    interactions = read(E / "INTERACTION_RESULTS.json")
    y = [interactions[str(s)]["learned_questions_correct"] for s in SEEDS]
    ax.bar(range(3), y, color=orange, alpha=.85, width=.6)
    for i, v in enumerate(y):
        ax.text(i, v + .18, f"{v}/7", ha="center", weight="bold")
    ax.set_xticks(range(3), [f"Seed {s}" for s in SEEDS]); ax.set_ylim(0, 8)
    ax.set_yticks(range(8)); ax.set_ylabel("Correct learned answers in live scenario")
    ax.set_title("D  Repeated conversation remains unqualified", loc="left", pad=12)
    ax.text(.5, .95, "Pending policy: 3/3 runs correct; history replay: zero", transform=ax.transAxes,
            ha="center", color=grey, fontsize=9)
    for ax in axs.flat:
        ax.set_axisbelow(True); ax.grid(axis="y", alpha=.15)
    fig.suptitle("COG Fusion 001  |  Real memory benefit, incomplete qualification", fontsize=16, weight="bold", x=.05, ha="left")
    fig.text(.05, .025, "Dots show three seeds on shared held data. Fixed 16-identity synthetic domain; no general attention-efficiency claim.\n"
             "New wording changes vocabulary and syntax. The long-history panel adds neutral filler; the live scenario interleaves answers and observations.",
             fontsize=9, color="#536274")
    fig.subplots_adjust(left=.075, right=.97, top=.88, bottom=.15, hspace=.55, wspace=.3)
    fig.savefig(E / "COG_FUSION_001_QUALIFICATION.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
