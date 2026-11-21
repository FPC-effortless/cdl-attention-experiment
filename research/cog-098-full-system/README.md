# COG-ATTN-098: full-system engineering gate

The end goal is one trained, persistent agent that uses the existing COG verification and planner alongside learned perception, memory, and online feedback. It must handle authentic dialogue, executable software fixes, and interactive control without gold data at inference time.

The source/receipt audit is deliberately strict. Running it against current COG code returns NOT_QUALIFIED. It does not claim a full model exists when 071's integrated workflow and 097e's independently trained retrieval policy have not been wired together.

From the repo root, run Python with these arguments:

    python -m pytest -q research/cog-098-full-system/test_gate.py
    python research/cog-098-full-system/gate.py --strict

The first command tests the admission safeguards. The second produces readiness.json and exits nonzero until the actual integrated model and real-world evaluation exist. Read PROTOCOL.md for the genuine research standard.