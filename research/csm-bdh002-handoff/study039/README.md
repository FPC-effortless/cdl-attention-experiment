# CSM-BDH-002 clause attachment

The study freezes a word-to-expression dictionary, record representations and conditional COPY/LINK execution. It compares token proximity, a rule that prefers references in the same punctuation-delimited segment, and a learned scorer combining segment distance, token distance, direction and nearby words. The learned scorer gets governing occurrence labels during training. The oracle receives the governing occurrence at evaluation. All arms use synthetic tokenized text and supplied expression boundaries and record bindings. None runs a natural-language parser. There are 240 training quartets per seed and 96 evaluation quartets per layout per seed. Seeds share evaluation rosters. A quartet crosses both reference orders with both orientations of approval.

The nearest-reference control passed two of five layouts. The learned scorer and segment rule passed four, with 96/96 exact quartets in each passing cell. Both failed all quartets on the same-clause competing reference layout, with 384 unauthorized conditional COPY or LINK operations per seed. The oracle passed every cell. In some cases, proximity selected the wrong occurrence of the same word and the final record happened to be correct. The gate checks occurrence identity as well as the expression, route, and executor effects. No deployable arm passed every layout, so the gated inference comparison was skipped.

After evaluation, intervention.py follows the explicit "applies to the" wording on the failing fixture. It selects the correct occurrence on 1,152/1,152 query evaluations. This exercise uses a rule written after observing the test layout. It scores governing occurrence only and reuses the original rosters.

A dependency-style frontend is a grounded follow-up. Universal Dependencies distinguishes links to a predicate from links to an antecedent in its enhanced representation for relative clauses. CSM can use a similar separation between governing occurrence and persistent record identity. Compare a structured rule with a compact learned scorer using the same candidate mentions and distinguish in-clause attribution from an authorization. Freeze the paraphrase families before fitting. Candidate quality and end-to-end inference cost are separate outcomes. Mamba provides a parallel method for input-dependent state-space updates. That property applies to the neural context encoder where the transitions can be prepared from its input. An arbitrary sequence of data-dependent CSM record mutations still has execution dependencies. Hymba motivates a bounded local attention control beside a recurrent frontend.

Sources

- Universal Dependencies, Enhanced Dependencies. https://universaldependencies.org/u/overview/enhanced-syntax.html
- Gu and Dao, Mamba. https://arxiv.org/abs/2312.00752
- Dong et al., Hymba. https://research.nvidia.com/publication/2025-04_hymba-hybrid-head-architecture-small-language-models

Replay with Python, NumPy, pandas and scikit-learn. Run run_study.py, audit.py and intervention.py from this folder. Protocol, fitted weights, raw predictions and result tables are included.

The complete handoff package contains the archived raw outputs and weights described above. The GitHub checkout includes the inputs needed to regenerate these outputs.
