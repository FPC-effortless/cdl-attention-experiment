# CSM-BDH-002 argument attachment

We compare four deployable mention selectors on the same five held-out synthetic layouts. The two logistic models use identical governing-occurrence labels and training examples. The structured model adds indicators for a following "to the" argument, a preceding argument adjacent to "receives", an intervening "rather", and the same punctuation-delimited segment. These are hand-engineered approximations to grammatical dependencies. The deterministic argument rule uses the positive argument indicators before falling back to proximity within the segment. The oracle receives the target occurrence. All arms share supplied expression boundaries and bindings and the archived encoder and conditional COPY/LINK executor.

Each of three seeds trains on 240 quartets per learned model and evaluates 96 quartets per layout. Both expression order and approval order change within each quartet. The evaluation rosters repeat across seeds. A quartet counts only when every member has the correct occurrence, slot and execution effects. Wrong authorizations are counted independently per COPY or LINK.

The learned arms and deterministic argument rule scored 96/96 on three layouts per seed, including an unseen connector. The original clause-first rule passed one layout. Both contrastive subject and indirect scope scored 0/96 for every deployable arm. On each of those layouts, the learned structured arm committed 768 unauthorized operations per seed. The oracle passed all five. The gate required 92/96 in every layout and seed, so there is no qualified speed benchmark.

The initial fixture renderer labeled repeated references through a shared placeholder, which could mark the wrong occurrence and leak a marker into the input. The final renderer has distinct governing and competing placeholders, requires exactly one label marker and strips it before scoring. All models were refit and all arms rerun after correction. check_targets.py independently locates the governing occurrence in each test layout, and audit.py checks absence of markers and both correct record choices per quartet.

The observed failures involve grammatical contrast and scope. A useful next comparison would test whether a compact dependency parser can distinguish the subject from a contrast phrase and a governing predicate from a subordinate denial before choosing a record. This result covers a synthetic kit with supplied record bindings and a fixed executor.

Replay with Python, NumPy, pandas and scikit-learn. Run run_study.py, audit.py, and check_targets.py.

The complete handoff package contains the archived raw outputs and weights described above. The GitHub checkout includes the inputs needed to regenerate these outputs.
