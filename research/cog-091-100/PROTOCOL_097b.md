# COG-ATTN-097b methodology and controlled follow-ups

Purpose: falsify the claim that a small non-attention dual retriever plus unstructured episodic memory already matches self-attention's re-query and language generalization.

Executed after trial-and-error local pilot iterations; **not prospectively preregistered**. Source and failure cases retained.

- Public synthetic environment: seven raw factual fields per unseen entity, including one field with no training queries; corrections add a newer raw record without deleting the old.
- Query interface: raw multi-sentence text, entity name never passed as a structured ID; engineered alias extraction from full names in the text and prior clause. Refuse when zero or multiple names resolve.
- Training: paired user question and verified supporting raw episode. Trainable independent projections on hashed character n-gram vectors with within-entity contrastive CE. Dev-only checkpoint selection.
- Comparators: no-fit overlap, per-query word/character TF-IDF, random frozen projections, learned projections, naive feature fusion, support-conditioned witness routing, support-shuffled negative, and oracle field (ceiling only).
- Witness route: **14 extra demonstrations include the exact held phrase families**, so measure few-shot binding, not zero-shot. Shuffled evidence preserves 14 witnessed queries and evidence frequency but destroys correct pair association.
- Six seeded worlds and disjoint train/dev/test entities; test cases have familiar wording, held paraphrases, previously unqueried relation, and ambiguous/missing entity contexts. 8,192-record primary evaluation; 2,048 and 32,768 growth diagnostics on three seeds.
- Ingestion and retrieval: append-only source-preserving store; latest-version index, rarity-aware alias postings and 7 candidate episodes from named entity. No unbounded raw scan in the indexed method; evaluate direct raw scan as an oracle control.
- Primary score: correct *final selected episode/payload* on 678 answerable test questions, and unsupported reference deferrals on 108 contexts. Candidate top-k separate.
- Cost: CPU single-thread per-query scorer latency; independently report name resolution, cold build and train. Do not claim complete COG inference latency from component scores. 
- Safety: no external signed write or live service access; authorization regression kept from 097a. Failures of a signed-but-wrong semantic action remain unresolved.
- Pending qualification: authentic data, unseen utterance families even after supports, loss of engineered seven-field candidates, acquired aliases, full agent integration, online correction learning, matched Transformer baseline, separate-machine replay and independent verification.
