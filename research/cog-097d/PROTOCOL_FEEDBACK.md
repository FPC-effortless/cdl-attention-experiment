# COG-ATTN-097d2: outcome-only continual learning (recorded before first 097d2 fresh-test score)

## Goal and scope
Does binary outcome feedback from a retrieval action improve the attention-free raw-prose encoder's ability to ground *new* author-written queries in an unseen archive? This is a closed-loop *simulated* retrieval environment based on naturally authored Python documentation. Not a full COG-ATTN or live human feedback.

## Interactions
- Training archive: the same 297 documentation bodies used in 097d; each naturally authored first summary is an incoming query.
- An agent sees the query and all archived raw passages and chooses one passage ID as an action. A hidden evaluator returns ONLY a success/failure bit for that chosen action. It never discloses the right passage after failure, any relation schema, semantic tags, or the reference solution.
- A publicly described attention-free **character TF-IDF policy** proposes up to three actions per query (each action must pay a feedback request). It operates only on the visible raw text, not test labels; successful and rejected actions become the entire supervision for an outcome-trained encoder. This is a deliberate bootstrapping strategy, so any gain must be compared against the same lexical policy rather than being described as learning from scratch with no priors.
- No ability to query the evaluator at test time. All model parameters freeze after training on the training modules.

## Split
Fresh independent test modules (not used in 097c or 097d): `smtplib, imaplib, difflib, cmd, queue, gettext, calendar, ipaddress, ftplib, ntpath, posixpath, logging.handlers, pickletools, fileinput, mailbox, xml.sax.handler, xml.sax.xmlreader, tokenize, codeop, code, numbers, shelve, wsgiref.simple_server, wsgiref.handlers, email.mime.multipart, email.mime.text, pprint, fnmatch, string, enum, runpy, ast, stringprep, unittest.suite`. Test archive is deduplicated against the train archive by both full passage hash and author query. Test source modules never contribute optimizer updates.

## Frozen hyperparameters and controls
Three seeds 1701,1707,1711. Initialize from the corresponding 097d raw-experience encoder checkpoint. Train on up to 3 lexical retrieval attempts per training query; feedback records only `(raw_query, selected_raw_document, boolean_outcome)`; 8 training epochs, AdamW learning rate 0.0003, 8.0 times cosine-similarity logits; BCE classification of observed actions. Compare: character TF-IDF, pre-feedback neural encoder, outcome-trained neural encoder, identically trained encoder with shuffled outcome bits. Do not change hyperparameters, dataset modules or test scoring based on the scores of the fresh test.

## Metrics and promotion
Top-1 and recall@5 on the fresh test, honest false-answer count for missing evidence, training and indexing work, number of action probes and feedback bits, training time and query latency. A positive qualification requires outcome-trained encoder to exceed all controls at matched competence and to reduce errors, not merely training loss. Negative results are preserved, without optimizing on the test.

## Important limitation
The evaluator's hidden association of an authored summary to its containing docstring is a proxy for externally observed task success. It is *not* live user intent, real conversation, causal reasoning, open-world skill discovery or general intelligence. A lexical bootstrapping policy is part of the learning system and must remain accounted for.
