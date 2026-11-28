# Learned position control on WikiText questions

Three 23,594-parameter full-prefix extractive readers began with weights trained for 384 updates on byte-level and generated-English revision pairs. The source window is 49 bytes per candidate and the question representation uses its last eight bytes. The reader can output any byte observed in the prompt. Position labels come from the generator during training. Inference receives only the single raw prompt.

Each parent was continued for 384 updates with 24 prompts per update. The real-text arm used 288 updates on WikiText and 96 updates of template replay. Its matched continuation used 384 template updates. Every pair used the same optimizer and starting weights. Each automatic WikiText training example picks two different digit values near two unique words in one authentic passage, asks about each word, and edits one digit. Each set of four inputs includes both questions before and after that edit. The definition of distance uses the character position at the center of the anchor word. These questions are algorithmically generated and teach lexical matching. The 11 natural-passage tests use hand-authored questions. They were held out of training, with synthetic edits constructed to retain valid digits, dates, denominators and sports outcomes.

| Measure | Real continuation, by seed | Template continuation, by seed |
| --- | --- | --- |
| Automatic QA, all four correct (192 passages) | 85, 67, 85 | 0, 0, 0 |
| Automatic QA, all four correct (175 passages with no shared 32-byte training substring) | 73, 58, 79 | 0, 0, 0 |
| Hand-authored questions, all six correct (11 passages) | 1, 0, 0 | 0, 0, 0 |
| Hand-authored questions, single answers (66 answers) | 37, 27, 31 | 0, 0, 0 |
| Generated English, balanced query swap (128 passages) | 128, 124, 128 | 128, 128, 128 |

The query-blind first-digit and last-digit rules each scored 33/66 single answers and 0/11 strict six-answer suites on the hand-authored questions. The real continuation learned to select evidence for the generated lexical task with a held natural passage. Reliable transfer to differently phrased questions remains unestablished.

Seed order is 9731, 9743, 9769. The train corpus provided 1,536 examples from 1,009 documents, with 192 automatic holdouts from 131 separate documents. The natural benchmark uses the development corpus. All three model seeds saw the same test passages. The overlap audit, per-seed scores and checkpoint hashes are included in the adjacent JSON files. The larger question-selection capacity of this reader comes from inspecting the full prefix; its results bound this specific task and access pattern.
