# NCM output-engine feasibility — proposed software evaluation

The earlier WOYENGI-ENDTOEND-020 scientific campaign is paused, not silently converted into diffusion or NCM. No 020 scientific fits completed. Its data, execution adapters and baseline sources remain recoverable.

## Purpose
Verify that the authors' released conditional NCM can run safely on the user's free T4 allocation, generate an actual coarse-to-fine parallel output, and expose full runtime/factual-preservation constraints before designing a CSM-conditioned training campaign. This is not a reproduction of paper quality scores, a trained CSM improvement, a general QA benchmark, or a superiority claim.

## Exact external assets
HazyResearch/ncm source c70274273f489abc73832b2b98584a8483b6b203 (MIT). Author checkpoint repository hazyresearch/ncm-checkpoints revision d342ce8aaea58830eef3b8ad473664673e41f692, XSum generator354355211bytes and tokenizer299626079bytes, exact SHA256 in checkpoints/MANIFEST.json. Config SHA2565dee6b7339645a22362f695fbadb27df6e5254db2d3be0bff8516138d95da493. GPT2 tokenization assets only from openai-community/gpt2 revision607a30d783dfa663caf39e06633721c8d4cfcd7e; no GPT2 model downloaded.

## Safety and reproducibility
Restricted torch.load(weights_only=True) passed on both checkpoint dictionaries. No blanket global allowlist or weights_only=False fallback. Verify every source/checkpoint/tokenizer artifact, strict state dictionaries and finite tensors. Replace only the fixed reviewed Hydra class-resolution indirection if required; preserve original source hashes and patch evidence. Load local GPT2 tokenization with trust_remote_code=False/local_files_only=True. Never run stock evaluation entrypoints: their data configuration defaults to bundled TEST/pickle. No dataset download or protected TEST access is permitted.

## Runtime probe
One fictional library-hours news article for one cold and three warm calls, plus three fixed fictional name/identifier/number records for separate factual-preservation diagnostics. There are exactly seven generation calls and three learned-VQ encode/decode roundtrips, plus three engineered string-copy controls. None is a protected evaluation example. The diagnostic panel is frozen in the reviewed adapter and ADAPTER_PROTOCOL.json; panel SHA256 is 6bc4556d759d938db2458f0bdc8642ca3dcc615267be8d2f2289c9eb0a4e6418. Fresh construction does not establish absence from external pretraining. VQ roundtrips receive the target text itself and test codec fidelity, not prefix-to-answer prediction. Literal omission in a summary is not automatically a factual contradiction. FP32, batch1, deterministic top1 selection, three warm repeats after one cold generation. Report actual loaded unique parameters (generator/tokenizer shared identities deduplicated), complete startup costs, peak allocated GPU memory, raw output and complete-call timing including input tokenization, all hierarchical levels, VAE decoding, published sequential final-token penalties and text decoding. No uncounted decoder or reranking stage. No transfer of the paper's H100 speedup to T4.

## Interpretation
The released XSum model is a summarizer with512-token input and48-token output configuration. An incorrect or omitted hour/name is a fidelity observation for this fixture, not proof that CSM memory fails. Generated text must never overwrite authoritative memory. NCM is hierarchical levelwise generation, not iterative masked diffusion; a separate diffusion decoder must genuinely run its reverse/denoising procedure.

## Preconditions before execution
Source-only GitHub recovery backup, private verified input closure, independent protocol/code review, native Forge/TrainLab/Veritas/Research Lab execution record. No paid resources. All failures retained. This file is not itself an admission or an execution result.
