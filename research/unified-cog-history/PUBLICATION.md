# Binary release publication contract

All current result snapshots are in this branch. **The binary assets are not yet uploaded.** There are 22 original ZIPs, 1,901,066,900 archive bytes, 776 checkpoint entries and 753 distinct checkpoint digests, including the new UNIFIED-016 artifact. These are archive objects, not campaign fit counts.

The authenticated GitHub connector used for this cycle supports text writes; the local runtime has no direct GitHub transport for these large files. Do not claim release custody until the actual bytes have been uploaded and retrieved.

## Release process

- Use the locally supplied publication kit (publish_cog.py and release_inventory.json) with the hash-indexed original archive files.
- Prepare the filtered release assets locally first, preserving each weight byte-for-byte and excluding raw corpus and non-allowlisted paths. Path filtering does not replace review for disclosure or licensing.
- Use an authorized GitHub CLI session to upload to a versioned draft release targeting this branch.
- Re-download and compare SHA-256 for each asset before publishing. Do not overwrite conflicts or replace prior releases. Save the verified remote receipt.
- Do not publish protected evaluation content, credentials, private data, or unlicensed corpora. The local kit is not evidence of a successful remote upload.

UNIFIED-016 original ZIP: cog_unified_016_evidence.zip, SHA-256 22c58f74da19dad051931f266a3e0ac404e0c85cbf79859c81dcaac5565bd401. Historical original checksums are in historical_archive_index.json. For future experiments, keep this contract, unique IDs, exact source and weights, and separate promotion and campaign admission.
