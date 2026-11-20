# COG-ATTN continuation binary evidence

This branch contains the exact 15 local research archives from WR-004 through the COG-ATTN-047–050 qualification attempt and its follow-ups. The separate [research index](https://github.com/FPC-effortless/cdl-attention-experiment/tree/research/cog-attn-reproducibility-20261120/research/cog-attn-handoff/continuation) provides study-level reports and source.

Archives up to 20 MiB appear as complete files. Larger archives have numbered parts. Each directory under [archives](archives) has a manifest with ordered parts, their SHA-256 hashes and the original archive hash. The global [manifest](manifest.json) covers all 15.

Use the standalone downloader to reconstruct and verify one bundle or all bundles. With a local copy of this branch, the downloader uses parts from disk. Given a revision, it fetches verified parts directly from GitHub.

    python download_evidence.py --list
    python download_evidence.py wr004-005-evidence.tar.gz --output ./verified
    python download_evidence.py --all --output ./verified
    (cd verified && sha256sum -c ../SHA256SUMS)

The parent handoff lists ten older ZIP archives from COG-ATTN-001–046, the lifecycle cycles and WR-001–003. The older reports and checksum inventory are available on the research branch. The ten older binary archives were unavailable for this publication.
