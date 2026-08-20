# Pinned Dennler upstream source

`bragg_filtering_72d37c3.py` is byte-identical to
`src/utils/bragg_filtering.py` at upstream commit
`72d37c3ca5a84cfa39337c7243a80e3ffcc0f648`. Its SHA-256 is
`f01cbda29d4d24530618cb508e40e8d17828cef2128b3504e00a15305ec9c67c`.
The adjacent MIT licence is retained.

`verify_dennler_upstream_parity.py` applies only two in-memory API
compatibility substitutions, then checks exact pixel equality against the
repository port on a deterministic, non-degenerate 1024×1024 fixture. The
stored parity JSON pins the input and output hashes. This verifies the released
weight-free Fourier/Bragg functions, not the unavailable trained VGG system.
