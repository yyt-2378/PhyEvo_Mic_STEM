# Public benchmark data

`manifest.yaml` records immutable source URLs, license, checksums, intended
tasks, and label limitations. Downloaded files live under
`datasets/public/<dataset-id>/` and are treated as read-only experimental
inputs.

Fetch the small public benchmark:

```bash
python scripts/fetch_public_dataset.py --dataset guo_2021_atomic_defects
```

The Guo et al. archive is CC BY 4.0 and accompanies:

- DOI: `10.1038/s41524-021-00642-1`
- Data DOI: `10.5281/zenodo.5520169`

The archive has paper-level visual examples but no exhaustive
machine-readable pixel masks. Public-data runs are therefore reproducibility
and qualitative sanity checks. Pixel accuracy is reported only on the
separate deterministic synthetic benchmark where exact masks are known.

## Admitted EMSeek WSe2 training group

The WSe2 atomic-defect subset from
[gary23ai/EMSeek](https://huggingface.co/datasets/gary23ai/EMSeek) is pinned to
revision:

```text
8f4a987c78af743098459e5d467f2c1b5f5c11b3
```

The admitted subset contains 107 1024x1024 images and 107 RGB point-label
images (214 remote files; 92,121,442 bytes). Its license is Apache-2.0.
Each remote LFS object and each local file is verified by SHA-256. The
source manifest is
`intake/emseek_wse2_source_manifest.json` with SHA-256
`41891936c965b4a636c6ad9671a706d8ba79c82706e5fc7cea2bf89cd88c8a0e`.

Public metadata does not resolve acquisition/sample relationships well enough
for leakage-safe splitting. All 214 data files plus the source manifest are
therefore admitted as one conservative `train` group under
`intake/emseek_wse2_train_v1.sealed.yaml`. Images may enter active learning;
annotations and metadata may not. Published materials describe consensus
annotations, but the reviewer count is not public, so the labels are recorded
as `author_curated`, not Gold. They cannot be used as independent calibration
or locked-test evidence.

Current development results and limitations are documented in
`../docs/periodic_counterfactual_and_consensus_zh.md`.

## Open-world v0.4 prospective intake

Three additional plans are sealed but have not been merged into
`partition_registry_v2.yaml`:

- `intake/emseek_remaining_train_v1.sealed.yaml`: 16 remaining EMSeek
  scientific subsets, pinned as 28,334 remote LFS SHA-256 objects; no bulk
  image download has been performed.
- `intake/guo_superscan_development_v1.sealed.yaml`: one complete SuperScan
  acquisition, verified locally by SHA-256.
- `intake/al_sigma3_development_v1.sealed.yaml`: two STEM orientations, EBSD,
  simulation, and repository metadata in five provenance groups, all verified
  locally by SHA-256.

The two development plans use `development_candidate`; they cannot enter
training or active learning. The EMSeek plan remains prospective until every
selected local file passes `--verify-files`. See
`../docs/open_world_v0_4_implementation_zh.md`.

## Prospectively locked external benchmark

The complete Dennler et al. III–V HRSTEM archive is registered as one
indivisible `locked_external_test` group:

- DOI: `10.5281/zenodo.5792917`
- paper DOI: `10.1016/j.micron.2021.103069`
- license: CC BY 4.0
- archive: `public/dennler_2021_hrstem_defects/hrstem_defects_dataset.zip`
- bytes: 63,624,231
- SHA-256:
  `abef46a0183ba08b925c64807d5e914cd97967444d9fd3e0799924896f0614ec`

It contains 31 expert-labeled defective images and 30 normal controls. The
executable pre-registration is
`../benchmarks/locked_dennler_v1_protocol.yaml`. The protocol seal froze the
archive, registry, configurations, inference code, target mapping, metrics,
and failure policy before labels were unpacked. The sealed zero-shot
prediction set was then scored once; its hash is
`f7b126bb10691d81c3e08d677325e2e36dac5d0f1619436bd9e220ef773232de`.
The low external scores are preserved, and this consumed asset can never
enter active learning, calibration, threshold selection, or training.

## Preserved download incident

An interrupted concurrent transfer produced
`public/guo_2021_atomic_defects/relaxed_single_Al_partial_dislocations.tif`
with MD5 `0cf437c1d378b95e095630f366800fb9`. It is intentionally preserved under
the local data-protection policy and is not referenced by the manifest or any
benchmark.

The verified input is
`relaxed_single_Al_partial_dislocations_verified.tif`, size 5,471,022 bytes,
MD5 `7f67ed89f081eab4e946c5916977f2cf`.
