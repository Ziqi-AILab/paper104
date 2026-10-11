# Tests and small checks

Run tests locally on CPU after installing `.[test]`. They do not require
the external research datasets or pretrained weights. A quick scientific
logic check is:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_chemistry.py tests/test_masking.py tests/test_relations.py \
  tests/test_data_workflow.py tests/test_prediction_records.py
```

This covers periodic identities/views, atom-mask budgets, relation matching,
split-role handling, and prediction records using small fixtures. RDKit is
required for the chemistry and relation checks; skipped tests are not evidence
that those components passed.

The supplied test directory is a CPU unit/regression suite. For broader
statistical and training-state coverage:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests
```

Some tests import PyTorch and create small CPU tensors. They do not run a
ChemBERTa training job, acquire a GPU, download model weights, or submit Slurm
jobs. The statistical tests check paired/nested seed treatment, metric units,
primary comparison handling, and separation of secondary analysis populations.
They do not recompute the paper's experimental results.

## Before expensive execution

Inspect a generated manifest and a representative script first. Generation is
a CPU file-writing operation, not an end-to-end smoke run. For frozen heads,
the existing dry-run prints the exact command without fitting:

```bash
python scripts/run_local_probe_manifest.py \
  --manifest runs/manifests/frozen_probes/frozen_probe_manifest.tsv \
  --row-index 0 --dry-run
```

A model smoke run is a separate GPU experiment. Use a distinct checkpoint
root and run ID, retain the specified model, and submit through Slurm on the
GPU appropriate to that stage. `run_pretraining.py` exposes
`--max-train-rows`, `--max-validation-rows`, and
`--stop-after-optimizer-steps` for a small training/resume check. Never write
such a shortened run into a formal output directory or treat it as a paper
result. Cache/fidelity runs belong on the evaluation GPU; frozen heads and
statistics remain local CPU work.

Useful completion checks are concrete: the expected checkpoint/cache exists,
sample IDs join to the intended split, predictions are finite, validation
metrics and final run records are present, and the expected representation and
head seeds are recorded. A successful process exit alone does not establish
experimental correctness. See [workflow.md](workflow.md) for output locations.

## Limits of these checks

A passing source-only suite establishes neither exact historical split
recovery nor matching trained weights or numerical paper results. Those require
the original raw versions, frozen exclusion identities, external model assets,
completed runs, and compatible numerical environments. The archived adaptation
initialization/seed limitation remains as described in
[parameters.md](parameters.md); the tests do not repair or conceal it.
