# Reproduction workflow

Run commands from the repository root, using the environment described in the
[README](../README.md). Data, exclusion identities, upstream model assets,
trained checkpoints, and prediction caches are not included. Follow
[data.md](data.md) and [models.md](models.md) before running the stages below.
The supplied selected recipes reproduce the recorded settings; they do not
remove the need for the original inputs. Newly acquired data or regenerated
identities are not automatically identical to the paper's splits.

## Stage order and execution

| Stage | Consumes | Produces | Execution path |
| --- | --- | --- | --- |
| Data preparation | External standardized tables and exclusion identities | Project splits, periodic PI1M table | `[CPU → local, no sbatch]` |
| Relation/input construction | Prepared PI1M partitions and endpoint tables | Relations, decoys, encoder-input tables | `[CPU → local, no sbatch]` |
| Continued pretraining | Base model, training partition, training relations | Token checkpoints and final model | `[GPU-4090 → sbatch]` |
| Embeddings and fidelity | Frozen checkpoints and evaluation inputs | Embedding caches and per-record metrics | `[GPU-A800 → sbatch]` |
| Frozen heads | Cached embeddings and task splits | Predictions, validation metrics, fitted heads | `[CPU → local, no sbatch]` |
| Adaptation/neural baselines | Task splits and model-specific inputs | Predictions and run records | `[GPU-A800 → sbatch]` |
| Classical baselines/statistics | Tables or completed predictions | Baseline predictions, tables, intervals | `[CPU → local, no sbatch]` |

Every `generate_*_matrix.py` command below writes manifests and scripts only.
It does not train, submit jobs, or wait for dependencies. Generated GPU scripts
use Slurm and `.conda/env/bin/python`; review the partition, GPU resource,
memory, wall time, and your chosen QoS before submission. Site resource names
are not portable. Run CPU stages locally, not in a GPU allocation.

## 1. Configure paths and prepare data

Replace each `/path/to/...` with your own existing asset location; the
checkpoint directory is a writable destination for newly trained models.

```bash
python scripts/configure_local_paths.py \
  --data-root /path/to/data \
  --model-root /path/to/models \
  --checkpoint-root /path/to/checkpoints \
  --exclusion-root /path/to/exclusions
python scripts/prepare_project_data.py
```

The data root must already contain the standardized layout described in
[data.md](data.md). The raw-data reconstruction script is a separate, local
CPU step documented there; `prepare_project_data.py` does not download data.

Preparation writes `data/splits/registry.json`, downstream manifests, and
`data/derived/pi1m_periodic_identity.parquet`. It preserves source held-out
roles, removes lower-priority overlapping rows, retains one representative
per eligible periodic identity, and excludes protected downstream identities
from PI1M training. It does not silently move held-out rows into training.

Construct relations separately within each partition, then build shared
encoder inputs:

```bash
python scripts/build_relations.py --partition train --target-triplets 10000 --seed 42
python scripts/build_relations.py --partition validation --target-triplets 1951 --seed 42
python scripts/build_relations.py --partition test --target-triplets 1880 --seed 42
python scripts/build_evaluation_inputs.py
```

Each relation directory contains `pi1m_triplets.parquet`, a fixed retrieval
decoy bank, and separate supplemental controls. Targets are not retained counts:
the historical validation search retained 1,949 admissible triplets. Do not use
`--allow-near-formula-fallback` for the formal primary relation set.
The final command writes `data/derived/evaluation_inputs/`, including the
structure-only inputs, separate PE-I metadata inputs, and fidelity anchors.

## 2. Generate and run continued pretraining

The repository includes the selected development recipe. To use it:

```bash
python scripts/generate_job_matrix.py formal
```

The generated `slurm/generated/formal/formal_manifest.tsv` enumerates six
trained arms and five representation seeds. A representative run is
`slurm/generated/formal/formal_A_eqs_seed42.sbatch`. Submit that script via
your reviewed Slurm procedure; generating it does not execute it.
Its Python payload is equivalent to the following, to be run inside the
allocated continued-pretraining GPU job, not on a login node:

```bash
python scripts/run_pretraining.py \
  --arm A_eqs --seed 42 --stage formal \
  --learning-rate 1e-4 --boundary-fraction 0.65 --interior-fraction 0.25 \
  --micro-batch-size 8 --gradient-accumulation 32 --resume-existing \
  --output-dir runs/pretraining/formal/A_eqs/seed-42
```

With the configured checkpoint root, the endpoint used by the evaluation
matrix is `/path/to/checkpoints/pretraining/A_eqs/seed-42/token-100.pt`.
The same run also writes `final_model/`, `last.pt`, earlier token checkpoints,
and `runs/pretraining/formal/A_eqs/seed-42/run_result.json`.
`--resume-existing` resumes the latest readable saved progress or skips a run
with an existing final model. Keep a distinct output root for a new experiment.

To repeat recipe selection rather than reuse its recorded output, generate
the `development` matrix, run its jobs, then execute:

```bash
python scripts/select_pretraining_recipe.py \
  --manifest slurm/generated/development/development_manifest.tsv \
  --output configs/pretraining/selected_recipe.yaml
```

This uses final validation MLM loss on development seeds 13 and 37. Regenerate
the formal matrix after selection. The separate `controls` matrix uses only
seeds 42, 123, and 777; it is not an additional five-seed core matrix.

## 3. Cache embeddings and evaluate fidelity

After the corresponding checkpoints are complete:

```bash
python scripts/generate_postpretraining_matrix.py evaluation
```

Inspect `slurm/generated/postpretraining/evaluation/evaluation_manifest.tsv`.
For the representative run, select rows with arm `A_eqs` and representation
seed `42`; do not submit rows whose input checkpoints are missing. The jobs
produce:

- `runs/embeddings/formal/A_eqs_seed-42/structure_only.npz` and a separate
  PE-I metadata cache;
- relation retrieval/topology metrics, invariance summaries, and boundary MLM
  measurements under `results/metrics/fidelity/`;
- ZINC retention MLM measurements under `results/metrics/retention/`.

The NPZ caches retain sample identities so downstream labels are joined to
the correct rows. Relation evaluation uses the stored partition-level decoys,
not a newly sampled retrieval task. Molecular retention and metadata-bearing
PE-I results remain separate from the primary polymer aggregate.

## 4. Fit frozen heads on local CPU

The recorded common head learning rate is `1e-3`. Generate the formal probe
manifest using that recipe:

```bash
python scripts/generate_postpretraining_matrix.py probes
python scripts/run_local_probe_manifest.py \
  --manifest runs/manifests/frozen_probes/frozen_probe_manifest.tsv \
  --row-index 0 --dry-run
```

Row zero is a manifest example, not necessarily `A_eqs`. Inspect the columns
`arm`, `representation_seed`, `task`, `protocol`, `fold`, `label_fraction`, and
`subsample_id` to choose a row. To fit a representative full-label EGC head
directly, after its cache exists:

```bash
python scripts/evaluate_downstream.py \
  --dataset /path/to/data/standardized/v1/transpolymer/egc.parquet \
  --split-manifest data/splits/transpolymer_egc.parquet \
  --embeddings runs/embeddings/formal/A_eqs_seed-42/structure_only.npz \
  --task EGC --protocol group_ood --fold 0 \
  --label-fraction 1 --subsample-id 0 --learning-rate 1e-3 --device cpu \
  --output results/predictions/example_A_eqs_42_egc.parquet \
  --metrics-output results/predictions/example_A_eqs_42_egc.validation.parquet \
  --probe-state-dir runs/probe_states/example_A_eqs_42_egc
```

This runs head seeds 11, 22, and 33. The shared learning rate is fixed, but
each fit still selects its best epoch by validation loss. Training labels
alone determine target centering and scaling; test predictions are restored
to the supplied target units. This is not a validation-free final refit.

For the complete manifest, after all caches exist, use
`python scripts/run_formal_frozen_probes.py --workers 2 --threads-per-worker 2`
and increase CPU concurrency only within your local resources. The runner
skips completed outputs and reuses loaded inputs within worker processes.

To repeat head-recipe selection, generate
`generate_postpretraining_matrix.py property-head-development`, run its A800
embedding jobs, then run the local development-probe manifest. Finally:

```bash
python scripts/select_property_head_from_development.py \
  --manifest runs/manifests/property_head_development/property_head_development_manifest.tsv
```

Only A0 development representations from seeds 13 and 37 enter this choice;
task/protocol/input-view units receive equal weight. Regenerate the formal
probe manifest after choosing the recipe.

## 5. Baselines and supervised adaptation

```bash
python scripts/generate_controlled_baseline_matrix.py
python scripts/generate_external_baseline_matrix.py
python scripts/generate_graph_external_baseline_matrix.py
python scripts/generate_postpretraining_matrix.py adaptation
```

These commands generate separate classical, Chemprop, TransPolymer,
PerioGT/GRIN, and LoRA/full-fine-tuning plans. Dependencies and external assets
are listed in [models.md](models.md); protocols are summarized in
[parameters.md](parameters.md). Run classical rows locally, for example:

```bash
python scripts/run_local_controlled_baseline_manifest.py \
  --manifest runs/manifests/controlled_baselines/classical_baseline_manifest.tsv \
  --row-index 0 --dry-run
```

Remove `--dry-run` to execute that CPU row. PerioGT preprocessing is also a
local CPU stage: `python scripts/run_local_periogt_prepare_manifest.py --all`.
It must finish before the corresponding PerioGT GPU jobs. Submit neural
training/adaptation scripts only after their external assets are available.
These runners are not downloaders and do not provide missing upstream models.

## 6. Statistics and secondary analyses

After the full frozen-probe prediction manifest is complete, run locally:

```bash
python scripts/run_statistics.py \
  --prediction-manifest runs/manifests/frozen_probes/frozen_probe_manifest.tsv \
  --output results/statistics/frozen_probe \
  --bootstrap-resamples 10000 --bootstrap-seed 2026
```

Outputs include `task_metrics.csv`, `task_effects.csv`, `primary_aggregate.csv`,
and `summary.json`. One arm/seed/task example is a workflow check, not enough
to reproduce the three paired primary comparisons. Keep the supplied
primary family (`A0/B0`, `A_att/A0`, `A_eqs/A_gcl`) and its Holm adjustment.
Do not add retention, metadata, adaptation, or external-reference results to
that family.

`analyze_completed_experiments.py` handles completed adaptation and baseline
manifests. `analyze_mechanism_controls.py`, `analyze_token_efficiency.py`, and
`analyze_property_views.py` handle their separately generated inputs;
`analyze_draft_closure.py` combines the secondary structural and common-record
property analyses. These are analysis entry points, not replacements for the
primary tests. Use each script's `--help` for its required input paths.
Keep original numerical observations, including very large baseline errors;
do not clip them to improve presentation.
