# Paper104 / PolyFiD

**Structural Fidelity in Polymer Cheminformatics: A Controlled Study of Molecular-to-Polymer Continued Pretraining**

Research code for a controlled study of molecular initialization, boundary masking, and polymer-relation supervision.
Equivalent-view consistency, relative structure discrimination, prediction stability, and property error are measured separately.

## Overview

The scientific object is a linear, single-component, two-attachment repeat-unit PSMILES.
An operational two-cell graph defines relation labels, not general physical equivalence of infinite polymers.

Six trained conditions and the fixed molecular reference C share a ChemBERTa architecture and tokenizer.
The primary contrasts are A0/B0 (initialization), A_att/A0 (mask location), and A_eqs/A_gcl (positive source/mixture).
A_eqs and A_gcl share observed negatives and the relation objective.
Controls, retention, adaptation, and external baselines provide separate analyses.

## Architecture

```text
Repeat-unit PSMILES → atom-token groups and periodic relations
                   → ChemBERTa MLM (+ relation projection when applicable)
                   → frozen mean-pooled representations → property MLP
                   → separate structural and prediction-view evaluations
```

| Component | Study configuration |
| --- | --- |
| Backbone | DeepChem/ChemBERTa-100M-MLM; 12 layers, hidden size 768 |
| Continued pretraining | Full backbone, length 256, BF16, 3 epochs, anchor batch 256 |
| Relation objective | MLM + 0.1 × relation loss; temperature 0.1; projection 768→768→256 |
| Frozen property head | 768→256→output; GELU; dropout 0.1 |
| Formal repetitions | Representation seeds 42, 123, 777, 2025, 2026; nested head seeds 11, 22, 33 |

“100M” denotes the upstream molecular corpus size, not the model parameter count.
See [parameters](docs/parameters.md) and the [method-to-code map](docs/method_map.md).

## Installation

```bash
git clone https://github.com/Ziqi-AILab/paper104.git
cd paper104
conda env create --prefix .conda/env --file environment.yml
conda activate ./.conda/env
python -m pip install -e '.[test]'
python -m polyfid.cli --help
```

Use Python 3.10 and a PyTorch build compatible with your GPU runtime.
`requirements.txt` records the study stack; its CUDA graph packages are not a universal CPU recipe.
[External-model setup](docs/models.md) explains baseline prerequisites.
Fresh-environment GPU installation and full retraining were not performed during repository preparation.

## Data Preparation

Data and weights are external: see [sources, layouts, and formats](docs/data.md) for PI1M,
VIPEA, TransPolymer tasks, molecular-retention data, and ChemBERTa.
Exact historical reconstruction requires two frozen exclusion lists; neither these nor trained Paper104 weights
have a public download. New upstream downloads are not asserted to reproduce old splits.

```bash
export DATA_ROOT=/path/to/paper104_data
export MODEL_ROOT=/path/to/chemistry_models
export CHECKPOINT_ROOT=/path/to/paper104_checkpoints
export EXCLUSION_ROOT=/path/to/paper104_frozen_exclusions
python scripts/configure_local_paths.py --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
  --exclusion-root "$EXCLUSION_ROOT"
```

After supplying those assets, these commands **execute** local CPU preparation.
Use a fresh `assets/rebuilt_v1` and an absent destination `standardized/v1`:

```bash
python scripts/rebuild_public_data.py all --output-root assets/rebuilt_v1 --seed 42
mkdir -p "$DATA_ROOT/standardized"
mv assets/rebuilt_v1 "$DATA_ROOT/standardized/v1"
python scripts/prepare_project_data.py
python scripts/build_relations.py --partition train --target-triplets 10000 --seed 42
python scripts/build_relations.py --partition validation --target-triplets 1951 --seed 42
python scripts/build_relations.py --partition test --target-triplets 1880 --seed 42
python scripts/build_evaluation_inputs.py
```

The historical validation search retained 1,949 admissible triplets. Splits precede augmentation.
OpenPoly is not a formal endpoint; its historical exclusions are retained.

## Training and Evaluation

The example below is **one A_eqs representation and one EGC fold**, not the full matrix.
The selected pretraining LR is `1e-4`; the frozen-head LR is `1e-3`.
`configs/pretraining/selected_recipe.yaml` and `configs/downstream/selected_property_head.yaml`
record validation-only development selection (seeds 13 and 37). Reuse these recipes
for formal seeds; do not select configurations on final test results.

Run continued pretraining inside an allocated **single RTX 4090** job:

```bash
python scripts/run_pretraining.py --arm A_eqs --seed 42 --learning-rate 1e-4 \
  --boundary-fraction 0.65 --interior-fraction 0.25 --stage formal
export ENCODER="$CHECKPOINT_ROOT/pretraining/A_eqs/seed-42/final_model"
```

After training, execute these encoder/fidelity commands inside a **single A800** job:

```bash
python scripts/cache_embeddings.py \
  --dataset data/derived/evaluation_inputs/downstream_structure_only.parquet \
  --checkpoint "$ENCODER" --method A_eqs --seed 42 \
  --output runs/embeddings/A_eqs_seed-42.npz
python scripts/evaluate_fidelity.py --relations data/relations/test/pi1m_triplets.parquet \
  --checkpoint "$ENCODER" --method A_eqs --seed 42 \
  --output results/new_runs/fidelity_A_eqs_seed-42
```

Train the frozen head **locally on CPU** after caching; validation early stopping selects the head for each seed:

```bash
python scripts/evaluate_downstream.py \
  --dataset "$DATA_ROOT/standardized/v1/transpolymer/egc.parquet" \
  --split-manifest data/splits/transpolymer_egc.parquet \
  --embeddings runs/embeddings/A_eqs_seed-42.npz --task EGC --protocol group_ood \
  --fold 0 --learning-rate 1e-3 --output results/new_runs/A_eqs_EGC_fold-0.parquet
```

[Full workflow](docs/workflow.md) covers development selection, other seeds/tasks,
property views, LoRA/full adaptation, retention, controls, and external baselines.
Matrix generators **write commands/manifests; they do not submit jobs**. Adapt their
site-specific Slurm settings before submission; CPU-only work stays local. Resume
is explicit (`run_pretraining.py --resume-existing`), not an automatic supervisor.
Historical adaptation initialization limitations remain documented and uncorrected.

## Results Reproduction

Recompute supported summaries and point estimates from the released lightweight
fit/seed tables, and render the **already saved** confidence limits:

```bash
python scripts/reproduce_results.py --output results/reproduced
```

This needs no datasets, checkpoints, inference, or GPU. It does **not** rerun the
row-level hierarchical bootstrap; that requires excluded per-record predictions.
See [results and estimands](docs/results.md) for source precision and exact coverage.

A CPU reference on the same prepared EGC fold can be run independently:

```bash
python scripts/evaluate_classical_baselines.py run --method rdkit-ridge \
  --dataset "$DATA_ROOT/standardized/v1/transpolymer/egc.parquet" \
  --split-manifest data/splits/transpolymer_egc.parquet --task EGC --protocol group_ood \
  --fold 0 --seeds 0 --output results/new_runs/rdkit_EGC_fold-0.parquet
```

For **new complete runs**, use `scripts/run_statistics.py` with the generated
prediction manifest, as shown in the workflow. One example fit cannot reproduce
the three primary aggregate comparisons. Their Holm family remains unchanged;
structural and same-record intervals remain secondary analyses.

## Project Structure

```text
src/polyfid/   chemistry, relations, masking, models, training, evaluation, analysis
configs/      data definitions, selected recipes, evaluation and analysis settings
scripts/      preparation, training, inference, local statistics, matrix generation
tests/        CPU component and statistical tests
results/      lightweight completed numerical sources; no individual predictions
docs/         workflow, parameters, method map, assets, testing, result interpretation
```

See [testing](docs/testing.md) and [implementation notes](docs/implementation.md) for checks and historical limits.
Raw data, weights, caches, embeddings, per-molecule predictions, logs, artwork, and manuscript files are excluded.

## Acknowledgments

The implementation uses PyTorch, Hugging Face Transformers, RDKit, and DeepChem's ChemBERTa checkpoint.
Data and models retain their attribution: PI1M, Aldeghi/VIPEA, TransPolymer, Chemprop, PerioGT, and GRIN.
See [data sources](docs/data.md) and [model sources](docs/models.md); upstream methods are not renamed as original contributions.
A_gcl is a generic-positive control, not a faithful PolyCL reproduction. Upstream licenses remain applicable;
this publication does not assign a new license to third-party data or model weights.
