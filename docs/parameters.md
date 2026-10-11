# Parameters and selection

The executed command and generated manifest specify a run. YAML files describe
the intended settings, but not every runner reads every YAML field. In
particular, do not rely on generic CLI defaults when reproducing a selected
recipe. The main entries are in [workflow.md](workflow.md).

## Continued pretraining

| Setting | Recorded recipe | Implementation/configuration |
| --- | --- | --- |
| Backbone | ChemBERTa checkpoint in `chemberta_100m_mlm` | `configs/pretraining/core.yaml`, external model assets |
| Maximum length / precision | 256 tokens / BF16 | `core.yaml`; pretraining runner |
| Training duration | 3 epochs | `core.yaml` |
| Effective anchor batch | 256 | 32 × 8 for MLM arms; 8 × 32 for relation arms |
| Optimizer | AdamW, weight decay 0.01 | Training runner |
| Schedule | Linear decay, warmup fraction 0.05 | `core.yaml` |
| Selected warm/scratch learning rate | Both `1e-4` | `configs/pretraining/selected_recipe.yaml` |
| Atom-group mask budget | 0.15 | `core.yaml`, masking planner |
| Boundary/interior/fallback proportions | 0.65 / 0.25 / 0.10 | Selected recipe; planner |
| Boundary distances | 0, 1, 2 with equal distance sampling | Masking planner |
| Relation projection | 768 → 768 → 256, GELU, L2 normalization | `models/projection.py` |
| Relation weight / temperature | 0.10 / 0.10 | Arm definitions and objective |
| Token checkpoint fractions | 0.01, 0.05, 0.10, 0.25, 0.50, 1.00 | Training progress tracking |

The attachment marker is not the primary atom-masking target. Token budgets
count non-padding anchor tokens; relation views add computation. Equal anchor
budgets therefore do not imply equal total encoder FLOPs or wall time.

Development seeds are 13 and 37. Formal representation seeds are 42, 123, 777,
2025, and 2026. The four mechanism controls use the first three formal seeds.
`C` is one fixed molecular reference, recorded with representation seed zero;
it is not five independent pretraining runs.

The development search compares warm learning rates `{1e-5, 3e-5, 1e-4}`,
scratch rates `{3e-5, 1e-4, 3e-4}`, and boundary/interior/fallback mixtures
`{0.50/0.40/0.10, 0.65/0.25/0.10, 0.80/0.10/0.10}`. Selection uses mean final
validation MLM loss across development seeds. A0 selects the shared warm
rate, B0 the scratch rate, and A_att the mixture at the selected warm rate.

`run_pretraining.py` otherwise defaults warm arms to `3e-5`. Pass
`--learning-rate 1e-4`, or use the generated formal script, to reproduce the
selected recipe. `--max-train-rows`, `--max-validation-rows`, and
`--stop-after-optimizer-steps` are engineering options, not formal settings.

## Frozen property heads

Representations use token-mean pooling. The frozen encoder feeds a
768 → 256 → output head with GELU and dropout 0.10. The common learning rate
is `1e-3`, selected from `{1e-4, 3e-4, 1e-3}` using A0 development
representations at seeds 13 and 37. The selector averages validation losses
within task/protocol/input-view units before averaging those units.

Each formal fit uses head seeds 11, 22, and 33, AdamW, batch size 256, at most
200 epochs, and validation patience 20. Regression targets are standardized
using training rows only and predictions returned to their original scale.
The recorded learning rate remains fixed while validation still selects the
best epoch. `evaluate_downstream.py` defaults to `3e-4`; pass
`--learning-rate 1e-3` or use the recipe-aware manifest runner.

EA/IP/EGC/XC use label fractions 0.01, 0.05, 0.10, 0.25, 0.50, and 1.00;
other formal polymer tasks use 0.10 and 1.00. Below full labels, five nested
group-subsample replicates are used. At full labels, only subsample zero is
used. Train/validation/test roles remain those of the declared protocol.

## Adaptation

`adapt_downstream.py` uses AdamW, learning rate `1e-4`, weight decay 0.01,
batch size 32, maximum length 256, at most 100 epochs, and validation-loss
patience 15. LoRA uses rank 8, alpha 16, and dropout 0.05. It covers all six
trained arms on EA/IP/EGC/XC; full fine-tuning covers A0 and A_eqs on EA/EGC.
Adaptation seeds 11, 22, and 33 are nested within representation seeds.

The retained implementation constructs the head and low-rank matrices before
setting the adaptation training seed. The seed controls training-time draws
and shuffling, but does not make initialization identical across arms or
establish exact seed-only reproduction. This limitation is unchanged; it
does not apply to the separately seeded frozen-head initialization.

## Baseline protocols

| Method | Runtime/training recipe | Comparison role |
| --- | --- | --- |
| ECFP/RDKit Ridge | Local CPU; Ridge alpha 1, train-fitted feature scaling; seed 0 | Same-split controlled common table |
| ECFP/RDKit XGBoost | Local CPU; 500 trees, depth 6, LR 0.05, row/column sampling 0.8; seeds 11/22/33 | Same-split controlled common table |
| Nearest neighbor | Local CPU; ECFP Tanimoto, or train-scaled Euclidean with PE-I metadata; seed 0 | Same-split controlled common table |
| Chemprop | D-MPNN; 30 epochs, batch 50; seeds 11/22/33 | Same-split controlled common table |
| TransPolymer | 20 epochs, patience 5, batch 32, max length 512; encoder/head LR `5e-5`/`1e-4`; upstream float64 | Same-split external reference |
| PerioGT | CPU graph preparation, then 50 epochs, patience 20, batch 32, LR `1e-4`, dropout 0.05 | Same-split external reference |
| GRIN | 500 epochs, patience 50, batch 128, LR `1e-3`, 5 layers, hidden size 300; repetition augmentation, no test-time augmentation | Same-split external reference |

ECFP uses radius 2 and 2,048 dimensions with chirality. RDKit descriptors omit
`Ipc`; undefined descriptor entries are handled as implemented in
`baselines/classical.py`. Do not add result clipping or discard large finite
errors. Neural baseline model selection uses the declared validation
criterion, not test performance.

The three external references use the seven compatible structure-only tasks
EGC/EGB/EEA/EI/XC/EPS/NC, five group-OOD folds, and seeds 11/22/33. Different
upstream pretraining data, tokenizers, graph inputs, and model sizes prevent
treating them as extra backbone-controlled arms. PerioGT needs its upstream
source, pretrained checkpoint, scaler, DGL 1.1.3-compatible runtime,
DGLLife 0.2.9, and Mordred community 2.0.7. GRIN needs its upstream source,
PyG 2.6.1, and CombineMols 1.0.1.0. See [models.md](models.md).

wD-MPNN and PoCo are not implemented formal baselines here. A related graph
model must not be relabeled as either one.

## Evaluation and statistics

Fidelity uses a fixed bank of 99 retrieval decoys per relation, batch size
128, and maximum length 256. Traversal and periodic-cut transformations are
reported separately. Randomized SMILES and atom-order permutation share one
traversal evidence axis. ZINC retention uses the retention partition, mask
seed 10407, and mask probability 0.15.

The primary utility aggregate gives tasks equal weight in paired log error
ratios: EA/IP use MAE under family OOD; other polymer tasks use RMSE under
their declared group-disjoint protocols. Raw task units are retained
(`configs/downstream/task_units.yaml`); PE-I/PE-II supplied logarithmic targets
must not be logged again. The bootstrap uses 10,000 resamples and seed 2026,
with heads nested within representation seeds and the appropriate
family/group/row levels. Only A0/B0, A_att/A0, and A_eqs/A_gcl belong to the
original three-comparison Holm family. Structural, adaptation, same-record
view, token-efficiency, and cost analyses remain secondary.
