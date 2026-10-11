# Method-to-code map

Paper104 separates equivalence consistency, relative structural discrimination,
property-prediction stability, and predictive accuracy. Improvements in one
are not a substitute for evidence in another. The controlled comparison uses
one ChemBERTa architecture; external baselines answer a different practical
comparison question.

## Seven representation conditions

| Arm | Initialization | PI1M objective | What it isolates |
| --- | --- | --- | --- |
| `C` | Molecular pretrained | None | Fixed molecular reference |
| `B0` | Random | Uniform atom-group MLM | Scratch reference |
| `A0` | Molecular pretrained | Uniform atom-group MLM | Initialization versus B0 |
| `A_att` | Molecular pretrained | Boundary-aware MLM | Boundary allocation versus A0 |
| `A_gcl` | Molecular pretrained | Boundary MLM + generic contrastive loss | Generic relation reference |
| `A_eqs` | Molecular pretrained | Boundary MLM + equivalence/topology relation loss | Polymer-relation recipe versus A_gcl |
| `B_eqs` | Random | Same objective as A_eqs | Secondary initialization comparison |

Definitions live in [`models/arms.py`](../src/polyfid/models/arms.py) and
[`configs/pretraining/arms.yaml`](../configs/pretraining/arms.yaml).
A_gcl and A_eqs share paired observed negatives; their positive construction
differs. This contrast changes positive sources and mixture weights together,
not verification alone. The controls `matched_random_masking`,
`permuted_boundary_masking`, `equivalence_only`, and `topology_negative_only`
are separate three-seed experiments.

## Scientific components

| Component | Main code | Inputs and interpretation |
| --- | --- | --- |
| Operational periodic identity | [`chemistry/structure.py`](../src/polyfid/chemistry/structure.py) | Explicit two-cell graph for linear single-component repeats with two attachment sites |
| Equivalent and edited views | [`chemistry/views.py`](../src/polyfid/chemistry/views.py) | Traversal, alternative periodic cuts, reversal, endpoint relocation, closure rewiring |
| Source standardization | [`data/public_standardization.py`](../src/polyfid/data/public_standardization.py) | Upstream tables and source roles; legacy data conventions remain explicit |
| Project splits and exclusions | [`workflows/data.py`](../src/polyfid/workflows/data.py), [`periodic_data.py`](../src/polyfid/workflows/periodic_data.py) | Preserved held-out roles, periodic duplicates, downstream overlap exclusions |
| Relation selection | [`relations/matching.py`](../src/polyfid/relations/matching.py), [`builders.py`](../src/polyfid/relations/builders.py) | Matched real distinct two-cell graphs; separate explicit-edit controls |
| Atom-group masking | [`masking/planner.py`](../src/polyfid/masking/planner.py), [`chemistry/token_alignment.py`](../src/polyfid/chemistry/token_alignment.py) | Shared atom budget and whole-atom token spans; boundary-distance allocation |
| Pretraining loss | [`training/objectives.py`](../src/polyfid/training/objectives.py), [`forward.py`](../src/polyfid/training/forward.py) | MLM plus weighted projected relation loss |
| Token budget and resume | [`training/progress.py`](../src/polyfid/training/progress.py), [`runner.py`](../src/polyfid/training/runner.py) | Anchor non-padding tokens, predefined checkpoint fractions, saved progress |
| Frozen embedding cache | [`evaluation/encoder.py`](../src/polyfid/evaluation/encoder.py), [`embeddings.py`](../src/polyfid/evaluation/embeddings.py) | Token-mean representations with sample IDs |
| Frozen property heads | [`evaluation/probes.py`](../src/polyfid/evaluation/probes.py), [`scripts/evaluate_downstream.py`](../scripts/evaluate_downstream.py) | Train-only target scaling; validation-selected epoch; held-out predictions |
| Supervised adaptation | [`evaluation/adaptation.py`](../src/polyfid/evaluation/adaptation.py) | Same split joins for LoRA/full fine-tuning, with the seed limitation in [parameters.md](parameters.md) |
| Baseline implementations | [`baselines/`](../src/polyfid/baselines) | Classical/Chemprop common table; TransPolymer/PerioGT/GRIN external references |

The primary relation matcher uses exact atom count and formula, token-length
difference at most five, and Morgan similarity at least 0.60. Primary
negatives differ in the stereo-agnostic two-cell graph; stereochemistry-only
differences are not topology negatives. Near-formula fallback is a development
diagnostic. H-capped and marker-substituted inputs are applicability controls,
not equivalent views or training-positive examples.

The two-cell identity is an operational definition, not a proof of universal
polymer equivalence. A representable identity need not admit a nontrivial
alternative cut. The implementation reports those populations separately.
Relation construction is algorithmic; no expert-labeled benchmark or
synthesizability assessment is supplied.

## Evidence and analysis

| Evidence | Evaluation/analysis code | Interpretation boundary |
| --- | --- | --- |
| Retrieval and topology separation | [`evaluation/fidelity.py`](../src/polyfid/evaluation/fidelity.py) | Fixed matched decoys; cosine, Recall@1, MRR, margin, triplet accuracy |
| Representation invariance | [`evaluation/invariance.py`](../src/polyfid/evaluation/invariance.py) | Traversal and periodic-cut responses are separate; layer drift is descriptive |
| Boundary MLM | [`evaluation/boundary_mlm.py`](../src/polyfid/evaluation/boundary_mlm.py) | Local NLL/accuracy by distance; token-weighted and atom-group-equal summaries |
| Predictive utility | [`analysis/statistics.py`](../src/polyfid/analysis/statistics.py) | Task-equal paired error ratios and the original three primary contrasts |
| Same-record property stability | [`evaluation/property_views.py`](../src/polyfid/evaluation/property_views.py), [`analysis/property_view_statistics.py`](../src/polyfid/analysis/property_view_statistics.py) | Reuses fitted heads; equivalent-view variation is not predictive accuracy |
| Structural/property bridge | [`analysis/draft_closure_statistics.py`](../src/polyfid/analysis/draft_closure_statistics.py) | Secondary structural intervals and common-eligible-record comparisons |
| Adaptation/baseline outcomes | [`analysis/posttraining_metrics.py`](../src/polyfid/analysis/posttraining_metrics.py), [`posttraining_bootstrap.py`](../src/polyfid/analysis/posttraining_bootstrap.py) | Preserves seed meanings and comparison categories |
| Learning dynamics and controls | [`analysis/token_efficiency_statistics.py`](../src/polyfid/analysis/token_efficiency_statistics.py), [`mechanism_control_statistics.py`](../src/polyfid/analysis/mechanism_control_statistics.py) | Secondary token checkpoints and three-seed mechanism contrasts |
| Workflow cost | [`evaluation/cost_profiling.py`](../src/polyfid/evaluation/cost_profiling.py), [`analysis/cost_accounting.py`](../src/polyfid/analysis/cost_accounting.py) | Separates reusable pretraining, task fitting, inference, and study repeats |

The primary full-label aggregate excludes molecular-retention endpoints,
metadata-augmented PE-I, and external references. PE-I structure-only and
structure-plus-metadata are distinct information conditions. OpenPoly is
disabled in the supplied formal configuration and is not silently restored by
a documentation example. Large finite baseline errors remain observations;
their size is not grounds for dropping them from the tables.

The source contains analytical and figure-building helpers, but the public
repository does not include the completed predictions or manuscript assets
needed to regenerate paper figures immediately after cloning.
