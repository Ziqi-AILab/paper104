# Public implementation notes

The experimental source is taken from the v0.3 reproduction code package.
This publication copy makes local paths configurable and provides a small
numerical-results entry point. It is not a new experiment, a rewritten method,
or a claim that the absent raw data and trained models can be reproduced from
the clone alone.

## Publication-specific changes

- `ProjectPaths` accepts an optional `exclusion_root`; `exclusion_path()`
  resolves the two externally supplied historical exclusion files. Existing
  project/data/model/checkpoint roles are unchanged. The default exclusion
  location is `assets/exclusions`.
- `scripts/configure_local_paths.py --exclusion-root` configures that location
  alongside the data, model, and checkpoint roots. It writes local
  `configs/paths.yaml`; it does not download assets or submit jobs.
- Raw reconstruction and project periodic preparation now report a clear
  missing-file error when the corresponding external exclusion list is
  absent. The original exclusion semantics remain in place; no empty-list or
  newly inferred replacement is introduced.
- Datasets, exclusion lists, external source trees, trained weights, embedding
  caches, and sample-level predictions are omitted. The required layouts and
  availability limits are documented in [data.md](data.md) and
  [models.md](models.md).
- `scripts/reproduce_results.py` recomputes numerical aggregates and paired
  point estimates from the released metric summaries. It retains saved
  bootstrap intervals, p-values, and Holm adjustments; it does not rerun
  hierarchical resampling without the absent per-record inputs. See
  [results.md](results.md) for the exact inputs and outputs.
- Lightweight tests cover external path resolution and numerical-result
  aggregation. Source-only tests remain local CPU work; passing them is not
  a claim that a fresh GPU environment or full experimental reproduction has
  been validated.

The retained `polyfid paper` interface calls `scripts/build_paper.py`.
The optional legacy `paper/manuscript.tex` is not distributed, so that command
reports the missing manuscript explicitly. It is not part of the normal
public workflow and does not reconstruct a manuscript from the summary CSVs.

## Protocol details deliberately unchanged

The common pretraining and frozen-head recipes, model checkpoint identity,
maximum lengths, source split conventions, relation objectives, and statistical
comparison families retain their original meanings. Formal head fits still
use validation-based early stopping after the common learning rate has been
selected on development representations. There is no added final-test
approval procedure.

The archived adaptation path creates the property head and low-rank matrices
before setting the training seed. Consequently, the adaptation seed controls
training-time randomness and shuffling but not exact readout initialization.
This known limitation remains unchanged and is described in
[parameters.md](parameters.md); frozen-head training initializes after its
own seed is set and is not affected by that limitation.

Very large finite Ridge errors and other extreme recorded baseline values are
preserved. Publication preparation neither clips them nor reinterprets them
as missing runs. External-reference methods remain distinct from the
backbone-controlled conditions, and the original three-comparison primary
Holm family is not expanded by secondary analyses.

Relation eligibility follows the implemented graph and matching definitions.
The configuration records that manual relation review was not performed and
is not evidence; review is not required to run training. This is an ordinary
scientific implementation, not an expert-labeled benchmark.

No repository-wide license grant has been selected by the author. Upstream
source/model/data terms remain applicable to their own assets; their licenses
must not be treated as a license for this entire repository.
