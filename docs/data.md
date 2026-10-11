# Data sources, preparation, and formats

Datasets, record-level exclusion lists, and prepared research tables are not
uploaded with this repository. The retained numerical summaries are described
separately in [results.md](results.md). All acquisition, standardization,
splitting, and relation construction below use
`[CPU → local, no sbatch]`; they do not require a GPU allocation.

## Availability and historical reproduction

[data_sources.yaml](data_sources.yaml) records the public acquisition routes,
source versions, use notes, and layouts used by the study. The acquisition
date recorded there is 2026-08-08. Commands below obtain the public sources;
they do not promise that a current upstream branch or file has the historical
contents or row order. Row order matters because source row numbers contribute
to sample identities and supplied split indices.

Exact historical reconstruction also requires two external files:

| Filename under the configured exclusion root | Recorded contents | Used by |
| --- | --- | --- |
| `protected_repeat_groups.txt` | 7,647 legacy standardized repeat identities | Raw-to-standardized PI1M reconstruction |
| `protected_repeat_groups_periodic.txt` | 5,611 two-cell periodic identities | Project-local downstream-overlap exclusion |

Each is UTF-8 plain text, one complete identity per line, without a header.
These are scientific identity strings, not sample indices or property labels.
Configure their directory with `configure_local_paths.py --exclusion-root`;
the default is `assets/exclusions`. Neither file is included, and no public
download location is currently provided. Without these assets, the exact
historical data workflow is blocked even if all public raw datasets have been
downloaded. The scripts report the missing file instead of replacing it with
an empty exclusion set.

The lists retain exclusions from a historical conditional OpenPoly snapshot.
That snapshot is not a formal endpoint in this release, and no stable public
acquisition command or confirmed reuse terms are documented. It is not the
OpenPoly GitHub benchmark. Keep `formal_retained.openpoly: false`; enabling it
does not recover the original data. Newly derived exclusions or new raw
versions define a new preparation, not an asserted identical paper split.

## Obtain the public raw inputs

Run from the repository root. Define paths before using the examples and
choose destinations that do not contain another checkout or existing download.
Git, Git LFS, and curl must be installed. These commands execute downloads;
they are not part of the lightweight numerical reproduction.

```bash
export DATA_ROOT=/path/to/paper104_data
export MODEL_ROOT=/path/to/chemistry_models
export CHECKPOINT_ROOT=/path/to/paper104_checkpoints
export EXCLUSION_ROOT=/path/to/paper104_frozen_exclusions
mkdir -p "$DATA_ROOT/raw"
git clone https://github.com/RUIMINMA1996/PI1M.git "$DATA_ROOT/raw/pi1m"
git clone https://github.com/coleygroup/polymer-chemprop-data.git "$DATA_ROOT/raw/aldeghi_vipea"
mkdir -p "$DATA_ROOT/raw/transpolymer/exports"
git clone https://github.com/ChangwenXu98/TransPolymer.git "$DATA_ROOT/raw/transpolymer/upstream"
git -C "$DATA_ROOT/raw/transpolymer/upstream" lfs pull
cp "$DATA_ROOT/raw/transpolymer/upstream/data/"*.csv "$DATA_ROOT/raw/transpolymer/exports/"
```

PI1M requires `raw/pi1m/PI1M_v2.csv`. VIPEA requires these files beneath
`raw/aldeghi_vipea/datasets/vipea/`:

- `chemprop_inputs/dataset-poly_chemprop.csv`;
- `cv9-monomer_split.pkl`;
- `cv10-random_split.pkl`.

The TransPolymer export directory must contain the official endpoint CSVs,
including the separate supplied PE-I train/test files. Git LFS must retrieve
the real files, not leave LFS pointer text. The original PE-I workbook remains
under the upstream checkout as target-scale documentation; the preparation
reads the exported CSV target values.

Download the molecular-retention endpoints and the ZINC source shard:

```bash
mkdir -p "$DATA_ROOT/raw/molecular_retention" "$DATA_ROOT/raw/zinc20/zinc_processed"
curl -L -o "$DATA_ROOT/raw/molecular_retention/delaney-processed.csv" \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
curl -L -o "$DATA_ROOT/raw/molecular_retention/SAMPL.csv" \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv
curl -L -o "$DATA_ROOT/raw/molecular_retention/Lipophilicity.csv" \
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv
curl -L -o "$DATA_ROOT/raw/zinc20/zinc_processed/smiles_all_00_clean.jsonl.gz" \
  https://huggingface.co/datasets/zpn/zinc20/resolve/main/zinc_processed/smiles_all_00_clean.jsonl.gz
```

Source terms remain those of each upstream dataset. PI1M's repository records
MIT and separately states academic-purpose use; TransPolymer endpoint data
retain their cited source terms, and the ZINC source records ODbL. These notes
are not a new license grant for the data or this repository.

## Rebuild standardized tables

This step requires the two external exclusion files described above. It also
requires the installed project dependencies; external model setup is in
[models.md](models.md). First configure the raw-data location, then rebuild
into a separate destination:

```bash
export REBUILT_DATA_ROOT=/path/to/paper104_rebuilt_data
python scripts/configure_local_paths.py --data-root "$DATA_ROOT" \
  --model-root "$MODEL_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
  --exclusion-root "$EXCLUSION_ROOT"
python scripts/rebuild_public_data.py all \
  --output-root "$REBUILT_DATA_ROOT/standardized/v1" --seed 42
```

`REBUILT_DATA_ROOT` must differ from `DATA_ROOT`. The script refuses an output
inside the configured `DATA_ROOT/standardized/v1`; it does not overwrite that
frozen input location. The stages are `downstream`, `pi1m`, and `zinc20`, or
`all` in that order. The PI1M stage consumes the legacy exclusion list copied
to the rebuild output by the downstream stage.

The expected prepared layout under the new `standardized/v1/` is:

```text
pi1m/pi1m_v2.parquet
aldeghi_vipea/vipea.parquet
aldeghi_vipea/vipea_cv9_roles.parquet
aldeghi_vipea/vipea_butina_roles.parquet
aldeghi_vipea/vipea_random_roles.parquet
transpolymer/{eea,egb,egc,ei,eps,nc,opv,pe_ii,xc}.parquet
transpolymer/pe_i_official.parquet
molecular_retention/{esol,freesolv,lipophilicity}.parquet
zinc20/zinc20_1m.parquet
```

Preparation also writes rejection/metadata sidecars, including
`pi1m/pi1m_v2.metadata.json`, and `public_data_rebuild.json` at the rebuild root.
The metadata acquisition date describes the recorded source snapshot, not a
verification that a new download still matches that snapshot.

Point the rest of the workflow at the completed rebuilt root:

```bash
python scripts/configure_local_paths.py --data-root "$REBUILT_DATA_ROOT" \
  --model-root "$MODEL_ROOT" --checkpoint-root "$CHECKPOINT_ROOT" \
  --exclusion-root "$EXCLUSION_ROOT"
python scripts/prepare_project_data.py
```

Here `REBUILT_DATA_ROOT` becomes the data root used in subsequent commands in
[workflow.md](workflow.md). Alternatively, independently retained standardized
tables may be placed in the same layout and configured directly. Small options
`--pi1m-max-source-rows` and `--zinc-smoke-count` produce differently named smoke
tables; they are not replacements for the full formal input files.

## Data contracts and scientific roles

These are the fields used by the main stages, not an exhaustive listing of
diagnostic columns. Preserve the generated identifiers and join keys.

| Table | Important fields | Meaning |
| --- | --- | --- |
| Standardized PI1M | `sample_id`, `source_row_index`, `canonical_psmiles`, `repeat_equivalence_id`, `split` | Legacy standardized grouping and source partition assignment |
| Project periodic PI1M | Above, plus `source_split`, `source_repeat_equivalence_id`, `periodic_supercell_smiles`, `periodic_identity_status`, `retained_for_paper104` | Explicit two-cell identity, source roles, and retained/excluded rows; `repeat_equivalence_id` becomes the periodic identity |
| VIPEA | `sample_id`, `canonical_text`, `component_canonical_psmiles`, `monomer_pair_group_id`, `ea_ev`, `ip_ev` | Monomer-pair records with EA/IP in eV |
| TransPolymer | `sample_id`, `task`, `label`, `group_id`, `structure_only_text`, `structure_with_metadata_text`, `fold` or official roles | Separate structure-only and metadata-bearing inputs, with unchanged supplied labels |
| Molecular retention | `sample_id`, `canonical_text`, `group_id`, `scaffold_group`, `label`, `split` | Scaffold-grouped ESOL/FreeSolv/Lipophilicity endpoints |
| Project split manifests | `sample_id`, `split`, grouping fields, `fold` or `evaluation_fold`; task/protocol where applicable | Explicit train/validation/test membership for a selected protocol and fold |
| `pi1m_triplets.parquet` | `triplet_id`, `split`, `anchor_*`, `positive_*`, `negative_*`, `positive_relation`, `generic_positive_psmiles`, `retrieval_decoy_bank_indices` | PSMILES, sample IDs, graph identities, matched real negatives, and fixed decoy lookup indices |
| `retrieval_decoy_bank.parquet` | `decoy_bank_index`, `sample_id`, `decoy_psmiles`, `decoy_identity`, `decoy_stereo_agnostic_identity`, `split` | Shared partition-level bank; each triplet stores 99 indices into it |

`data/derived/pi1m_periodic_identity.parquet` contains excluded rows as well
as retained ones. The recorded study retained 935,604 distinct eligible
identities: 842,038 training, 46,784 validation, and 46,782 test. These are
historical reference counts, not guaranteed counts for fresh upstream inputs.
The project preserves held-out source roles and removes lower-priority
overlapping records, rather than relabeling held-out rows as training.

VIPEA manifests distinguish pair-ID, family-OOD, Butina-cluster, and random
protocols. TransPolymer uses five group-based folds except the separate PE-I
official group-disjoint protocol. PE-I sample IDs include their official
source partition because train and test row numbers restart independently.
Its supplied targets are already `log10` conductivity in S/cm; do not apply
another logarithm. PE-II likewise retains the supplied `logCond60` scale.

The project-generated `data/splits/protected_repeat_groups.txt` contains
periodic identities used during training. Despite the shared basename, it is
not the external legacy 7,647-identity list used for raw standardization.

Build relation sets only after splits exist. The recorded final counts are
10,000 training, 1,949 validation, and 1,880 test triplets; generation targets
are not themselves retained counts. Keep `supplemental_topology_controls.parquet`,
`semantic_applicability_controls.parquet`, and `typed_relation_registry.parquet`
separate from primary triplets. Applicability controls are not equivalent
views, and explicit synthetic edits are not measured property labels.
The [workflow](workflow.md) gives relation/input-construction commands.
