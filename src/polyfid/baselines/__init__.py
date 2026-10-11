"""Same-split classical and external baseline adapters."""

from .classical import (
    MetadataFeatureEncoder,
    ecfp_features,
    metadata_features,
    nearest_neighbor_predictions,
    normalize_structure_text,
    rdkit_2d_features,
    structure_smiles,
    train_classical,
)
from .external import (
    detect_external_assets,
    materialize_external_fold,
    read_external_predictions,
)
from .grin import GRINRunConfig, probe_grin_load, run_grin_fold
from .periogt import (
    PerioGTRunConfig,
    prepare_periogt_cache,
    probe_periogt_load,
    run_periogt_fold,
)

__all__ = [
    "detect_external_assets",
    "materialize_external_fold",
    "read_external_predictions",
    "GRINRunConfig",
    "probe_grin_load",
    "run_grin_fold",
    "PerioGTRunConfig",
    "prepare_periogt_cache",
    "probe_periogt_load",
    "run_periogt_fold",
    "MetadataFeatureEncoder",
    "ecfp_features",
    "metadata_features",
    "nearest_neighbor_predictions",
    "normalize_structure_text",
    "rdkit_2d_features",
    "structure_smiles",
    "train_classical",
]
