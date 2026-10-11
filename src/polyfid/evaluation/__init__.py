"""Intrinsic structural-fidelity and downstream utility evaluation."""

from .adaptation import prepare_downstream_frame
from .boundary_mlm import evaluate_atom_group_mlm, summarise_boundary_tokens
from .fidelity import (
    equivalence_metrics,
    evaluate_relation_embeddings,
    evaluate_topology_controls,
    topology_metrics,
)
from .invariance import (
    build_invariance_views,
    generate_atom_order_permutations,
    score_applicability_controls,
    score_layer_drift,
    score_view_consistency,
)
from .metrics import compute_metric
from .predictions import load_predictions, write_predictions
from .probes import (
    ProbeState,
    load_probe_state,
    predict_with_probe,
    save_probe_state,
    train_frozen_probe,
)
from .probe_selection import (
    PROPERTY_HEAD_LEARNING_RATES,
    select_property_head_learning_rate,
    write_property_head_selection,
)
from .property_views import (
    EquivalentViewPredictionResult,
    predict_equivalent_property_views,
    predict_property_applicability_controls,
    write_applicability_control_predictions,
    write_equivalent_view_predictions,
)

__all__ = [
    "build_invariance_views",
    "generate_atom_order_permutations",
    "compute_metric",
    "equivalence_metrics",
    "EquivalentViewPredictionResult",
    "evaluate_atom_group_mlm",
    "evaluate_relation_embeddings",
    "evaluate_topology_controls",
    "load_predictions",
    "load_probe_state",
    "predict_equivalent_property_views",
    "predict_property_applicability_controls",
    "predict_with_probe",
    "ProbeState",
    "prepare_downstream_frame",
    "score_applicability_controls",
    "score_layer_drift",
    "score_view_consistency",
    "save_probe_state",
    "summarise_boundary_tokens",
    "topology_metrics",
    "train_frozen_probe",
    "PROPERTY_HEAD_LEARNING_RATES",
    "select_property_head_learning_rate",
    "write_property_head_selection",
    "write_applicability_control_predictions",
    "write_equivalent_view_predictions",
    "write_predictions",
]
