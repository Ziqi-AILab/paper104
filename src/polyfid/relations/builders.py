"""Build positive and supplemental negative relation views."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from polyfid.chemistry import (
    generate_closure_rewirings,
    generate_endpoint_relocations,
    generate_equivalent_periodic_cuts,
    generate_randomized_smiles,
    generate_reversed_repeat_view,
)

from .schema import (
    RelationTriplet,
    StructureView,
    stereo_agnostic_graphs_differ,
)


def make_anchor_view(
    sample_id: str,
    psmiles: str,
    *,
    include_profile: bool = True,
) -> StructureView:
    return StructureView.from_psmiles(
        sample_id,
        psmiles,
        relation="anchor",
        source="observed",
        include_profile=include_profile,
    )


def build_positive_views(
    anchor: StructureView,
    *,
    randomized_count: int = 10,
    periodic_max_views: int = 10,
    include_reverse: bool = True,
    seed: int = 42,
) -> tuple[StructureView, ...]:
    """Create traversal and repeat-window positives for one eligible anchor."""

    if not anchor.eligible:
        return ()
    positives: list[StructureView] = []
    for index, text in enumerate(
        generate_randomized_smiles(
            anchor.psmiles, count=randomized_count, seed=seed
        )
    ):
        view = StructureView.from_psmiles(
            f"{anchor.sample_id}::randomized_traversal:{index}",
            text,
            relation="randomized_traversal",
            source="verified_transformation",
            metadata={"anchor_sample_id": anchor.sample_id, "view_index": index},
            include_profile=False,
        )
        if view.repeat_equivalence_id == anchor.repeat_equivalence_id:
            positives.append(view)

    cut_index = 0
    for periodic in generate_equivalent_periodic_cuts(
        anchor.psmiles, max_views=periodic_max_views, seed=seed
    ):
        if periodic.transformation == "reference":
            continue
        view = StructureView.from_psmiles(
            f"{anchor.sample_id}::alternative_periodic_cut:{cut_index}",
            periodic.psmiles,
            relation="alternative_periodic_cut",
            source="verified_transformation",
            metadata={
                "anchor_sample_id": anchor.sample_id,
                "cut_atom_indices": periodic.cut_atom_indices,
                "paired_cut_edges": periodic.paired_cut_edges,
                "cut_bond_type": periodic.cut_bond_type,
                "periodic_identity_status": periodic.periodic_identity_status,
            },
            include_profile=False,
        )
        if view.repeat_equivalence_id == anchor.repeat_equivalence_id:
            positives.append(view)
            cut_index += 1
    if include_reverse:
        reversal = generate_reversed_repeat_view(anchor.psmiles)
        if reversal is not None:
            view = StructureView.from_psmiles(
                f"{anchor.sample_id}::reverse",
                reversal.psmiles,
                relation="reverse_when_non_noop",
                source="verified_transformation",
                metadata={"anchor_sample_id": anchor.sample_id},
                include_profile=False,
            )
            if view.repeat_equivalence_id == anchor.repeat_equivalence_id:
                positives.append(view)
    return tuple(positives)


def build_supplemental_negative_views(
    anchor: StructureView,
    *,
    max_endpoint_relocations: int = 10,
    max_closure_rewirings: int = 10,
    seed: int = 42,
) -> tuple[StructureView, ...]:
    """Create verified synthetic controls, kept separate from observed negatives."""

    if not anchor.eligible:
        return ()
    negatives: list[StructureView] = []
    for index, control in enumerate(
        generate_endpoint_relocations(
            anchor.psmiles, max_views=max_endpoint_relocations, seed=seed
        )
    ):
        metadata = asdict(control)
        metadata["anchor_sample_id"] = anchor.sample_id
        view = StructureView.from_psmiles(
            f"{anchor.sample_id}::endpoint_relocation:{index}",
            control.psmiles,
            relation="endpoint_relocation",
            source="synthetic_control",
            metadata=metadata,
            include_profile=False,
        )
        if stereo_agnostic_graphs_differ(
            anchor.repeat_equivalence_id, view.repeat_equivalence_id
        ):
            negatives.append(view)
    for index, control in enumerate(
        generate_closure_rewirings(
            anchor.psmiles, max_views=max_closure_rewirings, seed=seed
        )
    ):
        metadata = asdict(control)
        metadata["anchor_sample_id"] = anchor.sample_id
        view = StructureView.from_psmiles(
            f"{anchor.sample_id}::closure_rewiring:{index}",
            control.psmiles,
            relation="closure_rewiring",
            source="synthetic_control",
            metadata=metadata,
            include_profile=False,
        )
        if stereo_agnostic_graphs_differ(
            anchor.repeat_equivalence_id, view.repeat_equivalence_id
        ):
            negatives.append(view)
    return tuple(negatives)


def assemble_triplets(
    anchor: StructureView,
    positives: Sequence[StructureView],
    negatives: Sequence[StructureView],
) -> tuple[RelationTriplet, ...]:
    """Pair supplied views and retain chemically valid relation triplets."""

    triplets = [
        RelationTriplet(anchor=anchor, positive=positive, negative=negative)
        for positive in positives
        for negative in negatives
    ]
    return tuple(triplet for triplet in triplets if triplet.eligible)


__all__ = [
    "assemble_triplets",
    "build_positive_views",
    "build_supplemental_negative_views",
    "make_anchor_view",
]
