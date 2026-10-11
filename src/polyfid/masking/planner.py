"""Controlled atom-budget masking strategies."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from .atom_groups import AtomTokenGroup, MaskPlan


MASK_STRATEGIES = (
    "uniform",
    "boundary",
    "matched_random",
    "permuted_boundary",
)


class MaskPlanner:
    """Select complete real-atom groups under one common atom budget.

    Boundary plans devote ``boundary_fraction`` of the atom budget to graph
    distances 0, 1 and 2, cycling through these distances so none dominates.
    The remaining budget is sampled from the unselected atoms.  The two
    controls retain exactly the same number of masked atoms: matched-random
    removes the spatial preference, while permuted-boundary applies it after
    shuffling distance labels across eligible atoms.
    """

    def __init__(
        self,
        *,
        mask_probability: float = 0.15,
        boundary_fraction: float = 0.65,
        interior_fraction: float | None = None,
    ) -> None:
        if not 0 < mask_probability <= 1:
            raise ValueError("mask_probability must be in (0, 1]")
        if not 0 <= boundary_fraction <= 1:
            raise ValueError("boundary_fraction must be in [0, 1]")
        if interior_fraction is None:
            interior_fraction = 1.0 - boundary_fraction
        if not 0 <= interior_fraction <= 1:
            raise ValueError("interior_fraction must be in [0, 1]")
        if boundary_fraction + interior_fraction > 1.0 + 1e-12:
            raise ValueError("boundary and interior fractions cannot exceed 1")
        self.mask_probability = float(mask_probability)
        self.boundary_fraction = float(boundary_fraction)
        self.interior_fraction = float(interior_fraction)

    def atom_budget(self, groups: Sequence[AtomTokenGroup]) -> int:
        eligible_count = sum(group.eligible for group in groups)
        if eligible_count == 0:
            return 0
        return max(1, min(eligible_count, math.ceil(eligible_count * self.mask_probability)))

    def plan(
        self,
        groups: Sequence[AtomTokenGroup],
        *,
        strategy: str,
        seed: int,
        atom_budget: int | None = None,
    ) -> MaskPlan:
        if strategy not in MASK_STRATEGIES:
            raise ValueError(f"unknown masking strategy: {strategy}")
        eligible = [index for index, group in enumerate(groups) if group.eligible]
        budget = self.atom_budget(groups) if atom_budget is None else int(atom_budget)
        if budget < 0 or budget > len(eligible):
            raise ValueError("atom_budget must fit the eligible atom groups")
        if budget == 0:
            return MaskPlan(strategy, (), 0)

        rng = random.Random(int(seed))
        if strategy == "uniform":
            selected = rng.sample(eligible, budget)
            roles = {index: "uniform" for index in selected}
        elif strategy == "matched_random":
            distances = {index: groups[index].boundary_distance for index in eligible}
            reference, reference_roles = self._boundary_selection(
                eligible, distances, budget, rng
            )
            selected = self._matched_random_selection(eligible, groups, reference, rng)
            roles = {
                selected_index: reference_roles[reference_index]
                for selected_index, reference_index in zip(
                    selected, reference, strict=True
                )
            }
        else:
            distances = {index: groups[index].boundary_distance for index in eligible}
            if strategy == "permuted_boundary":
                shuffled = list(distances.values())
                rng.shuffle(shuffled)
                distances = dict(zip(eligible, shuffled, strict=True))
            selected, roles = self._boundary_selection(
                eligible, distances, budget, rng
            )
        ordered = tuple(sorted(selected))
        return MaskPlan(
            strategy,
            ordered,
            budget,
            tuple(roles[index] for index in ordered),
        )

    def paired_plans(
        self,
        groups: Sequence[AtomTokenGroup],
        *,
        seed: int,
    ) -> dict[str, MaskPlan]:
        """Create all four strategies with an identical real-atom budget."""

        budget = self.atom_budget(groups)
        return {
            strategy: self.plan(
                groups,
                strategy=strategy,
                seed=int(seed),
                atom_budget=budget,
            )
            for strategy in MASK_STRATEGIES
        }

    def _boundary_selection(
        self,
        eligible: list[int],
        distances: dict[int, int | None],
        budget: int,
        rng: random.Random,
    ) -> tuple[list[int], dict[int, str]]:
        boundary_target = min(budget, round(budget * self.boundary_fraction))
        interior_target = min(
            budget - boundary_target,
            round(budget * self.interior_fraction),
        )
        buckets = {
            distance: [index for index in eligible if distances[index] == distance]
            for distance in (0, 1, 2)
        }
        for values in buckets.values():
            rng.shuffle(values)

        preferred: list[int] = []
        distance_order = [0, 1, 2]
        rng.shuffle(distance_order)
        while len(preferred) < boundary_target:
            added = False
            for distance in distance_order:
                if buckets[distance] and len(preferred) < boundary_target:
                    preferred.append(buckets[distance].pop())
                    added = True
            if not added:
                break

        preferred_set = set(preferred)
        interior = [
            index
            for index in eligible
            if index not in preferred_set and distances[index] not in {0, 1, 2}
        ]
        rng.shuffle(interior)
        selected_interior = interior[:interior_target]
        selected_set = preferred_set | set(selected_interior)
        fallback = [
            index
            for index in eligible
            if index not in selected_set
        ]
        rng.shuffle(fallback)
        selected = preferred + selected_interior
        selected.extend(fallback[: budget - len(selected)])
        roles = {index: "boundary" for index in preferred}
        roles.update({index: "interior" for index in selected_interior})
        roles.update({index: "fallback" for index in selected if index not in roles})
        return selected, roles

    @staticmethod
    def _matched_random_selection(
        eligible: list[int],
        groups: Sequence[AtomTokenGroup],
        reference: list[int],
        rng: random.Random,
    ) -> list[int]:
        """Match the boundary plan's atom-group token widths at random sites."""

        remaining = list(eligible)
        rng.shuffle(remaining)
        selected: list[int] = []
        for reference_index in reference:
            width = len(groups[reference_index].token_indices)
            matched = next(
                (index for index in remaining if len(groups[index].token_indices) == width),
                remaining[0],
            )
            selected.append(matched)
            remaining.remove(matched)
        return selected


__all__ = ["MASK_STRATEGIES", "MaskPlanner"]
