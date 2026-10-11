from polyfid.masking import AtomTokenGroup, MaskPlanner, apply_mask_plan


def _groups():
    return [
        AtomTokenGroup(0, (1,), boundary_distance=0),
        AtomTokenGroup(1, (2, 3), boundary_distance=1),
        AtomTokenGroup(2, (4,), boundary_distance=2),
        AtomTokenGroup(3, (5,), boundary_distance=3),
        AtomTokenGroup(4, (6,), boundary_distance=4),
        AtomTokenGroup(5, (7,), boundary_distance=5),
        AtomTokenGroup(6, (8,), boundary_distance=0, eligible=False),
    ]


def test_all_control_strategies_share_atom_budget_and_exclude_marker():
    planner = MaskPlanner(mask_probability=0.5, boundary_fraction=2 / 3)
    plans = planner.paired_plans(_groups(), seed=42)
    assert set(plans) == {
        "uniform",
        "boundary",
        "matched_random",
        "permuted_boundary",
    }
    assert {plan.atom_budget for plan in plans.values()} == {3}
    assert {len(plan.selected_group_indices) for plan in plans.values()} == {3}
    assert all(6 not in plan.selected_group_indices for plan in plans.values())
    assert all(
        len(plan.selection_roles) == len(plan.selected_group_indices)
        for plan in plans.values()
    )
    assert set(plans["boundary"].selection_roles).issubset(
        {"boundary", "interior", "fallback"}
    )
    assert set(plans["uniform"].selection_roles) == {"uniform"}


def test_boundary_plan_balances_available_distances_zero_one_two():
    planner = MaskPlanner(mask_probability=0.5, boundary_fraction=1.0)
    plan = planner.plan(_groups(), strategy="boundary", seed=7)
    distances = {_groups()[index].boundary_distance for index in plan.selected_group_indices}
    assert distances == {0, 1, 2}


def test_whole_multitoken_atom_is_corrupted_in_one_plan():
    groups = _groups()
    planner = MaskPlanner(mask_probability=1 / 6)
    plan = planner.plan(groups, strategy="uniform", seed=4, atom_budget=1)
    # Force the known multi-token atom to make the masking invariant explicit.
    plan = type(plan)(plan.strategy, (1,), 1)
    clean = list(range(10, 19))
    corrupted, labels = apply_mask_plan(clean, groups, plan, mask_token_id=99)
    assert corrupted[2:4] == [99, 99]
    assert labels[2:4] == clean[2:4]
    assert sum(label != -100 for label in labels) == 2


def test_mask_plans_are_seed_reproducible():
    planner = MaskPlanner(mask_probability=0.5)
    first = planner.paired_plans(_groups(), seed=123)
    second = planner.paired_plans(_groups(), seed=123)
    assert first == second
