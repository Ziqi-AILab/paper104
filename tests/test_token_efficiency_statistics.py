from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyfid.analysis.token_efficiency_statistics import (
    compute_seed_curves,
    load_paired_token_efficiency_samples,
    paired_token_efficiency_bootstrap,
    select_formal_token_efficiency_grid,
    write_token_efficiency_analysis_outputs,
)


def _write_result_cell(
    root: Path,
    *,
    method: str,
    seed: int,
    fraction: float,
    checkpoint_tokens: int,
    nll: tuple[float, float, float],
    changed_target: bool = False,
) -> Path:
    output = root / method / f"seed-{seed}" / f"fraction-{int(fraction * 100):03d}"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2"],
            "atom_group_id": ["s1:g0", "s1:g1", "s2:g0"],
            "token_position": [1, 3, 2],
            "target_token_id": [10, 11, 13 if changed_target else 12],
            "predicted_token_id": [10, 4, 5],
            "nll": list(nll),
            "correct": [True, False, False],
            "method": [method] * 3,
            "representation_seed": [seed] * 3,
            "checkpoint_fraction": [fraction] * 3,
            "checkpoint_tokens": [checkpoint_tokens] * 3,
        }
    )
    frame.to_parquet(output / "token_metrics.parquet", index=False)
    # This mirrors the actual evaluator schema: mask_seed belongs to the
    # generated matrix, while metrics.json records the observed token counter.
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "method": method,
                "representation_seed": seed,
                "checkpoint_fraction": fraction,
                "checkpoint_tokens": checkpoint_tokens,
                "token_mean_nll": float(np.mean(nll)),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _manifest_rows(
    root: Path,
    *,
    seeds: tuple[int, ...],
    fractions: tuple[float, ...],
    changed_target: bool = False,
) -> pd.DataFrame:
    rows = []
    for method in ("A0", "B0"):
        for seed in seeds:
            for fraction_index, fraction in enumerate(fractions):
                token_offset = 0 if method == "A0" else 7
                checkpoint_tokens = (fraction_index + 1) * 1000 + token_offset
                nll = (
                    (1.0, 3.0, 9.0)
                    if method == "A0"
                    else (2.0, 4.0, 10.0)
                )
                output = _write_result_cell(
                    root,
                    method=method,
                    seed=seed,
                    fraction=fraction,
                    checkpoint_tokens=checkpoint_tokens,
                    nll=nll,
                    changed_target=(
                        changed_target
                        and method == "B0"
                        and seed == seeds[0]
                        and fraction == fractions[0]
                    ),
                )
                rows.append(
                    {
                        "arm": method,
                        "representation_seed": seed,
                        "checkpoint_fraction": fraction,
                        "checkpoint_percent": int(round(100 * fraction)),
                        "mask_seed": 10407,
                        "output": str(output),
                    }
                )
    return pd.DataFrame(rows)


def test_sample_cluster_summary_retains_global_token_mean(tmp_path: Path) -> None:
    manifest = _manifest_rows(tmp_path, seeds=(42,), fractions=(0.01,))
    paired, inventory = load_paired_token_efficiency_samples(
        manifest, seeds=(42,), fractions=(0.01,)
    )
    assert len(inventory) == 2
    assert set(paired["supervised_tokens"]) == {1, 2}
    curve = compute_seed_curves(paired).iloc[0]
    assert curve["candidate_token_mean_nll"] == pytest.approx(13.0 / 3.0)
    assert curve["baseline_token_mean_nll"] == pytest.approx(16.0 / 3.0)
    # A sample-equal average would be 5.5; token positions are not bootstrap
    # units, but the stated token-NLL estimand remains token weighted.
    assert curve["candidate_token_mean_nll"] != pytest.approx(5.5)


def test_pairing_requires_the_same_fixed_mask_targets(tmp_path: Path) -> None:
    manifest = _manifest_rows(
        tmp_path, seeds=(42,), fractions=(0.01,), changed_target=True
    )
    with pytest.raises(ValueError, match="fixed-mask targets"):
        load_paired_token_efficiency_samples(
            manifest, seeds=(42,), fractions=(0.01,)
        )


def _constant_effect_samples() -> pd.DataFrame:
    rows = []
    for seed in (42, 123):
        for fraction_index, fraction in enumerate((0.01, 0.10, 1.00)):
            for sample_index, tokens in enumerate((1, 2, 4, 3)):
                baseline_nll = 5.0 - 0.25 * fraction_index
                candidate_nll = baseline_nll - 1.0
                rows.append(
                    {
                        "sample_id": f"s{sample_index}",
                        "representation_seed": seed,
                        "checkpoint_fraction": fraction,
                        "candidate_checkpoint_tokens": (fraction_index + 1) * 1000,
                        "baseline_checkpoint_tokens": (fraction_index + 1) * 1000,
                        "candidate_nll_sum": candidate_nll * tokens,
                        "baseline_nll_sum": baseline_nll * tokens,
                        "supervised_tokens": tokens,
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_resamples_samples_as_clusters_and_preserves_curve_pairing() -> None:
    fractions, seed_auc, aggregate = paired_token_efficiency_bootstrap(
        _constant_effect_samples(), n_resamples=200, seed=7, chunk_size=20
    )
    assert np.allclose(fractions["delta_nll"], -1.0)
    assert np.allclose(fractions["delta_nll_ci_lower"], -1.0)
    assert np.allclose(fractions["delta_nll_ci_upper"], -1.0)
    assert np.allclose(seed_auc["log_token_average_delta_nll"], -1.0)
    assert aggregate.iloc[0]["log_token_average_delta_nll"] == pytest.approx(-1.0)
    assert aggregate.iloc[0]["probability_a0_lower_log_token_average_nll"] == 1.0


def test_log_token_auc_interpolates_to_a_common_observed_interval() -> None:
    rows = []
    candidate_tokens = np.asarray([100, 1000, 10_000])
    baseline_tokens = np.asarray([200, 2000, 20_000])
    for fraction, candidate_token, baseline_token in zip(
        (0.01, 0.10, 1.00), candidate_tokens, baseline_tokens, strict=True
    ):
        rows.append(
            {
                "sample_id": "s0",
                "representation_seed": 42,
                "checkpoint_fraction": fraction,
                "candidate_checkpoint_tokens": candidate_token,
                "baseline_checkpoint_tokens": baseline_token,
                "candidate_nll_sum": 6.0 - 0.1 * np.log(candidate_token),
                "baseline_nll_sum": 6.0 - 0.1 * np.log(baseline_token),
                "supervised_tokens": 1,
            }
        )
    _, seed_auc, aggregate = paired_token_efficiency_bootstrap(
        pd.DataFrame(rows), n_resamples=20, seed=3, chunk_size=5
    )
    assert seed_auc.iloc[0]["common_token_lower"] == pytest.approx(200.0)
    assert seed_auc.iloc[0]["common_token_upper"] == pytest.approx(10_000.0)
    assert seed_auc.iloc[0]["log_token_average_delta_nll"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert aggregate.iloc[0]["log_token_average_delta_nll"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_writer_emits_compact_analysis_tables(tmp_path: Path) -> None:
    fractions = (0.01, 0.10, 1.00)
    manifest = _manifest_rows(tmp_path / "metrics", seeds=(42, 123), fractions=fractions)
    manifest_path = tmp_path / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    output = tmp_path / "statistics"
    summary = write_token_efficiency_analysis_outputs(
        manifest_path,
        output,
        n_resamples=20,
        bootstrap_seed=5,
        bootstrap_chunk_size=5,
        seeds=(42, 123),
        fractions=fractions,
    )
    assert summary["result_cells"] == 12
    assert summary["bootstrap_hierarchy"] == "representation seed -> validation sample"
    for filename in (
        "result_inventory.csv",
        "seed_curves.csv",
        "fraction_effects.csv",
        "seed_auc.csv",
        "curve_summary.csv",
        "summary.json",
    ):
        assert (output / filename).is_file()


def test_formal_grid_reports_missing_cells() -> None:
    manifest = pd.DataFrame(
        [
            {
                "arm": "A0",
                "representation_seed": 42,
                "checkpoint_fraction": 0.01,
            }
        ]
    )
    with pytest.raises(ValueError, match="grid is incomplete"):
        select_formal_token_efficiency_grid(
            manifest, seeds=(42,), fractions=(0.01,)
        )
