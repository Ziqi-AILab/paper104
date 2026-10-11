"""Local-CPU checks for the dependency-free numerical-result entry point."""

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("reproduce_results", ROOT / "scripts/reproduce_results.py")
REPRODUCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPRODUCE)


class ReleasedResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name)
        cls.report = REPRODUCE.reproduce(ROOT / "results", cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_release_matches_retained_values(self):
        self.assertEqual(self.report["pooled_fit_rows"], 33768)
        self.assertEqual(self.report["primary_fold_rows"], 5952)
        self.assertEqual(self.report["primary_pooled_fit_rows"], 1116)
        self.assertEqual(self.report["descriptive_summary_rows"], 4170)
        self.assertEqual(self.report["point_estimates_recomputed"],
                         {"original_primary": 3, "completed": 70, "geometry": 24, "property_bridge": 24})
        self.assertIn("not resampled", self.report["confidence_intervals"])

    def test_rmse_uses_squared_error_pooling(self):
        common = {"source_group": "example", "method": "example", "representation_seed": "42",
                  "head_seed": "11", "task": "example", "protocol": "group_ood", "split": "test",
                  "label_fraction": "1.0", "subsample_id": "0", "input_view": "valid", "metric": "rmse"}
        rows = [{**common, "fold": "0", "n_rows": "1", "value": "1"},
                {**common, "fold": "1", "n_rows": "3", "value": "3"}]
        pooled = REPRODUCE.pool_folds(rows)[0]
        self.assertAlmostEqual(pooled["value"], math.sqrt(7))
        self.assertNotAlmostEqual(pooled["value"], 2.5)
        self.assertEqual(pooled["n_rows"], 4)

    def test_saved_display_conversions(self):
        for family, keys in (("geometry", ("scope", "metric")),
                             ("property_bridge", ("task", "metric"))):
            output_name = "geometry_intervals.csv" if family == "geometry" else "property_bridge_intervals.csv"
            generated = REPRODUCE.read_csv(self.output / output_name)
            saved = REPRODUCE.read_csv(ROOT / "results/statistics/draft_closure" / family / "plot_data.csv")
            indexed = {tuple(row[key] for key in keys): row for row in saved}
            for row in generated:
                reference = indexed[tuple(row[key] for key in keys)]
                for field in ("estimate", "ci_lower", "ci_upper"):
                    self.assertAlmostEqual(float(row[f"display_{field}"]),
                                           float(reference[f"plot_{field}"]), places=10)
                self.assertEqual(row["interval_source"], "saved hierarchical bootstrap; not rerun")

    def test_primary_intervals_are_retained_exactly(self):
        saved = REPRODUCE.read_csv(ROOT / "results/statistics/formal_frozen_probes/primary_aggregate.csv")
        generated = REPRODUCE.read_csv(self.output / "original_primary.csv")
        for original, reproduced in zip(saved, generated):
            for field in ("estimate", "ci_lower", "ci_upper", "p_value_two_sided", "holm_adjusted_p"):
                self.assertEqual(original[field], reproduced[field])
            self.assertAlmostEqual(float(original["estimate"]), float(reproduced["recomputed_estimate"]), places=12)


if __name__ == "__main__":
    unittest.main()
