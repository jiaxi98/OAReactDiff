import tempfile
import unittest
from pathlib import Path

from experiments.oa_failure_audit.merge_screening import merge_rows, rebase_path_fields
from experiments.oa_failure_audit.prepare_irc_worklist import prepare_rows
from experiments.oa_failure_audit.select_dft_subset import select_rows


class TestFailureWorkflow(unittest.TestCase):
    def test_derived_manifest_rebases_artifact_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "generation"
            output_dir = root / "screen"
            source_dir.mkdir()
            output_dir.mkdir()
            artifact = source_dir / "candidate.xyz"
            artifact.write_text("0\n\n", encoding="utf-8")
            rows = rebase_path_fields(
                [{"candidate_path": "candidate.xyz"}],
                source_dir / "manifest.csv",
                output_dir / "screened.csv",
            )
        self.assertEqual(rows[0]["candidate_path"], "../generation/candidate.xyz")

    def test_merge_screening_prefixes_metrics_and_preserves_missing_rows(self):
        manifest_rows = [
            {"dataset_index": "0", "candidate_id": "0", "candidate_path": "a.xyz"},
            {"dataset_index": "0", "candidate_id": "1", "candidate_path": "b.xyz"},
        ]
        screen_rows = [
            {
                "dataset_index": "0",
                "candidate_id": "0",
                "candidate_path": "a.xyz",
                "model_label": "horm_left_efh",
                "force_rms_ev_per_angstrom": "0.1",
                "error": "",
            }
        ]
        fields, rows = merge_rows(
            ["dataset_index", "candidate_id", "candidate_path"],
            manifest_rows,
            [
                (
                    "horm_left_efh",
                    [
                        "dataset_index",
                        "candidate_id",
                        "candidate_path",
                        "model_label",
                        "force_rms_ev_per_angstrom",
                        "error",
                    ],
                    screen_rows,
                )
            ],
        )
        self.assertIn("horm_left_efh_force_rms_ev_per_angstrom", fields)
        self.assertEqual(rows[0]["horm_left_efh_force_rms_ev_per_angstrom"], "0.1")
        self.assertEqual(rows[1]["horm_left_efh_force_rms_ev_per_angstrom"], "")

    def test_dft_selection_covers_failure_strata_and_caps_reactions(self):
        def row(reaction, candidate, rmsd, ef_class, efh_class, force="0.1", ts_like="false"):
            return {
                "dataset_index": str(reaction),
                "candidate_id": str(candidate),
                "candidate_path": f"{reaction}-{candidate}.xyz",
                "paper_style_rmsd": str(rmsd),
                "horm_left_ef_significant_negative_mode_class": ef_class,
                "horm_left_ef_error": "",
                "horm_left_efh_significant_negative_mode_class": efh_class,
                "horm_left_efh_force_rms_ev_per_angstrom": force,
                "horm_left_efh_is_ts_like": ts_like,
                "horm_left_efh_error": "",
            }

        rows = [
            row(0, 0, 0.1, "0", "1"),
            row(0, 1, 0.3, "0", "0"),
            row(1, 0, 0.4, "1", "1"),
            row(1, 1, 0.6, "1", "1"),
            row(2, 0, 0.1, ">=2", ">=2"),
            row(2, 1, 0.7, "0", "0"),
        ]
        selected = select_rows(
            rows,
            budget=6,
            max_per_reaction=1,
            ef_prefix="horm_left_ef",
            efh_prefix="horm_left_efh",
            threshold=0.2,
            seed=7,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({row["dataset_index"] for row in selected}), 3)
        strata = {row["dft_stratum"] for row in selected}
        self.assertIn("ef_vs_efh_curvature_disagreement", strata)
        self.assertIn("coverage_failure_best", strata)
        self.assertIn("geometry_good_physics_bad", strata)

    def test_irc_worklist_requires_a_stationary_index_one_point(self):
        rows = [
            {
                "dataset_index": "0",
                "candidate_id": "0",
                "dft_scf_converged": "true",
                "dft_error": "",
                "dft_is_index_one": "true",
                "dft_is_ts_like": "true",
                "paper_style_rmsd": "0.4",
                "horm_left_efh_significant_negative_mode_class": "1",
            },
            {
                "dataset_index": "0",
                "candidate_id": "1",
                "dft_scf_converged": "true",
                "dft_error": "",
                "dft_is_index_one": "true",
                "dft_is_ts_like": "false",
                "paper_style_rmsd": "0.1",
                "horm_left_efh_significant_negative_mode_class": "1",
            },
        ]
        prepared = prepare_rows(rows, "dft", "horm_left_efh", 0.2, False)
        self.assertEqual(prepared[0]["followup_action"], "run_bidirectional_irc")
        self.assertEqual(prepared[0]["irc_eligible_now"], "true")
        self.assertEqual(prepared[1]["followup_action"], "ts_optimize_then_rehessian")
        self.assertEqual(prepared[1]["irc_eligible_now"], "false")


if __name__ == "__main__":
    unittest.main()
