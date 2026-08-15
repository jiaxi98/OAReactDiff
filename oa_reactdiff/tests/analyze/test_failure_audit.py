import tempfile
import unittest
from pathlib import Path

import numpy as np

from oa_reactdiff.analyze.failure_audit import (
    BOHR_M,
    ELECTRON_VOLT_J,
    HARTREE_J,
    ANGSTROM_M,
    canonical_cartesian_hessian,
    element_permuted_kabsch_rmsd,
    eigenvalues_to_wavenumbers,
    ev_angstrom_eigenvalues_to_wavenumbers,
    ordered_kabsch_rmsd,
    projected_mass_weighted_hvp,
    read_xyz,
    summarize_candidate_rows,
    vibrational_basis,
    vibrational_eigenvalues,
)


class TestFailureAudit(unittest.TestCase):
    def test_ordered_kabsch_rmsd_is_rigid_transform_invariant(self):
        reference = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.2, 0.8, 0.3],
            ]
        )
        rotation = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        candidate = reference @ rotation + np.asarray([4.0, -2.0, 7.0])
        self.assertLess(ordered_kabsch_rmsd(reference, candidate), 1e-12)

    def test_pyscf_hessian_layout(self):
        matrix = np.arange(81, dtype=float).reshape(9, 9)
        matrix = 0.5 * (matrix + matrix.T)
        pyscf_layout = matrix.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
        np.testing.assert_allclose(canonical_cartesian_hessian(pyscf_layout, 3), matrix)

    def test_element_permuted_rmsd_matches_equivalent_atoms(self):
        symbols = ["C", "H", "H", "O"]
        reference = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.7, 0.0],
                [-0.6, 0.9, 0.2],
                [0.1, -1.2, 0.3],
            ]
        )
        candidate = reference[[0, 2, 1, 3]]
        candidate = candidate @ np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        candidate += np.asarray([3.0, 4.0, -2.0])
        self.assertLess(element_permuted_kabsch_rmsd(symbols, reference, candidate), 1e-12)

    def test_vibrational_projection_removes_rigid_modes(self):
        symbols = ["O", "H", "H"]
        coordinates = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.95, 0.0, 0.0],
                [-0.24, 0.92, 0.0],
            ]
        )
        masses = np.repeat(np.sqrt([15.999, 1.00784, 1.00784]), 3)

        # Recover the same internal basis used by the implementation by projecting
        # an identity Hessian, then construct a Hessian with one unstable mode.
        identity_spectrum = vibrational_eigenvalues(np.eye(9), symbols, coordinates)
        self.assertEqual(identity_spectrum.shape, (3,))

        center_of_mass = np.average(coordinates, axis=0, weights=np.square(masses[::3]))
        centered = coordinates - center_of_mass
        rigid = []
        for axis in np.eye(3):
            rigid.append((masses.reshape(3, 3) * axis[None, :]).reshape(-1))
        for axis in np.eye(3):
            rigid.append((masses.reshape(3, 3) * np.cross(axis[None, :], centered)).reshape(-1))
        left, singular_values, _ = np.linalg.svd(np.column_stack(rigid), full_matrices=True)
        rank = int(np.sum(singular_values > 1e-10))
        internal_basis = left[:, rank:]
        target = np.diag([-0.02, 0.03, 0.05])
        mass_weighted = internal_basis @ target @ internal_basis.T
        cartesian = masses[:, None] * mass_weighted * masses[None, :]
        np.testing.assert_allclose(
            vibrational_eigenvalues(cartesian, symbols, coordinates),
            np.diag(target),
            atol=1e-10,
        )

        implementation_basis, implementation_masses = vibrational_basis(symbols, coordinates)
        test_vector = np.asarray([0.2, -0.5, 0.7])
        actual_hvp = projected_mass_weighted_hvp(
            test_vector,
            implementation_basis,
            implementation_masses,
            lambda vector: cartesian @ vector,
        )
        np.testing.assert_allclose(actual_hvp, target @ test_vector, atol=1e-10)

    def test_frequency_conversions_agree_across_equivalent_units(self):
        hartree_eigenvalues = np.asarray([-0.02, 0.03])
        conversion = HARTREE_J / BOHR_M**2 * ANGSTROM_M**2 / ELECTRON_VOLT_J
        ev_angstrom_eigenvalues = hartree_eigenvalues * conversion
        np.testing.assert_allclose(
            eigenvalues_to_wavenumbers(hartree_eigenvalues),
            ev_angstrom_eigenvalues_to_wavenumbers(ev_angstrom_eigenvalues),
        )

    def test_read_xyz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "water.xyz"
            path.write_text("3\nwater\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
            symbols, coordinates = read_xyz(path)
        self.assertEqual(symbols, ["O", "H", "H"])
        self.assertEqual(coordinates.shape, (3, 3))

    def test_failure_summary_preserves_missing_evidence(self):
        rows = [
            {
                "dataset_index": 0,
                "candidate_id": 0,
                "paper_style_rmsd": 0.10,
                "confidence": 0.1,
            },
            {
                "dataset_index": 0,
                "candidate_id": 1,
                "paper_style_rmsd": 0.40,
                "confidence": 0.9,
            },
            {
                "dataset_index": 1,
                "candidate_id": 0,
                "paper_style_rmsd": 0.10,
                "force_rms_ev_per_angstrom": 0.2,
                "strict_imaginary_modes": 1,
                "irc_connects_target": "false",
            },
        ]
        summaries = summarize_candidate_rows(rows)
        self.assertEqual(summaries[0]["coverage_failure"], "false")
        self.assertEqual(summaries[0]["ranking_failure"], "true")
        self.assertEqual(summaries[0]["geometry_good_physics_bad"], "NA")
        self.assertEqual(summaries[0]["alternate_saddle"], "NA")
        self.assertEqual(summaries[1]["coverage_failure"], "false")
        self.assertEqual(summaries[1]["ranking_failure"], "NA")
        self.assertEqual(summaries[1]["geometry_good_physics_bad"], "true")
        self.assertEqual(summaries[1]["alternate_saddle"], "true")
        self.assertEqual(summaries[1]["alternate_saddle_evidence"], "irc")


if __name__ == "__main__":
    unittest.main()
