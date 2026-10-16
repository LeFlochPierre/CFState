"""Numerical regression against formulas in frozen 04k4, 07h and 08g.

Run from project root: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from cfstate.imodulons import (counts_to_logtpm, project_calibrated,
                               residualize_controls)


class FrozenMathRegression(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261016)
        self.m = rng.normal(size=(120, 9))
        self.x = rng.normal(size=(120, 5))
        self.slopes = np.linspace(0.6, 1.4, 9)
        self.b = rng.normal(size=(7, 3))

    def test_projection_matches_frozen_04k4_formula(self):
        expected = (np.linalg.pinv(self.m) @ self.x) * self.slopes[:, None]
        actual = project_calibrated(self.m, self.x, self.slopes)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            project_calibrated(self.m, self.x[:, 0], self.slopes),
            expected[:, 0], rtol=1e-13, atol=1e-13)

    def test_residual_matches_frozen_07h_08g_formula(self):
        activities = (np.linalg.pinv(self.m) @ self.x) * self.slopes[:, None]
        selected = activities[:7]
        expected = selected - self.b @ np.linalg.lstsq(self.b, selected, rcond=None)[0]
        actual = residualize_controls(selected, self.b)
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        np.testing.assert_allclose(residualize_controls(selected[:, 0], self.b),
                                   expected[:, 0], atol=1e-13)
        np.testing.assert_allclose(self.b.T @ actual, 0, atol=1e-12)

    def test_tpm_matches_original_panel_denominator(self):
        counts = np.array([[20., 1.], [3., 12.], [0., 5.]])
        lengths = np.array([1000., 1200., 1500.])
        rates = counts / (lengths[:, None] / 1000)
        expected = np.log2(1.0 + rates / rates.sum(axis=0) * 1e6)
        np.testing.assert_allclose(counts_to_logtpm(counts, lengths), expected,
                                   atol=1e-12)

    def test_fail_closed_invalid_alignment(self):
        with self.assertRaises(ValueError):
            project_calibrated(self.m[:-1], self.x, self.slopes)
        with self.assertRaises(ValueError):
            project_calibrated(self.m, self.x, self.slopes[:-1])
        with self.assertRaises(ValueError):
            residualize_controls(np.ones(8), self.b)
        with self.assertRaises(ValueError):
            counts_to_logtpm(np.zeros((3, 2)), np.ones(3))
        with self.assertRaises(ValueError):
            counts_to_logtpm(np.ones((3, 2)), np.array([1, 0, 3]))


if __name__ == "__main__":
    unittest.main()
