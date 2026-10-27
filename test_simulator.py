"""Standard-library tests for both simulator versions."""

import math
import unittest

from investment_model import (
    percentile,
    summarize_final_values,
    validate_inputs,
)
from monte_carlo_v1 import simulate as simulate_v1

try:
    from monte_carlo_v2 import simulate as simulate_v2
    from monte_carlo_v2 import summarize_numpy
except ImportError:
    simulate_v2 = None
    summarize_numpy = None


class InputValidationTests(unittest.TestCase):
    def test_valid_inputs_are_normalized(self):
        values = validate_inputs(10_000, 5, 10, 5, 1_000, 42)
        self.assertEqual(values["initial_investment"], 10_000.0)
        self.assertEqual(values["expected_return_pct"], 5.0)
        self.assertEqual(values["seed"], 42)

    def test_rejects_negative_investment(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            validate_inputs(-1, 5, 10, 5, 1_000)

    def test_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_inputs(10_000, math.inf, 10, 5, 1_000)

    def test_rejects_return_below_total_loss(self):
        with self.assertRaisesRegex(ValueError, "between -100%"):
            validate_inputs(10_000, -101, 10, 5, 1_000)

    def test_rejects_negative_volatility(self):
        with self.assertRaisesRegex(ValueError, "volatility"):
            validate_inputs(10_000, 5, -1, 5, 1_000)

    def test_rejects_invalid_horizon_and_simulation_count(self):
        with self.assertRaisesRegex(ValueError, "whole number of years"):
            validate_inputs(10_000, 5, 10, 1.5, 1_000)
        with self.assertRaisesRegex(ValueError, "between 1 and"):
            validate_inputs(10_000, 5, 10, 5, 0)


class StatisticsTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 50), 5)
        self.assertEqual(percentile([1, 2, 3, 4], 0), 1)
        self.assertEqual(percentile([1, 2, 3, 4], 100), 4)

    def test_summary_reports_loss_probability_and_average_gain(self):
        summary = summarize_final_values([80, 100, 120, 140], 100)
        self.assertEqual(summary["mean"], 110)
        self.assertEqual(summary["probability_below_initial"], 0.25)
        self.assertEqual(summary["average_gain_loss"], 10)


class StandardLibrarySimulationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = validate_inputs(10_000, 5, 10, 5, 1_000, 1234)

    def test_fixed_seed_is_reproducible(self):
        first = simulate_v1(self.inputs, seed=1234)
        second = simulate_v1(self.inputs, seed=1234)
        self.assertEqual(first["final_values"], second["final_values"])
        self.assertEqual(first["sample_paths"], second["sample_paths"])

    def test_zero_volatility_is_deterministic(self):
        inputs = validate_inputs(10_000, 5, 0, 3, 10, 7)
        result = simulate_v1(inputs, seed=7)
        expected = 10_000 * (1.05**3)
        for final_value in result["final_values"]:
            self.assertAlmostEqual(final_value, expected)

    def test_values_never_fall_below_zero(self):
        inputs = validate_inputs(10_000, -100, 500, 20, 100, 4)
        result = simulate_v1(inputs, seed=4)
        self.assertTrue(all(value >= 0 for value in result["final_values"]))
        self.assertTrue(
            all(
                value >= 0
                for path in result["sample_paths"]
                for value in path
            )
        )


@unittest.skipIf(simulate_v2 is None, "NumPy and matplotlib are not installed")
class NumpySimulationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = validate_inputs(10_000, 5, 10, 5, 1_000, 1234)

    def test_fixed_seed_is_reproducible(self):
        first = simulate_v2(self.inputs, seed=1234)
        second = simulate_v2(self.inputs, seed=1234)
        self.assertTrue((first["final_values"] == second["final_values"]).all())

    def test_zero_volatility_is_deterministic(self):
        inputs = validate_inputs(10_000, 5, 0, 3, 10, 7)
        result = simulate_v2(inputs, seed=7)
        expected = 10_000 * (1.05**3)
        self.assertTrue(
            all(abs(value - expected) < 1e-8 for value in result["final_values"])
        )

    def test_values_never_fall_below_zero(self):
        inputs = validate_inputs(10_000, -100, 500, 20, 100, 4)
        result = simulate_v2(inputs, seed=4)
        self.assertTrue((result["final_values"] >= 0).all())
        self.assertTrue((result["paths"] >= 0).all())

    def test_numpy_summary_calculations(self):
        summary = summarize_numpy([80, 100, 120, 140], 100)
        self.assertEqual(summary["mean"], 110)
        self.assertEqual(summary["probability_below_initial"], 0.25)
        self.assertEqual(summary["average_gain_loss"], 10)


if __name__ == "__main__":
    unittest.main()
