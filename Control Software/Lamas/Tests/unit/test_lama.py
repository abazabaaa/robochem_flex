"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import unittest
from typing import List, Dict
import numpy as np
from lamas.utils import VoigtPeakIdentification, VoigtLinearFitResult
from lamas.lama import Lama
from lamas.logger import Logger

# Mock implementations of required classes and methods


class Guanaco_mock:
    def __init__(self, filtered_peak_fits: Dict[str, List[VoigtLinearFitResult]]):
        self.filtered_peak_fits = filtered_peak_fits


def peaks_same(peak1: VoigtPeakIdentification, peak2: VoigtPeakIdentification) -> bool:
    return 1.0 if np.abs(peak1.mu - peak2.mu) < 1.0 else 0.0


# Your Lama class here (as updated above)


# Test suite for the Lama class
class TestLama(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Lama instance
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="lama", use_console=True)

    def setUp(self):
        # Initialize Lama instance
        self.lama = Lama()
        self.lama.peaks_same = peaks_same  # Use the peaks_same method

        # Create mock linear fit results
        peak_set_A = [
            VoigtPeakIdentification(
                mu=100.0,
                amplitude=10.0,
                area=100.0,
                min_x=99.5,
                max_x=100.5,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
            VoigtPeakIdentification(
                mu=100.1,
                amplitude=12.0,
                area=110.0,
                min_x=99.6,
                max_x=100.6,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
        ]
        fit_result_A = VoigtLinearFitResult(
            peak_set=peak_set_A,
            slope_amplitude=1.0,
            intercept_amplitude=0.0,
            r_value_amplitude=0.99,
            p_value_amplitude=0.01,
            slope_area=0.1,
            intercept_area=0.0,
            r_value_area=0.99,
            p_value_area=0.01,
        )

        peak_set_B = [
            VoigtPeakIdentification(
                mu=200.0,
                amplitude=20.0,
                area=200.0,
                min_x=199.5,
                max_x=200.5,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
            VoigtPeakIdentification(
                mu=200.2,
                amplitude=22.0,
                area=210.0,
                min_x=199.7,
                max_x=200.7,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
        ]
        fit_result_B = VoigtLinearFitResult(
            peak_set=peak_set_B,
            slope_amplitude=2.0,
            intercept_amplitude=0.0,
            r_value_amplitude=0.98,
            p_value_amplitude=0.02,
            slope_area=0.2,
            intercept_area=0.0,
            r_value_area=0.98,
            p_value_area=0.02,
        )

        # Assume we have filtered_peak_fits from Guanaco
        self.guanaco = Guanaco_mock(
            filtered_peak_fits={
                "Compound_A": [fit_result_A],
                "Compound_B": [fit_result_B],
            }
        )

    def test_absorb_fits_from_class(self):
        # Test absorbing fits from Guanaco
        self.lama.absorb_fits_from_class(self.guanaco)
        # Check that linear_fit_results is set
        self.assertIsNotNone(self.lama.linear_fit_results)
        self.assertIn("Compound_A", self.lama.linear_fit_results)
        self.assertIn("Compound_B", self.lama.linear_fit_results)

    def test_absorb_fits_from_class_missing_attribute(self):
        # Test behavior when Guanaco has no filtered_peak_fits
        guanaco_without_fits = Guanaco_mock(filtered_peak_fits=None)
        with self.assertRaises(AttributeError):
            self.lama.absorb_fits_from_class(guanaco_without_fits)

    def test_predict_concentrations(self):
        # Absorb fits first
        self.lama.absorb_fits_from_class(self.guanaco)

        # Create a new spectrum with peaks
        new_spectrum = [
            VoigtPeakIdentification(
                mu=100.05,
                amplitude=11.0,
                area=105.0,
                min_x=99.55,
                max_x=100.55,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
            VoigtPeakIdentification(
                mu=200.1,
                amplitude=21.0,
                area=205.0,
                min_x=199.6,
                max_x=200.6,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
            VoigtPeakIdentification(
                mu=300.0,
                amplitude=15.0,
                area=150.0,
                min_x=299.0,
                max_x=301,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),  # Unmatched peak
        ]

        known_compounds = {"Compound_A", "Compound_B"}

        # Predict concentrations
        results = self.lama.predict_concentrations(new_spectrum, known_compounds)

        # Check results
        concentration_results = results["concentration_results"]
        unmatched_peaks = results["unmatched_peaks"]

        # Check that concentrations are predicted for known compounds
        self.assertIn("Compound_A", concentration_results)
        self.assertIn("Compound_B", concentration_results)

        # Verify predicted concentrations
        expected_conc_A = 1.0 * 11.0  # slope * amplitude + intercept
        expected_conc_B = 2.0 * 21.0  # slope * amplitude + intercept

        self.assertAlmostEqual(
            concentration_results["Compound_A"]["predicted_concentration"],
            expected_conc_A,
            places=2,
        )
        self.assertAlmostEqual(
            concentration_results["Compound_B"]["predicted_concentration"],
            expected_conc_B,
            places=2,
        )

        # Check unmatched peaks
        self.assertEqual(len(unmatched_peaks), 1)
        self.assertAlmostEqual(unmatched_peaks[0].mu, 300.0)

    def test_predict_concentrations_no_match(self):
        # Absorb fits first
        self.lama.absorb_fits_from_class(self.guanaco)

        # Create a new spectrum with peaks that do not match
        new_spectrum = [
            VoigtPeakIdentification(
                mu=150.0,
                amplitude=10.0,
                area=100.0,
                min_x=149.5,
                max_x=150.5,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
            VoigtPeakIdentification(
                mu=250.0,
                amplitude=20.0,
                area=200.0,
                min_x=249.5,
                max_x=250.5,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
        ]

        known_compounds = {"Compound_A", "Compound_B"}

        # Predict concentrations
        results = self.lama.predict_concentrations(new_spectrum, known_compounds)

        # Check results
        concentration_results = results["concentration_results"]
        unmatched_peaks = results["unmatched_peaks"]

        # Check that concentrations are zero
        self.assertEqual(
            concentration_results["Compound_A"]["predicted_concentration"], 0.0
        )
        self.assertEqual(
            concentration_results["Compound_B"]["predicted_concentration"], 0.0
        )

        # All peaks should be unmatched
        self.assertEqual(len(unmatched_peaks), 2)

    def test_predict_concentrations_missing_linear_fit_results(self):
        # Do not absorb fits
        # Create a new spectrum
        new_spectrum = [
            VoigtPeakIdentification(
                mu=100.05,
                amplitude=11.0,
                area=105.0,
                min_x=99.55,
                max_x=100.55,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
        ]

        known_compounds = {"Compound_A"}

        # Predict concentrations and expect ValueError
        with self.assertRaises(ValueError):
            self.lama.predict_concentrations(new_spectrum, known_compounds)

    def test_predict_concentrations_no_known_compounds(self):
        # Absorb fits first
        self.lama.absorb_fits_from_class(self.guanaco)

        # Create a new spectrum
        new_spectrum = [
            VoigtPeakIdentification(
                mu=100.05,
                amplitude=11.0,
                area=105.0,
                min_x=99.55,
                max_x=100.55,
                y_col="y_col",
                sigma=0.1,
                gamma=0.3,
            ),
        ]

        known_compounds = set()  # Empty set

        # Predict concentrations
        results = self.lama.predict_concentrations(new_spectrum, known_compounds)

        # Check that no concentrations are predicted
        concentration_results = results["concentration_results"]
        self.assertEqual(len(concentration_results), 0)

        # Peak should be unmatched
        unmatched_peaks = results["unmatched_peaks"]
        self.assertEqual(len(unmatched_peaks), 1)


if __name__ == "__main__":
    unittest.main()
