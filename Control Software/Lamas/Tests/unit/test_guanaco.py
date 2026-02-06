"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from lamas.guanaco import Guanaco

import unittest
import pandas as pd
import os
import tempfile
import shutil
from lamas.logger import Logger


# Mock implementations of required classes and methods
class MockLogger:
    def log_message(self, message, level="info", indent=None, **kwargs):
        print(f"{level.upper()}: {message}")


class VoigtPeakIdentification:
    def __init__(self, **kwargs):
        self.mu = kwargs.get("mu")
        self.amplitude = kwargs.get("amplitude")
        self.area = kwargs.get("area")
        self.min_x = kwargs.get("min_x")
        self.max_x = kwargs.get("max_x")
        self.sigma = kwargs.get("sigma")
        self.gamma = kwargs.get("gamma")
        y_col = kwargs.get("y_col", "y_col")
        # Store other attributes as needed

    @property
    def peak_to_df_row(self):
        # Return a dictionary representing the peak data
        return {
            "mu": self.mu,
            "amplitude": self.amplitude,
            "area": self.area,
            "min_x": self.min_x,
            "max_x": self.max_x,
            "sigma": self.sigma,
            "gamma": self.gamma,
            "y_col": "y_col",
        }


class VoigtLinearFitResult:
    def __init__(
        self,
        peak_set,
        slope_amplitude,
        intercept_amplitude,
        r_value_amplitude,
        p_value_amplitude,
        slope_area,
        intercept_area,
        r_value_area,
        p_value_area,
    ):
        self.peak_set = peak_set
        self.slope_amplitude = slope_amplitude
        self.intercept_amplitude = intercept_amplitude
        self.r_value_amplitude = r_value_amplitude
        self.p_value_amplitude = p_value_amplitude
        self.slope_area = slope_area
        self.intercept_area = intercept_area
        self.r_value_area = r_value_area
        self.p_value_area = p_value_area
        self.mu = peak_set[0].mu  # Assume mu is consistent across peaks in peak_set


class Vicuna_mock:
    def __init__(self, concentrations):
        self.concentrations = concentrations


class Glama_mock:
    def __init__(self):
        pass


# Your Guanaco class here (as provided earlier)


# Test suite for the Guanaco class
class TestGuanaco(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="Guanaco", use_console=True)

    def setUp(self):
        # Set up a Guanaco instance and any necessary data
        self.guanaco = Guanaco(logger=self.logger)
        self.guanaco.filename_col = "filename"  # Assuming this is needed

        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        # self.test_dir = "temp"
        # Create temporary models
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        filenames = ["spectrum1.models", "spectrum2.models"]
        for fname in filenames:
            filepath = os.path.join(self.models_dir, fname)
            model_df = pd.DataFrame(
                {
                    "mu": [1.0, 2.0, 3.0],
                    "sigma": [0.1, 0.2, 0.3],
                    "gamma": [0.1, 0.2, 0.3],
                    "amplitude": [10, 20, 30],
                    "area": [100, 200, 300],
                    "min_x": [0.5, 1.5, 2.5],
                    "max_x": [1.5, 2.5, 3.5],
                }
            )
            model_df.to_csv(filepath, index=False)

        # Create temporary concentrations file
        self.conc_file = os.path.join(self.test_dir, "concentrations.csv")
        conc_data = pd.DataFrame(
            {
                "filename": ["spectrum1", "spectrum2"],
                "Conc_A": [1.0, 2.0],
                "Conc_B": [0.0, 3.0],
            }
        )
        conc_data.to_csv(self.conc_file, index=False)

    def tearDown(self):
        # Clean up any temporary files or directories
        shutil.rmtree(self.test_dir)
        # pass

    def test_load_models_from_dir(self):
        # Test the load_models_from_dir method
        self.guanaco.load_models_from_dir(self.models_dir)
        # Check that the models are loaded
        self.assertIn("spectrum1", self.guanaco.models)
        self.assertIn("spectrum2", self.guanaco.models)
        # Check that the models have the correct data
        model1 = self.guanaco.models["spectrum1"]
        self.assertEqual(len(model1), 3)
        self.assertEqual(model1[0].mu, 1.0)
        self.assertEqual(model1[0].amplitude, 10)
        self.assertEqual(model1[0].area, 100)

    def test_load_concentrations_from_file(self):
        # Test the load_concentrations_from_file method
        self.guanaco.load_concentrations_from_file(self.conc_file)
        # Check that conc_df is loaded
        self.assertIsNotNone(self.guanaco.conc_df)
        self.assertEqual(len(self.guanaco.conc_df), 2)
        # Check data
        self.assertEqual(self.guanaco.conc_df.loc[0, "filename"], "spectrum1")
        self.assertEqual(self.guanaco.conc_df.loc[0, "Conc_A"], 1.0)
        self.assertEqual(self.guanaco.conc_df.loc[0, "Conc_B"], 0.0)

    def test_make_subdataframes_concentration(self):
        # Ensure conc_df is loaded
        self.guanaco.conc_df = pd.DataFrame(
            {
                "filename": ["spectrum1", "spectrum2", "spectrum3"],
                "Conc_A": [1.0, 0.0, 2.0],
                "Conc_B": [0.0, 3.0, 0.0],
            }
        )
        # Call the method
        self.guanaco.make_subdataframes_concentration()
        # Check that sub_df_dict is created
        self.assertIsNotNone(self.guanaco.sub_df_dict)
        # Check that sub_df_dict has keys for compounds
        self.assertIn("A", self.guanaco.sub_df_dict)
        self.assertIn("B", self.guanaco.sub_df_dict)
        # Check that the sub-dataframes contain only rows where concentration is not zero
        sub_df_A = self.guanaco.sub_df_dict["A"]
        self.assertEqual(len(sub_df_A), 2)
        self.assertListEqual(sub_df_A["filename"].tolist(), ["spectrum1", "spectrum3"])
        sub_df_B = self.guanaco.sub_df_dict["B"]
        self.assertEqual(len(sub_df_B), 1)
        self.assertListEqual(sub_df_B["filename"].tolist(), ["spectrum2"])

    def test_peak_matching(self):
        # Set up models
        self.guanaco.models = {
            "spectrum1": [
                VoigtPeakIdentification(
                    mu=1.0,
                    amplitude=10,
                    area=100,
                    min_x=0.5,
                    max_x=1.5,
                    sigma=0.1,
                    gamma=0.1,
                ),
                VoigtPeakIdentification(
                    mu=2.0,
                    amplitude=20,
                    area=200,
                    min_x=1.5,
                    max_x=2.5,
                    sigma=0.2,
                    gamma=0.2,
                ),
            ],
            "spectrum2": [
                VoigtPeakIdentification(
                    mu=1.05,
                    amplitude=12,
                    area=110,
                    min_x=0.55,
                    max_x=1.55,
                    sigma=0.1,
                    gamma=0.1,
                ),
                VoigtPeakIdentification(
                    mu=3.0,
                    amplitude=30,
                    area=300,
                    min_x=2.5,
                    max_x=3.5,
                    sigma=0.3,
                    gamma=0.3,
                ),
            ],
        }
        # Set up conc_df and sub_df_dict
        self.guanaco.conc_df = pd.DataFrame(
            {"filename": ["spectrum1", "spectrum2"], "Conc_A": [1.0, 2.0]}
        )
        self.guanaco.sub_df_dict = {"A": self.guanaco.conc_df}

        # Define peaks_same method
        def peaks_same(peak1, peak2, **kwargs):
            # For testing, consider peaks the same if mu difference is less than 0.1
            return abs(peak1.mu - peak2.mu) < 0.1

        self.guanaco.peaks_same = peaks_same
        # Call the method
        self.guanaco.peak_matching()
        # Check that common_peaks is created
        self.assertIsNotNone(self.guanaco.common_peaks)
        self.assertIn("A", self.guanaco.common_peaks)
        common_peaks_A = self.guanaco.common_peaks["A"]
        # Should have one common peak at mu ~1.0
        self.assertEqual(len(common_peaks_A), 1)
        matched_peaks = common_peaks_A[0]
        self.assertEqual(len(matched_peaks), 2)  # One peak from each spectrum
        self.assertAlmostEqual(matched_peaks[0].mu, 1.0)
        self.assertAlmostEqual(matched_peaks[1].mu, 1.05)

    def test_fit_and_filter_peaks(self):
        # Set up common_peaks and sub_df_dict
        self.guanaco.common_peaks = {
            "A": [
                [
                    VoigtPeakIdentification(
                        mu=1.0,
                        amplitude=10,
                        area=100,
                        min_x=0.5,
                        max_x=1.5,
                        sigma=0.1,
                        gamma=0.1,
                    ),
                    VoigtPeakIdentification(
                        mu=1.05,
                        amplitude=12,
                        area=110,
                        min_x=0.55,
                        max_x=1.55,
                        sigma=0.1,
                        gamma=0.1,
                    ),
                ]
            ]
        }
        self.guanaco.sub_df_dict = {
            "A": pd.DataFrame(
                {"filename": ["spectrum1", "spectrum2"], "Conc_A": [1.0, 2.0]}
            )
        }
        # Call the method with low thresholds to ensure inclusion
        self.guanaco.fit_and_filter_peaks(r2_threshold=0.0, p_value_threshold=1.0)
        # Check that filtered_peak_fits is created
        self.assertIsNotNone(self.guanaco.filtered_peak_fits)
        self.assertIn("A", self.guanaco.filtered_peak_fits)
        filtered_fits_A = self.guanaco.filtered_peak_fits["A"]
        self.assertEqual(len(filtered_fits_A), 1)
        fit_result = filtered_fits_A[0]
        # Check fit_result attributes
        self.assertIsNotNone(fit_result.slope_amplitude)
        self.assertIsNotNone(fit_result.slope_area)
        self.assertEqual(len(fit_result.peak_set), 2)

    def test_save_data_and_load_fits(self):
        # Set up filtered_peak_fits
        self.guanaco.filtered_peak_fits = {
            "A": [
                VoigtLinearFitResult(
                    peak_set=[
                        VoigtPeakIdentification(
                            mu=1.0,
                            amplitude=10,
                            area=100,
                            min_x=0,
                            max_x=5,
                            sigma=0.1,
                            gamma=0.1,
                        ),
                        VoigtPeakIdentification(
                            mu=1.05,
                            amplitude=12,
                            area=110,
                            min_x=0,
                            max_x=5,
                            sigma=0.1,
                            gamma=0.1,
                        ),
                    ],
                    slope_amplitude=1.0,
                    intercept_amplitude=0.0,
                    r_value_amplitude=1.0,
                    p_value_amplitude=0.0,
                    slope_area=10.0,
                    intercept_area=0.0,
                    r_value_area=1.0,
                    p_value_area=0.0,
                )
            ]
        }
        # Call save_data
        base_directory = os.path.join(self.test_dir, "filtered_peak_fits")
        self.guanaco.save_data(base_directory=base_directory)
        # Check that files are created
        main_summary_file = os.path.join(
            base_directory, "filtered_peak_fits_summary.csv"
        )
        self.assertTrue(os.path.exists(main_summary_file))
        # Now create a new Guanaco instance and load the data
        new_guanaco = Guanaco()
        new_guanaco.logger = MockLogger()
        new_guanaco.load_fits(base_directory=base_directory)
        # Check that filtered_peak_fits is loaded
        self.assertIsNotNone(new_guanaco.filtered_peak_fits)
        self.assertIn("A", new_guanaco.filtered_peak_fits)
        filtered_fits_A = new_guanaco.filtered_peak_fits["A"]
        self.assertEqual(len(filtered_fits_A), 1)
        fit_result = filtered_fits_A[0]
        # Check fit_result attributes
        self.assertEqual(fit_result.slope_amplitude, 1.0)
        self.assertEqual(fit_result.slope_area, 10.0)
        self.assertEqual(len(fit_result.peak_set), 2)
        self.assertAlmostEqual(fit_result.peak_set[0].mu, 1.0)
        self.assertAlmostEqual(fit_result.peak_set[1].mu, 1.05)

    def test_load_concentrations_from_class(self):
        # Prepare existing concentrations in Guanaco
        self.guanaco.conc_df = pd.DataFrame(
            {"filename": ["spectrum1", "spectrum3"], "Compound_X": [5.0, 6.0]}
        )
        # Prepare Vicuna class with concentrations
        vicuna_conc = pd.DataFrame(
            {"filename": ["spectrum2", "spectrum3"], "Compound_Y": [7.0, 8.0]}
        )
        vicuna = Vicuna_mock(concentrations=vicuna_conc)
        # Call the method
        self.guanaco.load_concentrations_from_class(vicuna)
        # Check that concentrations are merged correctly
        expected_conc = pd.DataFrame(
            {
                "filename": ["spectrum1", "spectrum2", "spectrum3"],
                "Compound_X": [5.0, 0.0, 6.0],
                "Compound_Y": [0.0, 7.0, 8.0],
            }
        ).fillna(0)
        pd.testing.assert_frame_equal(
            self.guanaco.conc_df.sort_index(axis=1), expected_conc.sort_index(axis=1)
        )

    def test_check_models_loaded(self):
        # Prepare conc_df and models
        self.guanaco.conc_df = pd.DataFrame(
            {"filename": ["spectrum1", "spectrum2", "spectrum3"]}
        )
        self.guanaco.models = {
            "spectrum1": [],
            "spectrum2": [],
            # 'spectrum3' is missing
        }
        # Capture the assertion
        with self.assertRaises(AssertionError):
            self.guanaco._check_models_loaded()


if __name__ == "__main__":
    unittest.main()
