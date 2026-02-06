"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


import unittest
from unittest import TestCase
import numpy as np

from lamas.glama import Glama
from lamas.logger import Logger
from lamas.utils import VoigtPeakIdentification
from matplotlib import pyplot as plt
from scipy.integrate import simpson


def plot_dtw_alignment(profile1, profile2, path, spacing=0.2, x_total=None):
    """
    Plots two profiles with some spacing between them and highlights the DTW alignment path.

    :param profile1: 1D numpy array of the first profile.
    :param profile2: 1D numpy array of the second profile.
    :param path: List of tuples representing the DTW alignment path.
    :param spacing: Float, the vertical space between the profiles for clarity.
    """
    plt.figure(figsize=(12, 6))

    # Plot the first profile
    plt.plot(x_total, profile1, label="Profile 1", color="blue")

    # Plot the second profile with spacing
    plt.plot(x_total, profile2 + spacing, label="Profile 2 (spaced)", color="green")

    # Draw lines to represent the DTW alignment path
    for i, j in path[::10]:
        plt.plot(
            [x_total[i], x_total[j]],
            [profile1[i], profile2[j] + spacing],
            color="gray",
            linestyle="--",
            linewidth=0.5,
        )

    plt.title("DTW Alignment Path Between Two Profiles")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")


def plotting_function(x, y, label=None, fig=None):
    """
    Plots data onto an existing figure or creates a new figure if none is provided.

    :param fig: Optional; matplotlib Figure object. If None, creates a new figure.
    :param data: The data to plot; expects a dictionary with 'x' and 'y' keys.
    :param label: Optional; legend label for the data being plotted.
    :return: matplotlib Figure object with the plotted data.
    """

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes[0]

    ax.plot(x, y, label=label)

    if label is not None:
        ax.legend()

    return fig


class TestBaselining(TestCase):
    """testing the baselining methods with the Glama class on a bit of raman data"""

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="Glama", use_console=True)

    @classmethod
    def tearDownClass(cls):
        cls.logger.stop_logging_thread()

    def setUp(self):
        self.Glama = Glama(logger=self.logger)
        data_path = "Tests/test_data/raman_test.csv"
        self.Glama.load_data(input_data=data_path, from_filetype="ramaberry")
        self.fig = plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="raw data",
        )

    def test_cholesky_baseline(self):
        """Testing the Cholesky baseline method"""
        self.Glama._apply_function(self.Glama._cholesky_baseline)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Cholesky Baseline",
            fig=self.fig,
        )
        # Second application with modified parameters
        self.Glama._apply_function(self.Glama._cholesky_baseline, p=0.05, lam=1e6)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Cholesky Baseline (p=0.05, lam=1e6)",
            fig=self.fig,
        )

        # Third application with further modified parameters
        self.Glama._apply_function(self.Glama._cholesky_baseline, p=0.1, lam=1e4)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Cholesky Baseline (p=0.1, lam=1e4)",
            fig=self.fig,
        )

        self.fig.show()

    def test_linear_fit_baseline(self):
        """Testing the linear fit baseline method"""
        self.Glama._apply_function(
            self.Glama._linear_fit_baseline, flat_range=(3000, 3500)
        )
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Linear Fit Baseline",
            fig=self.fig,
        )
        # Second application with modified flat range
        self.Glama._apply_function(
            self.Glama._linear_fit_baseline, flat_range=(2000, 2500)
        )
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Linear Fit Baseline (2000-2500)",
            fig=self.fig,
        )

        # Third application with a different flat range
        self.Glama._apply_function(
            self.Glama._linear_fit_baseline, flat_range=(1000, 1500)
        )
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Linear Fit Baseline (1000-1500)",
            fig=self.fig,
        )

        plt.show()

    def test_als_baseline(self):
        """Testing the ALS baseline method"""
        self.Glama._apply_function(self.Glama._als_baseline)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="ALS Baseline",
            fig=self.fig,
        )
        # Second application with modified parameters
        self.Glama._apply_function(self.Glama._als_baseline, lam=1e7, p=0.02)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="ALS Baseline (lam=1e7, p=0.02)",
            fig=self.fig,
        )

        # Third application with further modified parameters
        self.Glama._apply_function(self.Glama._als_baseline, lam=1e5, p=0.05)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="ALS Baseline (lam=1e5, p=0.05)",
            fig=self.fig,
        )

        plt.show()

    def test_rolling_ball_baseline(self):
        """Testing the rolling ball baseline method"""
        self.Glama._apply_function(self.Glama._rolling_ball_baseline, window_size=50)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Rolling Ball Baseline",
            fig=self.fig,
        )
        # Second application with modified window size
        self.Glama._apply_function(self.Glama._rolling_ball_baseline, window_size=30)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Rolling Ball (window=30)",
            fig=self.fig,
        )

        # Third application with a larger window size
        self.Glama._apply_function(self.Glama._rolling_ball_baseline, window_size=70)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Rolling Ball (window=70)",
            fig=self.fig,
        )

        plt.show()

    def test_polyfit_baseline(self):
        """Testing the polynomial fit baseline method"""
        self.Glama._apply_function(self.Glama._polyfit_baseline, degree=3)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Polyfit Baseline",
            fig=self.fig,
        )
        # Second application with higher degree
        self.Glama._apply_function(self.Glama._polyfit_baseline, degree=5)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Polyfit Baseline (degree=5)",
            fig=self.fig,
        )

        # Third application with lower degree
        self.Glama._apply_function(self.Glama._polyfit_baseline, degree=2)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Polyfit Baseline (degree=2)",
            fig=self.fig,
        )

        plt.show()

    def test_wavelet_baseline(self):
        """Testing the wavelet baseline method"""
        self.Glama._apply_function(self.Glama._wavelet_baseline, wavelet="db4", level=1)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Wavelet Baseline",
            fig=self.fig,
        )
        # Second application with modified wavelet and level
        self.Glama._apply_function(self.Glama._wavelet_baseline, wavelet="db2", level=2)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Wavelet Baseline (db2, level=2)",
            fig=self.fig,
        )

        # Third application with another wavelet and higher level
        self.Glama._apply_function(
            self.Glama._wavelet_baseline, wavelet="haar", level=3
        )
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Wavelet Baseline (haar, level=3)",
            fig=self.fig,
        )

        plt.show()


class TestDenoising(TestCase):
    """testing the denoising methods with the Glama class on a bit of raman data"""

    """testing the baselining methods with the Glama class on a bit of raman data"""

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="Glama", use_console=True)

    @classmethod
    def tearDownClass(cls):
        cls.logger.stop_logging_thread()

    def setUp(self):
        self.Glama = Glama(logger=self.logger)
        data_path = "Tests/test_data/raman_test.csv"
        self.Glama.load_data(input_data=data_path, from_filetype="ramaberry")
        self.fig = plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="raw data",
        )

    def test_savgol_filter_denoise(self):
        """Testing the Savitzky-Golay filter denoise method with different window fractions"""
        # First application with default window fraction
        self.Glama._apply_function(self.Glama._savgol_filter_denoise)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Savitzky-Golay (window_frac=0.005)",
            fig=self.fig,
        )

        # Second application with a smaller window fraction
        self.Glama._apply_function(self.Glama._savgol_filter_denoise, window_frac=0.001)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Savitzky-Golay (window_frac=0.001)",
            fig=self.fig,
        )

        # Third application with a larger window fraction
        self.Glama._apply_function(self.Glama._savgol_filter_denoise, window_frac=0.01)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Savitzky-Golay (window_frac=0.01)",
            fig=self.fig,
        )

        plt.show()

    def test_median_filter_denoise(self):
        """Testing the median filter denoise method with different window sizes"""
        # First application with default window size
        self.Glama._apply_function(self.Glama._median_filter_denoise)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Median Filter (window=3)",
            fig=self.fig,
        )

        # Second application with a larger window size
        self.Glama._apply_function(self.Glama._median_filter_denoise, window_size=5)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Median Filter (window=5)",
            fig=self.fig,
        )

        # Third application with an even larger window size
        self.Glama._apply_function(self.Glama._median_filter_denoise, window_size=7)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Median Filter (window=7)",
            fig=self.fig,
        )

        plt.show()

    def test_wavelet_denoise(self):
        """Testing the wavelet denoise method with varying wavelet parameters"""
        # First application with default wavelet and level
        self.Glama._apply_function(self.Glama._wavelet_denoise)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Wavelet (db1, level=1)",
            fig=self.fig,
        )

        # Second application with different wavelet and level
        self.Glama._apply_function(
            self.Glama._wavelet_denoise, wavelet="db2", level=2, threshold=0.1
        )
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Wavelet (db2, level=2, threshold=0.1)",
            fig=self.fig,
        )

        # Third application with another wavelet and a higher level
        self.Glama._apply_function(
            self.Glama._wavelet_denoise, wavelet="haar", level=3, threshold=0.3
        )
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Wavelet (haar, level=3, threshold=0.3)",
            fig=self.fig,
        )

        plt.show()

    def test_moving_average_denoise(self):
        """Testing the moving average denoise method with different window sizes"""
        # First application with default window size
        self.Glama._apply_function(self.Glama._moving_average_denoise)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Moving Average (window=5)",
            fig=self.fig,
        )

        # Second application with a larger window size
        self.Glama._apply_function(self.Glama._moving_average_denoise, window_size=10)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Moving Average (window=10)",
            fig=self.fig,
        )

        # Third application with an even larger window size
        self.Glama._apply_function(self.Glama._moving_average_denoise, window_size=20)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="Moving Average (window=20)",
            fig=self.fig,
        )

        plt.show()

    def test_fft_denoise(self):
        """Testing the FFT denoise method with different frequency thresholds"""
        # First application with default frequency threshold
        self.Glama._apply_function(self.Glama._fft_denoise)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="FFT Denoise (threshold=0.3)",
            fig=self.fig,
        )

        # Second application with a smaller threshold
        self.Glama._apply_function(self.Glama._fft_denoise, threshold=0.05)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="FFT Denoise (threshold=0.05)",
            fig=self.fig,
        )

        # Third application with a larger threshold
        self.Glama._apply_function(self.Glama._fft_denoise, threshold=0.7)
        plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="FFT Denoise (threshold=0.7)",
            fig=self.fig,
        )

        plt.show()


class TestIntegrals(TestCase):
    """
    Testing the integral calculation methods:
    """

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="Glama", use_console=True)

    @classmethod
    def tearDownClass(cls):
        cls.logger.stop_logging_thread()

    def setUp(self):
        self.Glama = Glama(logger=self.logger)
        data_path = "Tests/test_data/raman_test.csv"
        self.Glama.load_data(input_data=data_path, from_filetype="ramaberry")
        self.fig = plotting_function(
            self.Glama.data[self.Glama.x_col],
            self.Glama.data[self.Glama.y_cols[0]],
            label="raw data",
        )

    def test_find_peaks(self):
        """Testing the peak finding method"""
        (
            peak_positions,
            peak_heigts,
            fwhm,
            left_bounds,
            right_bounds,
        ) = self.Glama._find_peaks(number_of_peaks=10, prominence=0.1)
        self.fig.axes[0].scatter(
            peak_positions, peak_heigts, 5, "r", "x", label="Peaks"
        )
        self.assertEqual(len(peak_positions), 10)
        self.assertEqual(len(peak_heigts), 10)
        self.assertEqual(len(fwhm), 10)
        self.assertEqual(len(left_bounds), 10)
        self.assertEqual(len(right_bounds), 10)

        noise_sample_y, noise_sample_x = self.Glama._find_noise_sample(
            peak_positions=peak_positions, fwhm=fwhm
        )
        print(noise_sample_y, noise_sample_x)
        self.fig.axes[0].scatter(
            noise_sample_x, noise_sample_y, 2, "g", "o", label="Noise Sample"
        )
        self.fig.axes[0].scatter(
            noise_sample_x, np.zeros(noise_sample_x.shape), 1, label="Noise Threshold"
        )
        snr = self.Glama._quantify_snr(noise_sample=noise_sample_y)
        print(snr)
        self.fig.legend()
        plt.show()


class TestSimilarity(TestCase):
    """
    Testing the methodology of peak similarities:

    """

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="Glama", use_console=True)
        cls.glam = Glama(logger=cls.logger)
        cls.x_full = np.linspace(0, 4000, 2047)

        num_profiles = 5
        original_profiles = []
        modified_profiles = {}
        for profile in range(num_profiles):
            start = np.random.randint(0, 3800)
            end = start + np.random.randint(50, 200)
            x_interval = cls.x_full[(cls.x_full >= start) & (cls.x_full <= end)]
            mu = np.random.uniform(start, end)
            sigma = np.random.uniform(1, 10)
            amplitude = np.random.uniform(0.1, 1)
            gamma = np.random.uniform(0.1, 1)

            y_original = Glama.voigt(
                x=x_interval, A=amplitude, mu=mu, sigma=sigma, gamma=gamma
            )
            area = simpson(y_original, x=x_interval)
            peak_as_class = VoigtPeakIdentification(
                y_col=f"y_{profile}",
                mu=mu,
                sigma=sigma,
                gamma=gamma,
                amplitude=amplitude,
                area=area,
                min_x=start,
                max_x=end,
            )
            original_profiles.append(peak_as_class)
            for modified_profile in range(3):
                mu_mod = mu + np.random.uniform(-10, 10)
                sigma_mod = sigma + np.random.uniform(-2, 2)
                amplitude_mod = amplitude + np.random.uniform(-0.5, 0.5)
                amplitude_mod = max(0.1, amplitude_mod)
                gamma_mod = gamma + np.random.uniform(-0.5, 0.5)
                y_modified = Glama.voigt(
                    x=x_interval,
                    A=amplitude_mod,
                    mu=mu_mod,
                    sigma=sigma_mod,
                    gamma=gamma_mod,
                )
                area_mod = simpson(y_modified, x=x_interval)

                peak_as_class = VoigtPeakIdentification(
                    y_col=f"y_{profile}_{modified_profile}",
                    mu=mu_mod,
                    sigma=sigma_mod,
                    gamma=gamma_mod,
                    amplitude=amplitude_mod,
                    area=area_mod,
                    min_x=start,
                    max_x=end,
                )
                if profile not in modified_profiles:
                    modified_profiles[profile] = []
                modified_profiles[profile].append(peak_as_class)

        cls.original_profiles = original_profiles
        cls.modified_profiles = modified_profiles

    @classmethod
    def tearDownClass(cls):
        cls.logger.stop_logging_thread()

    def _test_similarity_mode(self, mode, threshold=0.7):
        """
        Helper function to test a specific similarity mode.
        """
        related_similarity = []
        unrelated_similarity = []
        for i, original_peak in enumerate(self.original_profiles):
            for j, modified_peak in enumerate(self.modified_profiles[i]):
                # Check high similarity with modified peaks in the same bracket
                print(f"testing similarity betewen \n{original_peak} \n{modified_peak}")
                similarity = self.glam.peak_same(
                    original_peak, modified_peak, mode=mode
                )
                print("result:", similarity)
                related_similarity.append(similarity)
            # Check low similarity with unrelated peaks
        for k, unrelated_peak in enumerate(self.original_profiles):
            if unrelated_peak != original_peak:
                print(f"testing similarity betewen \n{original_peak} \n{modified_peak}")
                similarity = self.glam.peak_same(
                    original_peak, unrelated_peak, mode=mode
                )
                print("result:", similarity)
                unrelated_similarity.append(similarity)

        print(f"results for {mode}:")
        print(f"related_similarity: {related_similarity}")
        print(f"unrelated_similarity: {unrelated_similarity}")

        # Check that the related peaks have a higher similarity than the unrelated peaks
        self.assertTrue(
            all([similarity > threshold for similarity in related_similarity])
        )
        self.assertTrue(
            all([similarity < threshold for similarity in unrelated_similarity])
        )

    def test_relative_threshold(self):
        self._test_similarity_mode("relative_threshold")

    def test_weighted_similarity(self):
        self._test_similarity_mode(
            "weighted_similarity", threshold=0.5
        )  # Adjust threshold if necessary

    def test_mu_only(self):
        self._test_similarity_mode("mu_only")

    def test_correlation(self):
        self._test_similarity_mode("correlation", threshold=0.05)

    def test_fwhm(self):
        self._test_similarity_mode("fwhm")

    def test_overlap_area(self):
        self._test_similarity_mode("overlap_area", threshold=0.5)

    def test_dtw(self):
        self._test_similarity_mode("dtw", threshold=0.5)

    def test_cross_correlation(self):
        self._test_similarity_mode("cross_correlation", threshold=0.5)


if __name__ == "__main__":
    unittest.main()
