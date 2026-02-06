"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: repo of base classes for each of the backends

"""

from __future__ import annotations

import os.path
import time

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, peak_widths, find_peaks, correlate
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky, solve
from typing import Union, Tuple, List, Callable, Literal, Optional
import pywt
import nmrglue as ng
from lamas.utils import (
    initkwargs,
    extract_columns,
    VoigtPeakIdentification,
)
from scipy.ndimage import median_filter
from scipy.integrate import simpson
from scipy.stats import norm
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from lamas.logger import Logger


class Glama:
    """
    Base class for ALPACA, GUANACO, lamas, and VICUNA. Handles I/O operations, logging, and shared utility functions
    such as Gaussian, Lorentzian, Voigt, and Savitzky-Golay filtering.
    """

    nmr_solvent_dict = {
        "DMSO": {"integral": 6, "solvent_peak": 2.5, "solvent_molarity": 14.0 / 200},
        "CHCl3": {"integral": 1, "solvent_peak": 7.26, "solvent_molarity": 12.5 / 200},
        "CH3CN": {"integral": 3, "solvent_peak": 1.94, "solvent_molarity": 19.2 / 200},
        "MeOH": {"integral": 4, "solvent_peak": 3.31, "solvent_molarity": 24.7 / 200},
        "acetone": {
            "integral": 6,
            "solvent_peak": 2.11,
            "solvent_molarity": 13.6 / 200,
        },
    }
    _raw_data: pd.DataFrame | None  # data before processing as soon as it is read in
    data: pd.DataFrame | None  # data after processing
    x_col: str  # the column name for the x-axis values
    y_cols: List[str]  # the column names for the y-axis values
    raman_required_kwargs: List[initkwargs]  # required keyword arguments for Raman data
    nmr_required_kwargs: List[initkwargs]  # required keyword arguments for NMR data

    def __init__(self, logger: Logger | None = None, use_console: bool = True) -> None:
        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger
            self.logger.start_logging_thread(
                platform=self.__class__.__name__, use_console=use_console
            )

        self.log("Initialising Class")

        self._name = "glama"
        self._description = (
            "Base class for ALPACA, GUANACO, lamas and VICUNA. Handles IO."
        )
        self.data = None
        self.x_col = None
        self.y_col = None

    def log(self, message: str, **kwargs) -> None:
        """
        Log a message to the logger.

        :param message: str: message to log
        """
        self.logger.log_message(message, origin=self.__class__.__name__, **kwargs)

    def load_data(
        self,
        input_data: Union[str, pd.DataFrame],
        x_col: str = None,
        y_cols: List[str] = None,
        from_filetype: Union[str, None] = None,
    ) -> None:
        """
        Load data from a CSV file or pandas DataFrame.

        :param input_data: str (path to CSV) or pandas DataFrame
        :param x_col: str, column name for x-axis values
        :param y_cols: list of str, column names for y-axis values
        :param from_filetype: str, file type of the data allowed values are:

        Ramaberry (for our raman spectrometer),
        lama_nmr(from nmr data previously processed by lamas),
        lama_raman( from raman data previously processed by lamas),
        Bruker (NMR data from bruker),
        Spinsolve (NMR data from magritek spinsolve)
        jdx (JCAMP-DX file)


        """
        if isinstance(input_data, pd.DataFrame):
            # clean up input data:
            input_data = input_data.loc[:, input_data.columns.notnull()]
            if "Rshift" in input_data.columns:
                self.x_col = "Rshift"
            elif "PPM" in input_data.columns:
                self.x_col = "PPM"
            else:
                self.x_col = input_data.columns[0]

            self.data = input_data
            self.file_type = from_filetype
        elif isinstance(input_data, str):
            self.data_parent_dir = os.path.dirname(input_data)
            self.data_name_file = os.path.basename(input_data)
            self.file_type = from_filetype
            match from_filetype:
                case "ramaberry":
                    self._load_ramaberry(input_data)
                case "lama_nmr":
                    self._load_lama_nmr(input_data)
                case "lama_raman":
                    self._load_lama_raman(input_data)
                case "bruker":
                    self._validate_nmr_type()
                    self._load_bruker(input_data)
                case "spinsolve":
                    self._validate_nmr_type()
                    self._load_spinsolve(input_data)
                case "jdx":
                    self._validate_nmr_type()
                    self._load_jcampdx(input_data)
                case _:
                    self.log(
                        f"Input data must be a file path or pandas DataFrame. \n from_filetype: {from_filetype} is not a valid option",
                        level="error",
                    )

                    raise ValueError(
                        f"Input data must be a file path or pandas DataFrame. "
                        f"{from_filetype} is not a valid option"
                    )
        else:
            self.log(
                "Input data must be a file path or pandas DataFrame.", level="error"
            )
            self.log(
                f"data type: {type(input_data)} not supported.",
                level="error",
                indent="enter",
            )
            self.log("", indent="exit")
            raise TypeError(
                f"Input data must be a file path or pandas DataFrame. data type: {type(input_data)} not supported."
            )

        self.x_col = x_col or getattr(
            self, "x_col", "x"
        )  # gives priority to the x_col argument
        self.y_cols = y_cols or self.data.columns.drop(self.x_col).tolist()

        # always make sure that the df is sorted by the x column small to big
        self.data = self.data.sort_values(by=self.x_col, ascending=True)
        self.log("Successfully loaded data")

    def set_ycol_to(self, y_col: Union[str, List[str]]) -> None:
        """
        set the y columns to a specific name value. if y_col is a string it will set all y columns to that name plus
        a number. if y_col is a list of strings it will set the y columns to those names.(only if matching length)

        :param y_col: str or list of str: name of the y column(s)

        """
        if y_col is None:
            self.log("No y_col provided", level="error")
            raise ValueError("No y_col provided")
        if isinstance(y_col, str):
            if len(self.y_cols) == 1:
                self.y_cols = [y_col]
            else:
                self.y_cols = [f"{y_col}_{i}" for i in range(len(self.y_cols))]

        elif isinstance(y_col, list):
            if len(y_col) != len(self.y_cols):
                self.log(
                    "y_col list length does not match the number of y_cols",
                    level="error",
                )
                raise ValueError(
                    "y_col list length does not match the number of y_cols"
                )
            self.y_cols = y_col

        self.data.columns = [self.x_col] + self.y_cols

    def set_spectroscopy(
        self, spectroscopy: str = "Raman", required_args: List[initkwargs] = None
    ) -> None:
        """
        Initialize the Alpaca class with the given spectroscopy type and required arguments.

        :param spectroscopy: str: type of spectroscopy data to preprocess, default is "Raman"
        :param required_args: list of InitKwargs: required keyword arguments specific to the spectroscopy type
        """
        self.spectroscopy = spectroscopy
        self.required_args = required_args or []

        if spectroscopy in ["Raman", "NMR"]:
            self._set_spectroscopy_main(spectroscopy, self.required_args)
            self.log(f"Successfully set {spectroscopy} spectroscopy type")
        else:
            self.log(f"{spectroscopy} spectroscopy type not supported", level="error")
            raise ValueError("Spectroscopy type not supported")

    def _set_spectroscopy_main(
        self, spectroscopy: str, required_args: List[initkwargs]
    ) -> None:
        """
        Set the spectroscopy type and initialize the required parameters.

        :param spectroscopy: str: type of spectroscopy ("Raman" or "NMR")
        :param required_args: list of InitKwargs: required keyword arguments specific to the spectroscopy type
        """
        self.spectroscopy = spectroscopy

        if spectroscopy == "Raman":
            default_args = {
                kwarg.name: kwarg.default for kwarg in self.raman_required_kwargs
            }
            for arg in required_args:
                default_args[arg.name] = arg.default
            self.ex_wlen = default_args["ex_wlen"]
            self.norm_range = default_args["norm_range"]
        elif spectroscopy == "NMR":
            default_args = {
                kwarg.name: kwarg.default for kwarg in self.nmr_required_kwargs
            }
            for arg in required_args:
                default_args[arg.name] = arg.default
            self.solvent = default_args["solvent"]
        else:
            raise ValueError("Unsupported spectroscopy type")

    def absorb_class(self, class_data: Glama) -> None:
        """
        Absorb the data from another glama-type class. (i.e. going from Alpaca to Lama)
        """
        self.absorb_mapping = {
            "Alpaca": ["data", "x_col", "y_cols", "timestamp"],
            "Vicuna": [
                "data",
                "x_col",
                "y_cols",
                "timestamp",
                "model_spectrum",
                "voigt_params",
            ],
            "Guanaco": [
                "data",
                "x_col",
                "y_cols",
                "timestamp",
                "model_spectrum",
                "voigt_params",
            ],
            "Lamas": [
                "data",
                "x_col",
                "y_cols",
                "timestamp",
                "model_spectrum",
                "voigt_params",
            ],
        }
        # now check the self type and absorb specific data
        # Absorb specific attributes based on the class type
        source_class_name = type(class_data).__name__
        if source_class_name in self.absorb_mapping:
            for attr in self.absorb_mapping[source_class_name]:
                if hasattr(class_data, attr):
                    setattr(self, attr, getattr(class_data, attr))
                else:
                    self.log(
                        f"Warning: '{source_class_name}' does not have attribute '{attr}'"
                    )

    def _load_ramaberry(self, filepath: str) -> None:
        """Load Raman data from a csv file:
        data from the ramaberry spectrometer comes as a two column csv file
        with the first column ('R_shift') being the x values and the second column ('Intensity') being the y values
        """

        self.data = pd.read_csv(filepath, index_col=0)
        # save the timestamp of the file which is saved as the title of the second column:
        self.timestamp = self.data.columns[1]

        self.data.columns = ["R_shift", "Intensity"]

        self.x_col = "R_shift"
        self.y_cols = ["Intensity"]

    def _load_lama_nmr(self, filepath: str) -> None:
        """Load NMR data from a csv file:
        data from the lamas spectrometer comes as a two column csv file
        with the first column ('PPM') being the x values and the second column ('Intensity') being the y values
        """
        self.data = pd.read_csv(filepath)
        self.timestamp = self.data.columns[1]
        self.x_col = "PPM"
        self.y_cols = ["Intensity"]

    def _load_lama_raman(self, filepath: str) -> None:
        """Load Raman data from a csv file:
        data from the lamas spectrometer comes as a two column csv file
        with the first column ('R_shift') being the x values and the second column ('Intensity') being the y values
        """
        self.data = pd.read_csv(filepath)
        self.timestamp = self.data.columns[1]
        self.data.columns = ["R_shift", "Intensity"]
        self.x_col = "R_shift"
        self.y_cols = ["Intensity"]

    def _load_jcampdx(self, filepath: str) -> None:
        """Load JCAMP-DX file and convert it to a pandas DataFrame."""
        self.dict_data, self.dat = ng.jcampdx.read(filepath)
        self.udic = ng.jcampdx.guess_udic(self.dict_data, self.dat)
        self.unit_conversion = ng.fileiobase.uc_from_udic(self.udic)
        self.timestamp = time.strftime(
            "%Y%m%d_%H%M%S_%f", self.dict_data["acqus"]["time"]
        )

        x = self.unit_conversion.ppm_scale()
        y = self.dat

        self.x_col = "PPM"
        self.y_cols = ["Intensity"]
        self.data = pd.DataFrame({"PPM": x, "Intensity": y})
        self.data.sort_values(by="PPM", ascending=True, inplace=True)

    def _validate_nmr_type(self) -> None:
        """Validate if the current spectrum type is NMR."""
        if self.spec_type != "NMR":
            raise ValueError("Only NMR spectra are supported for this file type.")

    def _load_bruker(self):
        self.dic, self._raw_data = ng.bruker.read(self.file_path)
        self.udic = ng.bruker.guess_udic(self.dic, self._raw_data)
        self.unit_conversion = ng.fileiobase.uc_from_udic(self.udic)
        x = self.unit_conversion.ppm_scale()
        y = ng.bruker.remove_digital_filter(self.dic, self._raw_data)
        y = ng.proc_base.zf_size(y, int(self.udic[0]["size"]))
        y = ng.proc_base.fft(y)

        self.timestamp = time.strftime("%Y%m%d_%H%M%S_%f", self.dic["acqus"]["time"])
        self.x_col = "PPM"
        self.y_cols = ["Intensity"]

        self.data = pd.DataFrame({"PPM": x, "Intensity": y})
        self.data.sort_values(by="PPM", ascending=True, inplace=True)

    def _load_spinsolve(self):
        """Load NMR data from a Spinsolve file."""

        self.dic, self._raw_data = ng.spinsolve.read(self.file_path)
        self.udic = ng.spinsolve.guess_udic(self.dic, self._raw_data)
        self.unit_conversion = ng.fileiobase.uc_from_udic(self.udic)

        self.timestamp = time.strftime("%Y%m%d_%H%M%S_%f", self.dic["acqu"]["time"])

        x = self.unit_conversion.ppm_scale()
        y = ng.proc_base.fft(self._raw_data)

        self.x_col = "PPM"
        self.y_cols = ["Intensity"]

        self.data = pd.DataFrame({"PPM": x, "Intensity": y})
        self.data.sort_values(by="PPM", ascending=True, inplace=True)

    def _apply_function(self, function: Callable, **kwargs):
        """
        Applies a processing function to the data:
        :param function: function to apply to the data
        :param kwargs: additional arguments for the function

        DISCLAIMER: only use for functions that return an array to be applied to the data!
        other functions don't work (like find_peaks)


        """
        if self.data is None:
            raise ValueError("No data to process.")

        if len(self.y_cols) > 1:
            for y in self.y_cols:
                self.data[y] = function(y=self.data[y], **kwargs)
        elif len(self.y_cols) == 1:
            self.data[self.y_cols[0]] = function(**kwargs)
        else:
            raise ValueError("No data to process.")

    def save_data(self, output_path: str | None = None) -> None:
        """
        Save the processed data to a CSV file.

        :param output_path: str, path to the output CSV file. If output_path is None,
                            a filename will be generated based on timestamp and spec_type,
                            and it will save to self.data_parent_dict if available.

        :raises ValueError: If neither output_path nor self.data_parent_dict is set,
                            preventing random saving in the code directory.
        """

        # Check for output_path or generate one based on timestamp and spec_type
        if output_path is None:
            if self.data_parent_dir is not None:
                # Use the data_parent_dict as the directory for the output file
                output_path = os.path.join(
                    self.data_parent_dir, f"{self.timestamp}_{self.spec_type}.csv"
                )
            else:
                # Raise an error if neither output_path nor data_parent_dict is provided
                self.log(
                    "No output path provided, and no data_parent_dict set. "
                    "\n Specify an output_path or set data_parent_dict to save the file. ",
                    level="error",
                )
                raise ValueError(
                    "No output path provided, and no data_parent_dict set. "
                    "Specify an output_path or set data_parent_dict to save the file."
                )

        # Save the data if it exists
        if self.data is not None:
            self.data.columns = [self.x_col, self.timestamp]
            self.data.to_csv(output_path, index=False)
        else:
            raise ValueError("No data to save.")

    @staticmethod
    def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        Function for a Gaussian peak.

        :param x: array-like, x-axis values
        :param mu: float, mean of the Gaussian
        :param sigma: float, standard deviation of the Gaussian
        :return: array-like, Gaussian function values
        """
        return np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    @staticmethod
    def lorentzian(x: np.ndarray, mu: float, gamma: float) -> np.ndarray:
        """
        Function for a Lorentzian peak.

        :param x: array-like, x-axis values
        :param mu: float, location parameter (mean) of the Lorentzian
        :param gamma: float, scale parameter (half-width at half-maximum) of the Lorentzian
        :return: array-like, Lorentzian function values
        """
        return gamma / (np.pi * ((x - mu) ** 2 + gamma**2))

    @staticmethod
    def voigt(
        x: np.ndarray, A: float, mu: float, sigma: float, gamma: float
    ) -> np.ndarray:
        """
        Pseudo-Voigt function, which is a combination of Gaussian and Lorentzian peaks.

        :param x: array-like, x-axis values
        :param A: float, amplitude of the peak
        :param mu: float, mean of the peak
        :param sigma: float, standard deviation of the Gaussian component
        :param gamma: float, scale parameter of the Lorentzian component
        :return: array-like, Pseudo-Voigt function values
        """
        return A * (Glama.gaussian(x, mu, sigma) * Glama.lorentzian(x, mu, gamma))

    @extract_columns
    def find_fwhm(self, x: np.ndarray, y: np.ndarray, y_max: float) -> float:
        """
        Find the full width at half maximum (FWHM) of the peak defined by x and y.

        :param x: array-like, x-axis values
        :param y: array-like, y-axis values
        :param y_max: float, maximum y value of the peak
        :return: float, FWHM value
        """
        half_max = y_max / 2
        mask = np.where(y >= half_max)[0]
        if len(mask) < 2:
            return 0.01
        return x[mask[-1]] - x[mask[0]]

    @extract_columns
    def _find_noise_sample(
        self, y: np.ndarray, peak_positions: np.ndarray, fwhm: np.ndarray
    ) -> np.ndarray:
        """
        Find the noise sample given the peak positions and the FWHM of the peaks.

        :param cropped_data: 1D array of cropped data
        :param peak_positions: 1D array of peak positions
        :param fwhm: 1D array of FWHM of the peaks
        :return: noise sample
        """
        peak_noise_space = np.arange(
            max(peak_positions[0] - 3 * fwhm[0], 0),
            min(peak_positions[0] + 3 * fwhm[0], len(y)),
        )

        if len(peak_positions) > 1:
            for i in range(1, len(peak_positions)):
                peak_noise_space = np.concatenate(
                    (
                        peak_noise_space,
                        np.arange(
                            max(peak_positions[i] - 3 * fwhm[i], 0),
                            min(peak_positions[i] + 3 * fwhm[i], len(y)),
                        ),
                    )
                )

        noise_sample = np.delete(y, peak_noise_space)

        return noise_sample, peak_noise_space

    @extract_columns
    def _cholesky_baseline(
        self,
        y: np.ndarray,
        p: float = 0.01,
        lam: float = 1e5,
        niter: int = 10,
        return_baseline: bool = False,
    ) -> np.ndarray:
        """
        Algorithmically generated baseline using asymmetric weights and smoothness parameter.
        :param y: 1D array of y-axis values
        :param p: float, weight for the peaks
        :param lam: float, smoothness parameter
        :param niter: int, number of iterations
        :param return_baseline: bool, whether to return the baseline instead of the baseline corrected signal
        """

        n = len(y)
        w = np.ones(n)
        D = diags([1, -2, 1], [0, -1, -2], shape=(n, n - 2))
        D = lam * D.dot(D.transpose())

        for i in range(niter):
            W = diags([w.flatten()], [0])
            Z = W + D
            C = cholesky(Z.toarray())
            z = solve(C, solve(C.T, W.dot(y)))
            w = p * (y > z) + (1 - p) * (y <= z)

        if return_baseline:
            return z
        return y - z

    @extract_columns
    def _linear_fit_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        flat_range: Tuple[float, float],
        return_baseline: bool = False,
    ):
        """
        Fits a linear baseline in teh flat_range of the data, extrapolates the baseline to the entire data range
        and subtracts to the data

        :param x: 1D array of x-axis values
        :param y: 1D array of y-axis values
        :param flat_range: tuple of floats, lower and upper bounds for the flat range (in units of x)
        :param return_baseline: bool, whether to return the baseline instead of the baseline corrected signal
        """
        # Select points within the specified flat range
        mask = (x >= flat_range[0]) & (x <= flat_range[1])
        x_flat = x[mask]
        y_flat = y[mask]

        # Fit a linear model to the selected points
        coeffs = np.polyfit(x_flat, y_flat, 1)
        baseline = np.polyval(coeffs, x)

        if return_baseline:
            return baseline
        # Subtract the baseline
        return y - baseline

    @extract_columns
    def _als_baseline(
        self,
        y: np.ndarray,
        lam: float = 1e6,
        p: float = 0.01,
        n_iter: int = 100,
        reg: float = 1e-9,
        return_baseline: bool = False,
    ) -> np.ndarray:
        """
        Applies ALS baseline correction to y.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param lam: float, smoothness parameter (default 1e6)
        :param p: float, asymmetry parameter (default 0.01)
        :param n_iter: int, number of iterations (default 10)
        :param reg: float, regularization parameter (default 1e-9)
        :param return_baseline: bool, whether to return the baseline instead of the baseline corrected signal(default False)
        :return: np.ndarray, baseline-corrected y-axis values

        """
        y = y.flatten()
        L = len(y)

        # Create the second-order difference matrix D as a sparse matrix
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L))

        # Initialize weights w as a 1D array of ones with length L
        w = np.ones(L)

        for _ in range(n_iter):
            # Create the diagonal weight matrix W
            W = diags(w, 0)

            # Solve for Z using spsolve to avoid inversion issues, adding regularization
            Z = spsolve(W + lam * D.T @ D + reg * eye(L), W @ y)

            # Update weights based on the new baseline estimate Z
            w = np.where(y > Z, p, 1 - p)

        # Return the baseline-corrected signal
        if return_baseline:
            return Z
        return y - Z

    @extract_columns
    def _rolling_ball_baseline(
        self, y: np.ndarray, window_size: int = 50, return_baseline: bool = False
    ) -> np.ndarray:
        """
        Applies rolling ball baseline correction.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param window_size: int, window size for the rolling ball (default 50)
        :param kwargs: additional arguments
        :param return_baseline: bool, whether to return the baseline instead of the baseline corrected signal
        :return: np.ndarray, baseline-corrected y-axis values
        """
        baseline = np.minimum.accumulate(y)
        for i in range(1, len(y)):
            baseline[i] = min(
                baseline[i], np.min(y[max(0, i - window_size) : i + window_size])
            )
        if return_baseline:
            return baseline
        return y - baseline

    @extract_columns
    def _polyfit_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3,
        return_baseline: bool = False,
    ) -> np.ndarray:
        """
        Applies polynomial baseline correction.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param degree: int, polynomial degree (default 3)
        :param kwargs: additional arguments
        :param return_baseline: bool, whether to return the baseline instead of the baseline corrected signal
        :return: np.ndarray, baseline-corrected y-axis values
        """
        coeffs = np.polyfit(x, y, degree)
        baseline = np.polyval(coeffs, x)
        if return_baseline:
            return baseline
        return y - baseline

    @extract_columns
    def _wavelet_baseline(
        self,
        y: np.ndarray,
        wavelet: str = "db4",
        level: int = 1,
        return_baseline: bool = False,
    ) -> np.ndarray:
        """
        Applies wavelet transform baseline correction.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param wavelet: str, wavelet type (default 'db4')
        :param level: int, decomposition level (default 1)
        :param kwargs: additional arguments
        :param return_baseline: bool, whether to return the baseline instead of the baseline corrected signal
        :return: np.ndarray, baseline-corrected y-axis values
        """
        coeffs = pywt.wavedec(y.flatten(), wavelet, level=level)
        coeffs[0] = np.zeros_like(
            coeffs[0]
        )  # Remove the approximation (baseline) component
        baseline = pywt.waverec(coeffs, wavelet)
        if return_baseline:
            return baseline[: len(y)]
        return y - baseline[: len(y)]

    @extract_columns
    def _savgol_filter_denoise(self, y: np.ndarray, window_frac: float = 0.005) -> None:
        """
        Perform Savitzky-Golay filtering on the data with a window size of window_frac * len(data).

        :param window_frac: float, fraction of the data length to be used as the window size
        """

        window_size = (
            int(len(y) * window_frac) // 2 * 2 + 1
        )  # Ensure window size is an odd integer

        y_denoised = savgol_filter(y.flatten(), window_length=window_size, polyorder=2)
        return y_denoised.reshape(-1, 1)

    @extract_columns
    def _median_filter_denoise(
        self, x: np.ndarray, y: np.ndarray, window_size: int = 3, **kwargs
    ) -> np.ndarray:
        """
        Applies a median filter for noise reduction.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param window_size: int, size of the filter window (default 3)
        :param kwargs: additional arguments
        :return: np.ndarray, noise-reduced y-axis values
        """
        y_denoised = median_filter(y, size=window_size)
        return y_denoised

    @extract_columns
    def _wavelet_denoise(
        self,
        x: np.ndarray,
        y: np.ndarray,
        wavelet: str = "db1",
        level: int = 1,
        threshold: float = 0.2,
        **kwargs,
    ) -> np.ndarray:
        """
        Applies wavelet transform denoising by thresholding small wavelet coefficients.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param wavelet: str, type of wavelet (default 'db1')
        :param level: int, decomposition level (default 1)
        :param threshold: float, threshold for coefficient filtering (default 0.2)
        :param kwargs: additional arguments
        :return: np.ndarray, noise-reduced y-axis values
        """
        coeffs = pywt.wavedec(y.flatten(), wavelet, level=level)
        # Apply soft thresholding on detail coefficients
        coeffs[1:] = [pywt.threshold(c, threshold * np.max(c)) for c in coeffs[1:]]
        y_denoised = pywt.waverec(coeffs, wavelet)

        return y_denoised[: len(y)].reshape(-1, 1)

    @extract_columns
    def _moving_average_denoise(
        self, x: np.ndarray, y: np.ndarray, window_size: int = 5, **kwargs
    ) -> np.ndarray:
        """
        Applies a moving average filter to reduce noise.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param window_size: int, size of the moving window (default 5)
        :param kwargs: additional arguments
        :return: np.ndarray, noise-reduced y-axis values
        """
        y_denoised = np.convolve(
            y.flatten(), np.ones(window_size) / window_size, mode="same"
        )
        return y_denoised.reshape(-1, 1)

    @extract_columns
    def _fft_denoise(
        self, x: np.ndarray, y: np.ndarray, threshold: float = 0.3, **kwargs
    ) -> np.ndarray:
        """
        Applies FFT-based noise reduction by filtering high-frequency components.

        :param x: np.ndarray, x-axis values
        :param y: np.ndarray, y-axis values
        :param threshold: float, fraction of frequencies to retain (default 0.1)
        :param kwargs: additional arguments
        :return: np.ndarray, noise-reduced y-axis values
        """
        # Perform FFT on the signal
        fft_coeffs = np.fft.fft(y.flatten())
        frequencies = np.fft.fftfreq(len(y), d=x[1] - x[0])
        # Calculate magnitude spectrum
        magnitude_spectrum = np.abs(fft_coeffs)
        # Filter out high-frequency components

        # Zero out high frequencies beyond the threshold
        fft_coeffs[np.abs(frequencies) >= np.max(frequencies) * threshold] = 0

        # Perform the inverse FFT to get the filtered signal
        y_denoised = np.fft.ifft(fft_coeffs).real
        return y_denoised.reshape(-1, 1)

    @extract_columns
    def _integrate_peaks(
        self, y: np.ndarray, x: np.ndarray, peak_positions: np.ndarray
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """
        Calculate the integration of the peaks in the spectrum. Assumes perfect baseline and good smoothing.

        :param y: 1D array of cropped data
        :param x: 1D array of x-axis values
        :param peak_positions: 1D array of peak positions
        :return: Tuple containing a list of integration values and a list of tuples for integration bounds
        """
        bottom_width = peak_widths(y, peak_positions, rel_height=0.95)[0]
        integration_values = []
        integration_bounds = []
        for i, peak_pos in enumerate(peak_positions):
            start_index = max(int(peak_pos - bottom_width[i] / 2), 0)
            end_index = min(int(peak_pos + bottom_width[i] / 2), len(y))

            bounds = (x[start_index], x[end_index])
            integration_bounds.append(bounds)

            integration_value = self._integrate_basic(x=x, y=y, bounds=bounds)
            integration_values.append(integration_value)

        return integration_values, integration_bounds

    @extract_columns
    def _integrate_basic(
        self, x: np.ndarray, y: np.ndarray, bounds: Tuple[float, float]
    ) -> float:
        """runs trapz on the y axis data between the bounds, first checks that the bounds
        are sorted the right way around
        :param y: 1D array of data
        :param x: 1D array of data's X axis
        :param bounds: tuple of floats, length 2, lower and higher bound of the array

        :return: float, the integration value of y within the X bounds
        """

        tuple_low = min(bounds)
        tuple_high = max(bounds)

        mask = (tuple_low <= x) & (x <= tuple_high)

        x = x[mask]
        y = y[mask]

        return simpson(y=y, x=x)

    def integrate(
            self,
            mode: str = "area",
            bounds: Optional[Tuple[float, float]]= None,
            peak_positions: Optional[np.ndarray] = None,
    ) -> Union[float, Tuple[List[float], List[Tuple[float, float]]]]:
        """
        Public method to integrate either the full area or specific peaks in the spectrum.

        :param mode: Integration mode, either "area" for full integration or "peaks" for peak integration.
        :param y_col: Optional. The column name to integrate if working with a specific column of data.
        :param peak_positions: Optional. Array of peak positions (required if mode="peaks").
        :return:
            - If mode="area": float, the integral of the full area under the curve.
            - If mode="peaks": Tuple containing:
                - A list of integration values for each peak.
                - A list of tuples for the integration bounds of each peak.
        :raises ValueError: If required parameters are missing or invalid.
        """
        # Validate mode
        if mode not in ["area", "peaks"]:
            raise ValueError("Invalid mode. Use 'area' or 'peaks'.")

        # Validate peak positions if mode is "peaks"
        if mode == "peaks" and peak_positions is None:
            raise ValueError("Peak positions must be provided for 'peaks' mode.")
        if mode == "area" and bounds is None:
            raise ValueError("Bounds must be provided for 'area' mode.")

        if mode == "area":
            # Integrate the full area under the curve
            return self._integrate_basic(bounds=bounds)

        elif mode == "peaks":
            # Integrate specific peaks
            return self._integrate_peaks(peak_positions=peak_positions)


    def crop_data(self, bounds: Tuple[float, float]) -> None:
        """
        Crop the data within the specified x-axis range.

        :param bounds: tuple of floats, lower and upper bounds for the x-axis
        """
        if bounds[0] > bounds[1]:
            bounds = bounds[::-1]

        self.data = self.data[
            (self.data[self.x_col] >= bounds[0]) & (self.data[self.x_col] <= bounds[1])
        ]
        self.data.reset_index(drop=True, inplace=True)

    @extract_columns
    def _quantify_snr(self, y: np.ndarray, noise_sample: np.ndarray) -> float:
        """
        Calculate the signal-to-noise ratio (SNR) of the signal sample using the noise sample.

        :param noise_sample: 1D array of noise sample
        :param signal_sample: 1D array of signal sample
        :return: float, signal-to-noise ratio
        """
        baseline = np.mean(noise_sample)
        signal_mean = np.max(y) - baseline
        noise_std = np.std(noise_sample)

        return signal_mean / noise_std

    @extract_columns
    def _find_peaks(
        self,
        x: np.ndarray,
        y: np.ndarray,
        number_of_peaks: int = 1,
        prominence: float = 0.1,
        distance: float = 0.05,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the numper_of_peaks largest peaks in the data, return their positions and heights.

        :param y: 1D array of y-axis values
        :param x: 1D array of x-axis values
        :param number_of_peaks: int, number of peaks to find (default 1) if -1 finds all peaks and looks for prominence
        :param prominence: float, minimum prominence of the peaks (default 0.1)
        :param distance: float, minimum distance between peaks (default 0.05)
        :return: Tuple containing the peak positions and the peak heights, the FWHM and the right and left bounds at bottom
        """
        y = y.flatten()
        if number_of_peaks == -1:
            peak_positions, _ = find_peaks(
                y,
                distance=int(len(y) * distance),
                prominence=prominence,
                threshold=kwargs.get("threshold", None),
                height=kwargs.get("height", None),
            )
            best_n_peaks = np.argsort(y[peak_positions])[::-1]
        else:
            peak_positions, _ = find_peaks(
                y,
                distance=int(len(y) * distance),
                prominence=prominence,
                threshold=kwargs.get("threshold", None),
                height=kwargs.get("height", None),
            )
            best_n_peaks = np.argsort(y[peak_positions])[-number_of_peaks:][::-1]

        fwhm = peak_widths(
            y,
            peak_positions[best_n_peaks],
            rel_height=kwargs.get("rel_height_fwhm", 0.5),
        )[0]
        left_bounds = peak_widths(
            y,
            peak_positions[best_n_peaks],
            rel_height=kwargs.get("rel_height_widths", 0.95),
        )[2]
        right_bounds = peak_widths(
            y,
            peak_positions[best_n_peaks],
            rel_height=kwargs.get("rel_height_widths", 0.95),
        )[3]

        # these three values are in units of index, we need to convert them to x values
        fwhm = fwhm * (x[-1] - x[0]) / len(x)
        left_bounds = x[left_bounds.round(decimals=0).astype(int)]
        right_bounds = x[right_bounds.round(decimals=0).astype(int)]

        return (
            x[peak_positions[best_n_peaks]],
            y[peak_positions[best_n_peaks]],
            fwhm,
            left_bounds,
            right_bounds,
        )

    @extract_columns
    def _find_noise_sample(
        self, x: np.ndarray, y: np.ndarray, peak_positions: np.ndarray, fwhm: np.ndarray
    ) -> np.ndarray:
        """
        function finds a sample of noise in the data, by looking at places without peaks

        :param y: 1D array of y-axis values
        :param x: 1D array of x-axis values
        :param peak_positions: 1D array of peak positions (in units of X)
        :param fwhm: 1D array of FWHM of the peaks (in units of X)

        :return: 1D array of noise sample
        """
        if len(peak_positions) != len(fwhm):
            raise ValueError("peak_positions and fwhm must have the same length.")

        # Initialize an array to store indices around each peak
        peak_noise_indices = np.array([], dtype=int)

        for pos, width in zip(peak_positions, fwhm):
            # Calculate lower and upper bounds for each peak in x-units
            lower_bound = pos - 3 * width
            upper_bound = pos + 3 * width

            # Find indices in x where values fall within this range
            peak_indices = np.where((x >= lower_bound) & (x <= upper_bound))[0]
            peak_noise_indices = np.concatenate((peak_noise_indices, peak_indices))

        # Exclude the peak regions from y to get a noise sample
        noise_sample_y = np.delete(y, peak_noise_indices)
        noise_sample_x = np.delete(x, peak_noise_indices)
        return noise_sample_y, noise_sample_x

    @staticmethod
    def _relative_threshold_similarity(peak1, peak2, threshold=0.05):
        """Checks similarity based on a relative difference threshold for mu, sigma, and gamma parameters."""
        for param in ["mu", "min_x", "max_x"]:
            if (
                abs(
                    (getattr(peak1, param) - getattr(peak2, param))
                    / max(abs(getattr(peak1, param)), abs(getattr(peak2, param)), 1e-8)
                )
                > threshold
            ):
                return 0.0
        return 1.0

    @staticmethod
    def _weighted_similarity(peak1, peak2, threshold=0.05, weights=None):
        """
        Calculates a weighted similarity score based on relative differences for mu, sigma, gamma, and amplitude.

        :param weights: dict, specifying weights for each parameter (default: {"mu": 0.5, "sigma": 0.2, "gamma": 0.2, "amplitude": 0.1}).
        """
        if weights is None:
            weights = {
                "mu": 0.5,
                "sigma": 0.2,
                "gamma": 0.2,
                "amplitude": 0.1,
                "min_x": 0.1,
                "max_x": 0.1,
            }

        def relative_difference(val1, val2):
            return abs(val1 - val2) / max(abs(val1), abs(val2), 1e-8)

        score = 0.0
        for param, weight in weights.items():
            diff = relative_difference(getattr(peak1, param), getattr(peak2, param))
            score += weight * (1 - min(diff / threshold, 1.0))
        return min(score, 1.0)

    @staticmethod
    def _mu_only_similarity(peak1, peak2, threshold=0.05):
        """Checks similarity based solely on the mu (center) value within the given threshold."""
        return (
            1.0
            if abs((peak1.mu - peak2.mu) / max(abs(peak1.mu), abs(peak2.mu), 1e-8))
            <= threshold
            else 0.0
        )

    @staticmethod
    def _correlation_similarity(peak1, peak2, num_points=100):
        """
        Compares the profiles of two peaks by normalizing their amplitudes and calculating the area of overlap.

        :param num_points: int, the number of points used to calculate the correlation.
        :return: float, a similarity score between 0 and 1, where 1 means high correlation.
        """
        # Create x-axis values based on the range covering both peaks
        x_values = np.linspace(
            min(peak1.min_x, peak2.min_x), max(peak1.max_x, peak2.max_x), num_points
        )

        # Generate normalized profiles
        profile1 = (
            peak1.amplitude
            * np.exp(-((x_values - peak1.mu) ** 2) / (2 * peak1.sigma**2))
            * peak1.gamma
            / (np.pi * ((x_values - peak1.mu) ** 2 + peak1.gamma**2))
        )
        profile2 = (
            peak2.amplitude
            * np.exp(-((x_values - peak2.mu) ** 2) / (2 * peak2.sigma**2))
            * peak2.gamma
            / (np.pi * ((x_values - peak2.mu) ** 2 + peak2.gamma**2))
        )

        # Normalize profiles
        profile1 /= profile1.max()
        profile2 /= profile2.max()

        # Calculate overlap using cosine similarity as a proxy for correlation
        return 1 - cosine(profile1, profile2)

    @staticmethod
    def _fwhm_similarity(peak1, peak2, threshold=0.05):
        """
        Compares the Full Width at Half Maximum (FWHM) of both peaks.

        :param threshold: float, maximum relative difference allowed for FWHM comparison.
        :return: float, similarity score (1 if within threshold, otherwise 0).
        """
        fwhm1 = (
            0.53 * peak1.gamma + (0.216 * peak1.gamma**2 + 0.75 * peak1.sigma**2) ** 0.5
        )
        fwhm2 = (
            0.53 * peak2.gamma + (0.216 * peak2.gamma**2 + 0.75 * peak2.sigma**2) ** 0.5
        )
        relative_diff = abs(fwhm1 - fwhm2) / max(abs(fwhm1), abs(fwhm2), 1e-8)
        return 1.0 if relative_diff <= threshold else 0.0

    @staticmethod
    def _overlap_area_similarity(peak1, peak2, num_points=100):
        """
        Calculates the area of overlap between the two normalized peak profiles.

        :param num_points: int, number of points used to sample the profiles.
        :return: float, similarity score between 0 and 1 (1 indicates high overlap).
        """
        x_values = np.linspace(
            min(peak1.min_x, peak2.min_x), max(peak1.max_x, peak2.max_x), num_points
        )
        profile1 = Glama.voigt(
            x_values, peak1.amplitude, peak1.mu, peak1.sigma, peak1.gamma
        )
        profile2 = Glama.voigt(
            x_values, peak2.amplitude, peak2.mu, peak2.sigma, peak2.gamma
        )

        # Normalize profiles
        profile1 /= simpson(profile1, x=x_values)
        profile2 /= simpson(profile2, x=x_values)

        # Calculate overlap area
        overlap_area = simpson(np.minimum(profile1, profile2), x=x_values)
        return overlap_area

    @staticmethod
    def _dtw_similarity(peak1, peak2, num_points=100):
        """
        Calculates similarity using Dynamic Time Warping (DTW) to allow for flexible alignment.

        :param num_points: int, the number of points used to calculate the profiles.
        :return: float, a similarity score between 0 and 1, where 1 indicates high similarity.
        """
        x_values = np.linspace(
            min(peak1.min_x, peak2.min_x), max(peak1.max_x, peak2.max_x), num_points
        )
        profile1 = Glama.voigt(
            x_values, peak1.amplitude, peak1.mu, peak1.sigma, peak1.gamma
        )
        profile2 = Glama.voigt(
            x_values, peak2.amplitude, peak2.mu, peak2.sigma, peak2.gamma
        )

        # Normalize profiles
        profile1 /= max(profile1)
        profile2 /= max(profile2)

        # Calculate DTW distance
        distance, _ = fastdtw(
            profile1.reshape(-1, 1), profile2.reshape(-1, 1), dist=euclidean
        )
        normalising = euclidean(np.zeros_like(profile1), profile1) + euclidean(
            np.zeros_like(profile2), profile2
        )
        similarity = 1 - distance / normalising
        return max(0.0, min(similarity, 1.0))

    @staticmethod
    def _cross_correlation_similarity(peak1, peak2, num_points=100, shift_penalty=0.1):
        """
        Calculates similarity based on cross-correlation of the profiles.

        :param num_points: int, the number of points used to calculate the profiles.
        :return: float, a similarity score between 0 and 1, where 1 indicates high similarity.
        """
        x_values = np.linspace(
            min(peak1.min_x, peak2.min_x), max(peak1.max_x, peak2.max_x), num_points
        )
        profile1 = peak1.amplitude * norm.pdf(x_values, peak1.mu, peak1.sigma)
        profile2 = peak2.amplitude * norm.pdf(x_values, peak2.mu, peak2.sigma)

        # Normalize profiles
        profile1 /= profile1.max()
        profile2 /= profile2.max()

        # Compute cross-correlation
        correlation = correlate(profile1, profile2, mode="full")
        max_correlation_index = np.argmax(correlation)
        max_correlation = correlation[max_correlation_index]

        # Penalty for shift: If peak is centered, max_correlation_index will be near the center
        ideal_index = len(correlation) // 2
        shift_distance = abs(max_correlation_index - ideal_index)
        shift_penalty_factor = 1 - (shift_penalty * shift_distance / num_points)

        # Compute area overlap for additional normalization
        area_overlap = np.trapz(np.minimum(profile1, profile2), x_values)
        area_total = np.trapz(profile1, x_values) + np.trapz(profile2, x_values)
        area_similarity = 2 * area_overlap / area_total  # Jaccard-like overlap measure

        # Final similarity score: combine max correlation with area similarity and shift penalty
        similarity_score = (
            (max_correlation / max(correlation))
            * area_similarity
            * shift_penalty_factor
        )
        return max(0.0, min(similarity_score, 1.0))

    @staticmethod
    def peaks_same(
        peak1: VoigtPeakIdentification,
        peak2: VoigtPeakIdentification,
        mode: Literal[
            "relative_threshold",
            "weighted_similarity",
            "mu_only",
            "correlation",
            "fwhm",
            "overlap_area",
            "dtw",
            "cross_correlation",
        ] = "relative_threshold",
        threshold: float = 0.05,
        weights: dict = None,
        **kwargs,
    ) -> float:
        """
        Checks for similarity between two VoigtPeakIdentification instances based on the selected mode.

        :param peak1: VoigtPeakIdentification, the first peak to compare.
        :param peak2: VoigtPeakIdentification, the second peak to compare.
        :param mode: str, the comparison mode to use:
            - "relative_threshold": Checks if parameters are within a relative difference threshold.
            - "weighted_similarity": Calculates a weighted similarity score.
            - "mu_only": Only checks the center position (mu).
            - "correlation": Compares the peak profiles based on amplitude-normalized overlap.
            - "fwhm": Compares the FWHM (Full Width at Half Maximum) of both peaks.
            - "overlap_area": Calculates the area of overlap between the two normalized peak profiles.
            - "dtw": Uses Dynamic Time Warping to calculate similarity.
            - "cross_correlation": Uses cross-correlation for similarity of peak profiles.
        :param threshold: float, the relative difference threshold for applicable modes.
        :param weights: dict, parameter weights for "weighted_similarity" mode.

        :return: float, similarity score between 0 and 1 (1 indicates high similarity).
        """
        if mode == "relative_threshold":
            return Glama._relative_threshold_similarity(
                peak1, peak2, threshold=threshold
            )

        elif mode == "weighted_similarity":
            return Glama._weighted_similarity(
                peak1, peak2, threshold=threshold, weights=weights
            )

        elif mode == "mu_only":
            return Glama._mu_only_similarity(peak1, peak2, threshold=threshold)

        elif mode == "correlation":
            return Glama._correlation_similarity(
                peak1, peak2, num_points=kwargs.get("num_points", 100)
            )

        elif mode == "fwhm":
            return Glama._fwhm_similarity(peak1, peak2, threshold=threshold)

        elif mode == "overlap_area":
            return Glama._overlap_area_similarity(
                peak1, peak2, num_points=kwargs.get("num_points", 100)
            )

        elif mode == "dtw":
            return Glama._dtw_similarity(
                peak1, peak2, num_points=kwargs.get("num_points", 100)
            )

        elif mode == "cross_correlation":
            return Glama._cross_correlation_similarity(
                peak1,
                peak2,
                num_points=kwargs.get("num_points", kwargs.get("num_points", 100)),
            )

        else:
            raise ValueError(
                "Invalid mode. Choose from the available similarity methods."
            )

    def load_model_from_class(self, vicuna_class: "Vicuna"):
        """
        loads model from a vicuna class, if the vicuna class has multiple models it will load all of them,
        alternatively it will load the first model

        """
        self.log("Loading models from a vicuna class", indent="enter")
        if hasattr(self, "models"):
            self.log(
                "Current guanaco class already has some models loaded, adding new models..."
            )
            self.models.update(
                {
                    key: [VoigtPeakIdentification(**row) for _, row in df.iterrows()]
                    for key, df in vicuna_class.voigt_parameters.items()
                }
            )
            self.log("Models added successfully", indent="reset")

        else:
            self.models = {
                key: [VoigtPeakIdentification(**row) for _, row in df.iterrows()]
                for key, df in vicuna_class.voigt_parameters.items()
            }
            self.log("Models loaded successfully", indent="reset")
