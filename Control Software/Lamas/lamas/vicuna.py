"""
Author: Elia Savino
GitHub: github.com/EliaSavino

Happy Hacking!

Descr:
Vicunas are a subspecies of Llamas known to be the best peak fitters in the animal kingdom, due to their ability to climb
the highest peaks in the Andes mountains.

This module does something similar, it takes a spectrum and proceeds to automatically fit a number of Voigt peaks to describe
the spectrum for our deconvolutions later.
"""

import os.path
from typing import List, Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy import interpolate
from lamas.utils import (
    PeakIdentification,
    VoigtPeakIdentification,
    LamaParameter,
    LamaMethod,
)
from lamas.glama import Glama
import inspect
import matplotlib.pyplot as plt


class Vicuna(Glama):
    """This class is responsible for fitting the peaks and generating model spectra
    Usage: instantiate the class Vicuna(**kwargs), kwargs see Glama class for more info

    load the data with the load_data method (for doc see Glama class)

    OPTIONAL: provide concentration of the components in the spectrum with the provide_concentrations method
    Vicuna.provide_concentrations(["column_name1", "column_name2"], [{"Conc_A": 1.0, "Conc_B": 0.5}, {"Conc_A": 0.5, "Conc_B": 1.0}]) (this will be made into a DF available for Guanaco Later)

    fit the peaks with the fit_all method:
    Vicuna.fit_all(**kwargs)
    kwargs:
    fitting kwargs
        termination_method: list of str, termination criteria to apply.
                                        Options: 'peak_limit', 'residual_stabilization', 'no_more_successful_fits'
        residual_threshold: float, threshold for residual stabilization.
        stabilization_window: int, number of iterations to consider for stabilization.
        min_improvement: float, minimum residual improvement to consider a fit successful.
    baseline kwargs:
       baseline_method: str, the method to use for baseline generation. (default: "als", see Glama for available options) each method requires different kwargs, see Glama for more info

    FWHM calculation kwargs:
        distance: int, minimum distance between peaks (default: 1)
        prominence: float, minimum prominence of peaks (default: 0.1)
        height: float, minimum height of peaks (default: None)
        peak_multiplier: int, multiplier for the number of peaks (default: 3)
        see Glama._find_peaks for more info





    """

    zero_crossing_buffer: int = 5
    max_residual_history: dict[List[float]] = {}
    stop_flag: dict[bool] = {}
    concentrations: pd.DataFrame = None
    max_retries: int = 3

    structure: Dict[str, Any] = {
        "fit_all": LamaMethod(
            method_name="fit_all",
            parameters=[
                LamaParameter(
                    name="termination_method",
                    type=List[
                        Literal[
                            "peak_limit",
                            "residual_stabilization",
                            "no_more_successful_fits",
                        ]
                    ],
                    default=[
                        "peak_limit",
                        "residual_stabilization",
                        "no_more_successful_fits",
                    ],
                    description="Termination criteria to apply.",
                ),
                LamaParameter(
                    name="residual_threshold",
                    type=float,
                    default=0.05,
                    description="Threshold for residual stabilization.",
                ),
                LamaParameter(
                    name="stabilization_window",
                    type=int,
                    default=5,
                    description="Number of iterations to consider for stabilization.",
                ),
                LamaParameter(
                    name="min_improvement",
                    type=int,
                    default=1e-4,
                    description="Minimum residual improvement to consider a fit successful.",
                ),
            ],
            submethods=[],
        )
    }

    def __init__(self, **kwargs) -> None:
        """Initialize the Vicuna class by calling the superclass initializer."""
        super().__init__(**kwargs)

    def provide_concentrations(
        self, y_col: List[str], concentrations: List[Dict[str, float]]
    ) -> None:
        """
        Provide the concentration of each component known to be in the spectrum. This is saved in a separate DataFrame.

        :param y_col: List of filenames or columns the concentrations correspond to.
        :param concentrations: List of dictionaries containing concentration values for each component.
                               Each dictionary may have different combinations of compounds
                               (e.g., {"Conc_A": 1.0, "Conc_B": 0.5}).
        """
        self.log(
            "Providing concentrations for each component in the spectrum.",
            indent="enter",
        )
        if self.concentrations is None:
            self.log("Initializing concentrations DataFrame.")
            self.concentrations = pd.DataFrame(columns=["filename"])

        self.log("Processing concentration data.")
        new_rows = []
        for i, conc_dict in enumerate(concentrations):
            filename = y_col[i]

            if filename in self.concentrations["filename"].values:
                for compound, value in conc_dict.items():
                    if compound not in self.concentrations.columns:
                        self.concentrations[compound] = (
                            0.0  # Add new compound column with zeros
                        )
                    self.concentrations.loc[
                        self.concentrations["filename"] == filename, compound
                    ] = value
            else:
                new_row = {"filename": filename, **conc_dict}
                new_rows.append(new_row)

        if new_rows:
            self.log("Adding new concentration data to DataFrame.")
            new_df = pd.DataFrame(new_rows)
            self.concentrations = pd.concat(
                [self.concentrations, new_df], ignore_index=True
            )

        self.concentrations.fillna(0.0, inplace=True)
        self.log("Concentration data processed and stored.", indent="exit")

    def _generate_initial_model(self, **kwargs: Dict[str, Any]) -> None:
        """
        The first step is to model the spectrum by generating a baseline and an initial model spectrum.

        :param kwargs: Additional keyword arguments for baseline generation.
        """
        # Generate the baseline
        self._generate_baseline(method=kwargs.get("baseline_method", "als"), **kwargs)

        # Generate the initial model spectrum
        self._generate_model_spectrum()
        # Generate the residual
        self._calculate_residual()

        # Initialize the max residual history
        self.max_residual_history = {y_col: [] for y_col in self.y_cols}
        self.stop_flags = {y_col: False for y_col in self.y_cols}
        self.ignored_peaks = {y_col: [] for y_col in self.y_cols}
        # Find the number of peaks for each y_col
        self.noise_multiplier = kwargs.get("noise_multiplier", 10)
        self.peak_limit = {}
        self.noise_level = {}
        for y_col in self.y_cols:
            peaks_x, _, fwhm, _, _ = self._find_peaks(
                self.residual[y_col],
                number_of_peaks=-1,
                prominence=kwargs.get("prominence", 0.1),
                distance=kwargs.get("distance", 0.03),
                height=kwargs.get("height", None),
            )
            self.peak_limit[y_col] = kwargs.get("peak_multiplier", 3) * len(peaks_x)
            self.noise_level[y_col] = max(fwhm) / self.noise_multiplier

    def _calculate_fwhm(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
        """
        Calculate the Full Width at Half Maximum (FWHM) of a peak.

        :param x: np.ndarray, the x values of the peak.
        :param y: np.ndarray, the y values of the peak.
        :param kwargs: Additional keyword arguments.
        :return: float, the FWHM of the peak.
        """
        peak_x, peak_y, fwhm, _, _ = self._find_peaks(
            x=x, y=y, number_of_peaks=1, distance=kwargs.pop("distance", 1), **kwargs
        )
        try:
            return fwhm[0]
        except IndexError as e:
            self.log(f"Error calculating FWHM: {e}", level="warning")
            return e

    def _generate_baseline(self, method: str = "als", **kwargs: Any) -> None:
        """
        Generates a baseline for the data using one of the predefined methods
        and stores it in the baseline attribute of the class.

        :param method: str, the method to use for baseline generation.
                      Options include "als", "poly", "cholesky", "linear",
                      "rolling_ball", "wavelet".
        :param kwargs: Additional keyword arguments for the baseline generation method.
        """
        self.baseline = pd.DataFrame({self.x_col: self.data[self.x_col]})
        self.log("Generating baseline for the data.", indent="enter")

        match method:
            case "als":
                self.log("Using Asymmetric Least Squares (ALS) method.")
                func = self._als_baseline
            case "poly":
                self.log("Using Polynomial method.")
                func = self._poly_baseline
            case "cholesky":
                self.log("Using Cholesky method.")
                func = self._cholesky_baseline
            case "linear":
                self.log("Using Linear method.")
                func = self._linear_fit_baseline
            case "rolling_ball":
                self.log("Using Rolling Ball method.")
                func = self._rolling_ball_baseline
            case "wavelet":
                self.log("Using Wavelet method.")
                func = self._wavelet_baseline
            case _:
                self.log("Invalid baseline method. Using ALS method.")
                func = self._als_baseline

        sig = inspect.signature(func)
        basekwargs = {
            name: param.default
            for name, param in sig.parameters.items()
            if param.default is not inspect.Parameter.empty
        }

        if kwargs:
            self.log(
                f"Overwriting baseline defaults with user-provided values: {kwargs}"
            )
            valid_keys = set(sig.parameters.keys())
            invalid_keys = set(kwargs.keys()) - valid_keys
            if invalid_keys:
                self.log(f"Warning: Ignoring invalid kwargs: {invalid_keys}")
            basekwargs.update({k: v for k, v in kwargs.items() if k in valid_keys})

        basekwargs["return_baseline"] = True

        for y_col in self.y_cols:
            y_dat = self.data[y_col].to_numpy()
            self.log(f"Generating baseline for {y_col}.")
            self.baseline[y_col] = func(y=y_dat, **basekwargs)
            self.data[y_col] = self.data[y_col] - self.baseline[y_col]

        self.log("Baseline generated for all y_cols.", indent="exit")

    def _generate_model_spectrum(self, y_col: Optional[str] = None) -> None:
        """
        Generate an initial model spectrum for the data.

        :param y_col: Optional[str], the column name of the y data to generate the model spectrum for.
                     If provided, only the specified y_col will be processed; otherwise, all y_cols are processed.
        """
        assert hasattr(
            self, "baseline"
        ), "No baseline found. Run the baseline function first."

        if not hasattr(self, "voigt_parameters"):
            self.model_spectrum = self.baseline.copy(deep=True)
            return

        if y_col is None:
            for y_col in self.y_cols:
                self._generate_model_spectrum(y_col)
            return

        self.model_spectrum[y_col] = self.baseline[y_col]

        for i in range(len(self.voigt_parameters[y_col])):
            A = self.voigt_parameters[y_col].loc[i, "A"]
            mu = self.voigt_parameters[y_col].loc[i, "mu"]
            sigma = self.voigt_parameters[y_col].loc[i, "sigma"]
            gamma = self.voigt_parameters[y_col].loc[i, "gamma"]

            self.model_spectrum[y_col] += self.voigt(
                self.model_spectrum[self.x_col], A=A, mu=mu, sigma=sigma, gamma=gamma
            )

    def _find_zero_crossing(self, D1: np.ndarray, start_idx: int, step: int) -> float:
        """
        Helper function to find the zero crossing of the first derivative of the residual spectrum.

        :param D1: np.ndarray, the first derivative of the residual spectrum.
        :param start_idx: int, the index to start searching for the zero crossing.
        :param step: int, the direction to search for the zero crossing. (1 for right, -1 for left)
        :return: float, the x value of the zero crossing.
        """
        # make sure the start_idx is within the bounds of the array
        
        st_in = start_idx + step * self.zero_crossing_buffer
        if st_in < 0:
            return self.residual[self.x_col].iloc[0]
        elif st_in >= len(D1):
            return self.residual[self.x_col].iloc[-1]

        previous = D1[st_in]
        for offset in range(
            self.zero_crossing_buffer + step, 4 * self.zero_crossing_buffer
        ):
            idx = start_idx + step * offset
            if idx < 0 or idx >= len(D1) - 1:
                break
            current: object = D1[idx]
            if previous * current <= 0:
                return self.residual[self.x_col].iloc[idx]
            previous = current
            
        # make sure that if you got here it means you either reached the end of the array or the start
        if step == 1:
            return self.residual[self.x_col].iloc[-1]
        elif step == -1:
            return self.residual[self.x_col].iloc[0]
        else:
            raise ValueError("Invalid step value.")

    def _calculate_residual(self, y_col: Optional[str] = None) -> None:
        """
        Calculate the residual between the data and the model spectrum for a specified y_col.

        :param y_col: str, the column name of the y data to calculate the residual for.
        """
        if not hasattr(self, "residual"):
            self.residual = self.data.copy(deep=True)
            self.residual[self.y_cols] = (
                self.data[self.y_cols]
                - self.baseline[self.y_cols]
                - self.model_spectrum[self.y_cols]
            )
            return

        if y_col is None:
            for y_col in self.y_cols:
                self._calculate_residual(y_col)
            return

        self.residual[y_col] = (
            self.data[y_col] - self.baseline[y_col] - self.model_spectrum[y_col]
        )

        for peak in self.ignored_peaks.get(y_col, []):
            self.residual.loc[
                (self.residual[self.x_col] >= peak.min_x)
                & (self.residual[self.x_col] <= peak.max_x),
                y_col,
            ] = self.baseline.loc[
                (self.baseline[self.x_col] >= peak.min_x)
                & (self.baseline[self.x_col] <= peak.max_x),
                y_col,
            ]
        # make the residual all positive:
        self.residual[y_col] = np.abs(self.residual[y_col])

    def _ignore_peak(self, y_col: str, min_x: float, max_x: float) -> None:
        """makes a peak to ignore and add it to the ignored peaks list"""
        self.log(
            f"Ignoring peak region {min_x, max_x} for {y_col}.",
            level="warning",
            indent="enter",
        )
        ignored_peak = PeakIdentification(min_x=min_x, max_x=max_x, y_col=y_col)
        self.ignored_peaks[y_col].append(ignored_peak)
        self._calculate_residual(y_col)
        self.log(
            f"Ignored peak region {min_x, max_x} for {y_col}.",
            level="ok",
            indent="exit",
        )

    def _find_max_residual(
        self, y_col: str, append: bool = True, **kwargs
    ) -> None | float:
        """
        Find the current maximum residual peak from the residual spectrum for a specified y_col.
        Updates the position of the max residual and its peak boundaries.

        :param y_col: str, the column name of the y data to find the max residual for.
        """
        residual_series = self.residual[y_col]
        max_residual = residual_series.max()

        if max_residual <= self.noise_level[y_col] or max_residual < 0:
            self.stop_flag[y_col] = True
            self.log(
                f"Stopping fitting for {y_col}: max residual {max_residual} below threshold.",
                level="warning",
            )
            return

        if append:
            self.max_residual_history[y_col].append(max_residual)
        else:
            return max_residual

        max_idx = residual_series.idxmax()
        max_x = self.data[self.x_col].iloc[max_idx]

        # Calculate the first derivative
        derivative = residual_series.diff().fillna(0).to_numpy()

        # Find left and right boundaries
        left_boundary = self._find_zero_crossing(derivative, max_idx, step=-1)
        right_boundary = self._find_zero_crossing(derivative, max_idx, step=1)

        # Ensure left_boundary < right_boundary
        if left_boundary >= right_boundary:
            self.log(
                f"For {y_col}, left_boundary {left_boundary} >= right_boundary {right_boundary}. Adjusting."
            )
            left_boundary, right_boundary = right_boundary, left_boundary

        # Calculate FWHM within boundaries
        mask = (self.residual[self.x_col] >= left_boundary) & (
            self.residual[self.x_col] <= right_boundary
        )
        x_subset = self.residual[self.x_col][mask]
        y_subset = residual_series[mask]
        fwhm = self._calculate_fwhm(x_subset.to_numpy(), y_subset.to_numpy(), **kwargs)
        if isinstance(fwhm, Exception) and kwargs.get("retry", True):
            fwhm = (right_boundary - left_boundary) / 2

        elif isinstance(fwhm, Exception):
            self.log(f"Error calculating FWHM for {y_col}: {fwhm}", level="warning")
            self.log(
                f"Ignoring peak region {left_boundary, right_boundary} for {y_col}.",
                level="warning",
            )
            self._ignore_peak(y_col, left_boundary, right_boundary)
            return fwhm

        # Append a new row for Voigt parameters with placeholder values
        self.new_params = {
            "A": max_residual,
            "mu": max_x,
            "sigma": fwhm / 3,  # Convert FWHM to sigma (Gaussian)
            "gamma": fwhm / 1,  # Approximate relation
        }
        # Use pd.concat instead of append (deprecated)
        if not hasattr(self, "boundaries"):
            self.boundaries = {y_col: (left_boundary, right_boundary)}
        else:
            self.boundaries[y_col] = (left_boundary, right_boundary)

    def _extend_to_baseline(
        self, x_fit, y_fit, y_baseline, side="right", num_fit_points=3
    ) -> np.ndarray:
        """
        Extend x_fit and y_fit on the specified side using linear extrapolation until y_fit crosses the baseline.

        Parameters:
        - x_fit: Original x data (1D array).
        - y_fit: Original y data (1D array).
        - y_baseline: Baseline y data (1D array).
        - side: 'left' or 'right' indicating which side to extend.
        - num_fit_points: Number of boundary points to use for linear fitting.

        Returns:
        - x_ext: Extended x data for the specified side (1D array).
        - y_ext: Extended y data for the specified side (1D array).
        """
        # Calculate residuals
        self.log(f"Extending {side} side of the fit data.", indent="enter")
        y_residual = y_fit - y_baseline

        # Select boundary points based on the side
        if side == "right":
            x_boundary = x_fit[-num_fit_points:]
            y_boundary = y_residual[-num_fit_points:]
        elif side == "left":
            x_boundary = x_fit[:num_fit_points]
            y_boundary = y_residual[:num_fit_points:]
        else:
            raise ValueError("side must be 'left' or 'right'")

        # Fit a linear trend: y = m*x + c
        coeffs = np.polyfit(x_boundary, y_boundary, 1)
        m, c = coeffs

        # Calculate the x where y = 0 (y_fit crosses baseline)
        if m == 0:
            self.log(
                f"Linear fit has zero slope on the {side} side. No extension performed.",
                level="warning",
                indent="exit",
            )
            return x_fit, y_fit, y_baseline

        x_zero = -c / m

        if side == "right":
            x_current_end = x_fit[-1]
            extension_distance = x_zero - x_current_end
            if extension_distance <= 0:
                self.log(
                    f"No extension needed on the right side.",
                    level="warning",
                    indent="exit",
                )
                return x_fit, y_fit, y_baseline
            # Calculate the number of points to extend
            dx = x_fit[1] - x_fit[0]
            num_extend = int(np.ceil(extension_distance / dx))
            x_ext = x_fit[-1] + dx * np.arange(1, num_extend + 1)
            y_ext = m * x_ext + c + y_baseline[-1]  # Adjust by baseline
            # concat the extension in the right place:
            x_ext = np.concatenate([x_fit, x_ext])
            y_ext = np.concatenate([y_fit, y_ext])
            # cast the baseline to the same length
            y_baseline = np.concatenate(
                [y_baseline, y_baseline[-1] * np.ones(num_extend)]
            )
        elif side == "left":
            x_current_start = x_fit[0]
            extension_distance = x_current_start - x_zero
            if extension_distance <= 0:
                self.log(
                    f"No extension needed on the left side.",
                    level="warning",
                    indent="exit",
                )
                return x_fit, y_fit, y_baseline
            # Calculate the number of points to extend
            dx = x_fit[1] - x_fit[0]
            num_extend = int(np.ceil(extension_distance / dx))
            x_ext = x_fit[0] - dx * np.arange(num_extend, 0, -1)
            y_ext = m * x_ext + c + y_baseline[0]  # Adjust by baseline
            # concat the extension in the right place:
            x_ext = np.concatenate([x_ext, x_fit])
            y_ext = np.concatenate([y_ext, y_fit])
            # cast the baseline to the same length
            y_baseline = np.concatenate(
                [y_baseline[0] * np.ones(num_extend), y_baseline]
            )
        else:
            self.log(
                f"Invalid side: {side}. No extension performed.",
                level="warning",
                indent="exit",
            )
            return x_fit, y_fit, y_baseline
        self.log(f"Extended {side} side of the fit data.", level="ok", indent="exit")
        return x_ext, y_ext, y_baseline

    def _fit_voigt_peak(self, y_col: str) -> None:
        """
        Fit a Voigt peak within the specified boundaries for a given y_col with a retry mechanism.

        :param y_col: str, the y_col to fit.
        :param kwargs: Additional keyword arguments.
        """
        self.log(
            f"Fitting Voigt peak for {y_col} within boundaries {self.boundaries[y_col]}.",
            indent="enter",
        )
        left, right = self.boundaries[y_col]
        mask = (self.residual[self.x_col] >= left) & (
            self.residual[self.x_col] <= right
        )
        x_fit = self.residual[self.x_col][mask].values
        y_fit = self.residual[y_col][mask].values
        y_baseline = self.baseline[y_col][mask].values
        length = len(x_fit)

        f = interpolate.interp1d(x_fit, y_fit, kind="cubic")
        baseline_f = interpolate.interp1d(x_fit, y_baseline, kind="cubic")
        x_fit = np.linspace(x_fit[0], x_fit[-1], 5 * length)
        y_fit = f(x_fit)
        y_baseline = baseline_f(x_fit)

        x_fit, y_fit, y_baseline = self._extend_to_baseline(
            x_fit, y_fit, y_baseline, side="left", num_fit_points=3
        )

        x_fit, y_fit, y_baseline = self._extend_to_baseline(
            x_fit, y_fit, y_baseline, side="right", num_fit_points=3
        )
        #
        # Retrieve the latest initial parameters
        initial_params = [
            self.new_params["A"],
            self.new_params["mu"],
            self.new_params["sigma"],
            self.new_params["gamma"],
        ]

        bounds = ([0, left, 0, 0], [np.inf, right, np.inf, np.inf])

        for attempt in range(1, self.max_retries + 1):
            try:
                self.log("Rough Voigt peak fit attempt.", indent="enter")
                popt, _ = curve_fit(
                    self.voigt,
                    x_fit,
                    y_fit,
                    p0=initial_params,
                    bounds=bounds,
                )

                A_rough, mu_rough, sigma_rough, gamma_rough = popt
                self.log(
                    f"Rough fit parameters: A = {A_rough}, mu = {mu_rough}",
                    indent="exit",
                )

                # step 2: refine the mu fit:
                def voigt_mu(x, mu):
                    return self.voigt(x, A_rough, mu, sigma_rough, gamma_rough)

                self.log("Refining mu fit.", indent="enter")
                popt_mu, _ = curve_fit(
                    voigt_mu, x_fit, y_fit, p0=[mu_rough], bounds=([left], [right])
                )
                mu_refined = popt_mu[0]
                self.log(f"Refined mu fit: mu = {mu_refined}", indent="exit")

                # step 3: refine the other parameters:
                def voigt_refiner(x, A, sigma, gamma):
                    return self.voigt(x, A, mu_refined, sigma, gamma)

                self.log("Refining other parameters.", indent="enter")
                popt_refined, _ = curve_fit(
                    voigt_refiner,
                    x_fit,
                    y_fit,
                    p0=[A_rough, sigma_rough, gamma_rough],
                    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                )

                A_refined, sigma_refined, gamma_refined = popt_refined
                self.log(
                    f"Refined fit parameters: A = {A_refined}, sigma = {sigma_refined}, gamma = {gamma_refined}",
                    indent="exit",
                )
                refined_params = [A_refined, mu_refined, sigma_refined, gamma_refined]
                # Update model spectrum
                fitted_peak = self.voigt(self.data[self.x_col].values, *refined_params)
                self.model_spectrum[y_col] += fitted_peak

                # Calculate area
                area = simpson(y=fitted_peak, x=self.data[self.x_col].values)
                if area < 0 or np.isnan(area):
                    raise ValueError("Invalid area calculated.")

                param_dict = {
                    "A": refined_params[0],
                    "mu": refined_params[1],
                    "sigma": refined_params[2],
                    "gamma": refined_params[3],
                    "area": area,
                    "min_x": left,
                    "max_x": right,
                }

                if not hasattr(self, "voigt_parameters"):
                    self.voigt_parameters = {y_col: pd.DataFrame(param_dict, index=[0])}
                elif y_col not in self.voigt_parameters:
                    self.voigt_parameters = pd.DataFrame(param_dict, index=[0])
                else:
                    self.voigt_parameters[y_col] = pd.concat(
                        [
                            self.voigt_parameters[y_col],
                            pd.DataFrame(param_dict, index=[0]),
                        ],
                        ignore_index=True,
                    )

                self.log(
                    f"Successfully fitted Voigt peak with boundaries ({left, right}) for {y_col} on attempt {attempt}.",
                    level="ok",
                    indent="exit",
                )
                self.log("fitted peak parameters:", indent="enter")
                for k, v in param_dict.items():
                    self.log(f"{k}: {v}")
                self.log("Continuing to next peak.", indent="exit")
                return  # Exit after successful fit

            except (RuntimeError, ValueError) as e:
                self.log(
                    f"Attempt {attempt}: Failed to fit Voigt peak for {y_col}: {e}",
                    level="warning",
                )
                if attempt < self.max_retries:
                    # Slightly vary initial parameters for the next attempt
                    initial_params = [
                        param * np.random.uniform(0.98, 1.02)
                        for param in initial_params
                    ]
                else:
                    # After max retries, mark the peak region as ignored and restore baseline
                    self.log(
                        f"All {self.max_retries} attempts failed for {y_col}. Ignoring peak region.",
                        level="warning",
                        indent="exit",
                    )
                    ignored_peak = PeakIdentification(
                        min_x=left, max_x=right, y_col=y_col
                    )
                    self.ignored_peaks[y_col].append(ignored_peak)
                    # Restore the baseline in the ignored region
                    baseline_mask = (self.data[self.x_col] >= left) & (
                        self.data[self.x_col] <= right
                    )
                    self.residual.loc[baseline_mask, y_col] = self.model_spectrum.loc[
                        baseline_mask, y_col
                    ]
                    self.stop_flag[y_col] = (
                        True  # Optionally stop further fitting for this y_col
                    )
                    return

    def fit_all(
        self,
        termination_method: list = [
            "peak_limit",
            "residual_stabilization",
            "no_more_successful_fits",
        ],
        residual_threshold: float = 0.05,
        stabilization_window: int = 5,
        min_improvement: float = 1e-4,
        **kwargs,
    ):
        """
        Fit Voigt peaks for all y_cols until stopping conditions are met based on termination_method.

        :param termination_method: list of str, termination criteria to apply.
                                    Options: 'peak_limit', 'residual_stabilization', 'no_more_successful_fits'
        :param residual_threshold: float, threshold for residual stabilization.
        :param stabilization_window: int, number of iterations to consider for stabilization.
        :param min_improvement: float, minimum residual improvement to consider a fit successful.
        """
        # Validate termination methods
        valid_methods = {
            "peak_limit",
            "residual_stabilization",
            "no_more_successful_fits",
        }
        for method in termination_method:
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid termination method: {method}. Valid options are {valid_methods}"
                )
        self._generate_initial_model(**kwargs)
        for y_col in self.y_cols:
            self.log(f"Starting fitting for {y_col}", indent="enter")
            len_voigt = 0

            while not self.stop_flags[y_col] and len_voigt < self.peak_limit[y_col]:
                fwhm = self._find_max_residual(y_col, **kwargs)

                # self.plot_stuff()
                if self.stop_flags[y_col]:
                    self.log(
                        f"Stopping condition met for {y_col} based on max residual.",
                        level="ok",
                        indent="enter",
                    )
                    self.log(
                        f"Conditions: stop_flag={self.stop_flags[y_col]}", indent="exit"
                    )
                    break
                if isinstance(fwhm, Exception):
                    self.log(
                        "Error finding max residual. ignored peak, retrying",
                        level="warning",
                    )
                    continue

                # Record the current max residual before fitting
                previous_max_residual = (
                    self.max_residual_history[y_col][-1]
                    if self.max_residual_history[y_col]
                    else np.inf
                )

                self._fit_voigt_peak(y_col)

                # update the model spectrum
                self._generate_model_spectrum(y_col)
                self._calculate_residual(y_col)
                # Reset residual after updating the model
                max_residual = self._find_max_residual(y_col, append=False)

                # if the residual has not increased then we stop, we add the peak to the ignored peaks
                if (
                    isinstance(max_residual, Exception)
                    or max_residual is None
                    or max_residual >= previous_max_residual
                ):
                    self.log(
                        "Residual has increased in the last iteration, ignoring peak region",
                        level="warning",
                    )
                    ignored_peak = PeakIdentification(
                        min_x=self.boundaries[y_col][0],
                        max_x=self.boundaries[y_col][1],
                        y_col=y_col,
                    )
                    self.ignored_peaks[y_col].append(ignored_peak)
                    self.voigt_parameters[y_col].drop(
                        self.voigt_parameters[y_col].index[-1], inplace=True
                    )
                    # Restore the bline and model to the previous state
                    self._generate_model_spectrum(y_col)
                    self._calculate_residual(y_col)
                len_voigt = len(self.voigt_parameters[y_col])
                # Check for residual stabilization
                if "residual_stabilization" in termination_method:
                    history = self.max_residual_history[y_col]
                    if len(history) >= stabilization_window:
                        recent_changes = np.abs(
                            np.diff(history[-stabilization_window:])
                        )
                        if np.all(recent_changes < residual_threshold):
                            self.log(
                                f"Residual stabilization reached for {y_col}.",
                                level="ok",
                                indent="exit",
                            )
                            self.stop_flag[y_col] = True
                            break

                # Check for no more successful fits
                if "no_more_successful_fits" in termination_method:
                    if len(self.max_residual_history[y_col]) > stabilization_window:
                        improvements = np.diff(
                            self.max_residual_history[y_col][
                                -stabilization_window - 1 : -1
                            ]
                        )
                        if np.all(np.abs(improvements) < min_improvement):
                            self.log(
                                f"No significant improvement in residuals for {y_col}. Stopping fitting.",
                                level="ok",
                                indent="exit",
                            )
                            self.stop_flag[y_col] = True
                            break

                # Check peak limit
                if "peak_limit" in termination_method:
                    if (
                        len(self.voigt_parameters[y_col])
                        + len(self.ignored_peaks[y_col])
                        >= self.peak_limit[y_col]
                    ):
                        self.log(
                            f"Peak limit reached for {y_col}.",
                            level="ok",
                            indent="exit",
                        )
                        self.stop_flag[y_col] = True
                        break

            self.log(f"Completed fitting for {y_col}", level="ok", indent="reset")

        self.log("Fitting complete for all y_cols.")

    def voigt_to_df(self) -> pd.DataFrame:
        """
        Converts the dictionary of Voigt parameters into a single DataFrame.

        Each key in the dictionary is added as a new column ('y_col') in the resulting DataFrame,
        which contains all rows from the original DataFrames.

        Returns:
            pd.DataFrame: A concatenated DataFrame with an additional 'y_col' column representing
            the dictionary keys.
        """
        self.log("Converting Voigt parameters to DataFrame.", indent="enter")

        big_voigt_df = pd.concat(
            [
                sub_voigt_df.assign(y_col=y_col)
                for y_col, sub_voigt_df in self.voigt_parameters.items()
            ],
            ignore_index=True,
        )

        self.log("Voigt parameters converted to DataFrame.", indent="exit")
        return big_voigt_df

    def save_voigt_parameters(self, path: Optional[str] = None) -> None:
        """
        Saves the Voigt parameters to a CSV file.

        If a path is not provided, and the instance has a 'data_parent_dir' attribute, the file
        is saved to "voigt_parameters.csv" in that directory. Otherwise, an exception is raised.

        Args:
            path (Optional[str]): The parent directory path.'data_parent_dir'.

        Raises:
            Exception: If no path is provided and 'data_parent_dir' is not set.
        """
        self.log("Saving Voigt parameters to .model file.", indent="enter")

        if path is None and not hasattr(self, "data_parent_dir"):
            self.log(
                "No path provided and no data_parent_dir found.",
                level="error",
                indent="reset",
            )
            raise Exception("No path provided and no data_parent_dir found.")
        elif path is None:
            path = lambda x: os.path.join(
                self.data_parent_dir, f"{self.data_name_file}-{x}.model"
            )
        else:
            path = lambda x: os.path.join(path, f"{self.data_name_file}-{x}.model")

        for y_col in self.y_cols:
            p = path(y_col)
            self.log(f"Saving Voigt parameters to {p}.")
            self.voigt_parameters[y_col].to_csv(path(y_col), index=False)

        self.log("Voigt parameters saved to .model files.", level="ok", indent="exit")

    def plot_stuff(self):
        """debugging function to plot the data, model and residual"""
        plt.plot(
            self.data[self.x_col], self.data[self.y_cols[0]], label="Original Data"
        )
        plt.plot(
            self.data[self.x_col],
            self.model_spectrum[self.y_cols[0]],
            label="Model Spectrum",
        )
        plt.plot(self.data[self.x_col], self.residual[self.y_cols[0]], label="Residual")
        plt.legend()
        plt.show()

    def voigt_list(self, y_col: Optional[str] = None) -> List[VoigtPeakIdentification]:
        """returns a list of VoigtIdentificationPeaks for the y_col

        :param y_col: str, the y_col to return the peaks for, if none returns all peaks
        """

        if y_col is None:
            ret = []
            for y_col in self.y_cols:
                ret.extend(self.voigt_list(y_col))

            return ret

        return [
            VoigtPeakIdentification(y_col=y_col, df_row=row[1])
            for row in self.voigt_parameters[y_col].iterrows()
        ]

    def get_peak(
        self,
        y_col: Optional[str] = None,
        peak_position: Optional[float] = None,
        tolerance: float = 0.05,
    ) -> VoigtPeakIdentification:
        """
        Returns the VoigtPeakIdentification object for the specified peak that  in the specified y_col.

        :param y_col: str, the y_col to return the peak for.
        :param peak_position: float, the peak position to search for by matching the mu parameter.
        :param tolerance: float, the absolute tolerance to consider when matching the peak position.
        """

        if y_col is None:
            ret = []
            for y_col in self.y_cols:
                ret.extend(self.get_peak(y_col, peak_position, tolerance))
            return ret

        for row in self.voigt_parameters[y_col].iterrows():
            if np.isclose(row[1]["mu"], peak_position, atol=tolerance):
                return VoigtPeakIdentification(y_col=y_col, df_row=row[1])

        return None
