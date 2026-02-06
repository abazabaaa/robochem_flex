"""
File: nmr_analysis.py
Author: Oliver Bayley, Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/ombayley, https://github.com/simone16

Description: Standardized analytical class for NMR analysis.
"""

import os.path
from itertools import chain
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
import nmrglue as ng
import matplotlib.pyplot as plt
import time

from omniplatypus.utilities.general import dict_to_str, none_if_empty
from omniplatypus.devices.magritek.spinsolve import SpinsolveClient, SpinsolveProtocol
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    RecipeComponent,
)
from omniplatypus.procedures.experiments.experiment_parameters import (
    ExperimentalParameter,
    NumericalParameter,
)
from omniplatypus.procedures.analytics.analytics_parameters import AnalyticalParameter
from omniplatypus.procedures.analytics.analytics_template import AnalyticsTemplate
from omniplatypus.procedures.analytics.custom_baseline_methods import baseline


class NMRAnalysis(AnalyticsTemplate):
    result_metrics: list[str] = [
        "yield",
        "concentration",
        "integral",
        "width",
        "chemical_shift",
        "pass",
    ]

    _required_parameters: list[AnalyticalParameter] = [
        AnalyticalParameter(name="sample_name", value="test_1"),
        AnalyticalParameter(
            name="target_peak",
            value=0.0,
            units="ppm",
            tag="all",
        ),
        AnalyticalParameter(
            name="yield_calculation_chemical",
            value="",
            tag="all",
        ),
    ]

    _optional_parameters: list[AnalyticalParameter] = [
        # Acquisition parameters
        AnalyticalParameter(
            name="data_folder",
            value=os.path.join("Path", "To", "Data", "Folder"),
        ),
        AnalyticalParameter(name="solvent", value="None", tag="all"),
        AnalyticalParameter(name="comment", value=""),
        AnalyticalParameter(
            name="protocol",
            value=list(SpinsolveClient.protocols.keys())[0],
            discrete_values=sorted(set(SpinsolveClient.protocols.keys())),
            tag="all",
        ),
        # Acquisition protocol parameters
        # Note: not all values are compatible with all protocols
        # the following iterative delirium compiles the parameters based on the info in SpinsolveClient.protocols,
        # so update that to add more.
        *[
            AnalyticalParameter(
                name=protocol_option,
                value=list(
                    filter(
                        lambda x: x is not None,
                        [
                            protocol.get(protocol_option, dict()).get("default", None)
                            for protocol in SpinsolveClient.protocols.values()
                        ],
                    )
                )[0],
                units=list(
                    filter(
                        lambda x: x is not None,
                        [
                            protocol.get(protocol_option, dict()).get("units", "")
                            for protocol in SpinsolveClient.protocols.values()
                        ],
                    )
                )[0],
                discrete_values=none_if_empty(
                    sorted(
                        set(
                            chain(
                                *[
                                    protocol.get(protocol_option, dict()).get(
                                        "values", list()
                                    )
                                    for protocol in SpinsolveClient.protocols.values()
                                ]
                            )
                        ),
                        key=lambda x: abs(float(x)),
                    )
                ),
                tag="all",
            )
            for protocol_option in set(
                chain(
                    *[
                        protocol.keys()
                        for protocol in SpinsolveClient.protocols.values()
                    ]
                )
            )
        ],
        # Processing parameters
        AnalyticalParameter(
            name="target_peak_deviation",
            value=0.1,
            units="ppm",
            tag="simple_integration",
        ),
        AnalyticalParameter(name="min_SN_ratio", value=2, tag="simple_integration"),
        AnalyticalParameter(
            name="max_peak_width",
            value=0.2,
            units="ppm",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="peak_resolution", value=0.01, units="ppm", tag="simple_integration"
        ),
        AnalyticalParameter(
            name="target_peak_calibration_coeff_0",
            value=0.0,
            units="mM",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="target_peak_calibration_coeff_1",
            value=1091.72,
            units="mM/AU",
            tag="simple_integration",
        ),
    ]

    _processing_methods: list[str] = ["simple_integration"]

    def run_analysis_spinsolve(self, parameters: dict[str, AnalyticalParameter]) -> str:
        """
        Run analysis on Magritek Spinsolve device.

        @param parameters: dict[str, AnalyticalParameter]
            All analysis parameters (generate from self.validate_parameters).
        @return: str
            Path to results folder for this measurement.
        """
        sample_name = parameters["sample_name"].value
        result_file_location = str(
            os.path.join(
                parameters["data_folder"].value, parameters["sample_name"].value
            )
        )
        protocol = SpinsolveProtocol(name=parameters["protocol"].value)
        protocol_options = SpinsolveClient.protocols.get(
            parameters["protocol"].value, None
        )
        if protocol_options is None:
            error = ValueError(
                f"No reference options for protocol '{protocol.name}'. "
                "Make sure all supported options are listed in SpinsolveClient.protocols ."
            )
            self.log(error)
            raise error
        # take in only the valid options.
        for option_name in protocol_options.keys():
            protocol.options[option_name] = parameters[option_name].value
        self.log(f"Measuring {sample_name}...")
        try:
            self._device.reopen()
            self._device["sample"] = sample_name
            self._device["solvent"] = parameters["solvent"].value
            self._device["comment"] = parameters["comment"].value
            self._device["data_folder"] = result_file_location
            self._device["start"] = protocol  # returns when measurement is done.
        finally:
            self._device.close()
        self.log(f"Measured {sample_name}.", level="ok")
        return result_file_location

    def wait_for_file(self, path: str, timeout: float = 10.0) -> None:
        """
        Wait until a result file is generated.

        @param path: str
            path to the result file.
        @param timeout: float = 10.0
            Maximum timeout in S.
        @raise RuntimeError
            If the timeout is reached.
        """
        endtime = time.time() + timeout
        while time.time() <= endtime:
            if os.path.isfile(path):
                return
            time.sleep(0.5)
        error = RuntimeError(
            "Result file has not been produced within the expected time limit."
        )
        error.add_note(f"Expected file: {path}")
        self.log(error)
        raise error

    def open_spinsolve(self, sample_folder: str) -> tuple[np.array, np.array, dict]:
        """
        Open spinsolve output files.

        @param sample_folder: str
            Location of the measurement files.
        @return: tuple[np.array, np.array, dict]
            spectrum intensity data,
            spectrum chemical shift data,
            general measurement data
        """
        filename = "spectrum_processed.1d"
        filepath = os.path.join(sample_folder, filename)
        # open file, when ready
        self.wait_for_file(filepath)
        measurement, intensity = ng.spinsolve.read(dir=sample_folder, specfile=filename)

        # get x values in ppm
        universal_dictionary = ng.spinsolve.guess_udic(measurement, intensity)
        unit_conversion_object = ng.fileio.fileiobase.uc_from_udic(universal_dictionary)
        shift_ppm = unit_conversion_object.ppm_scale()
        intensity = intensity[::-1]

        return intensity, shift_ppm, measurement

    @staticmethod
    def optimize_phase(
        data: np.ndarray, ref_peak_ppm: float, ppm_scale: np.ndarray
    ) -> tuple[float, float]:
        """
        Optimize ph0 and ph1 to minimize the imaginary part at the reference peak.
        """

        # Function to minimize
        def objective(phase_params):
            ph0, ph1 = phase_params
            corrected_data = ng.proc_base.ps(data, p0=ph0, p1=ph1)
            # Find index of reference peak
            peak_idx = np.argmin(np.abs(ppm_scale - ref_peak_ppm))
            # Minimize imaginary part at the reference peak
            return np.abs(np.imag(corrected_data[peak_idx]))

        # Initial guesses for ph0 and ph1
        initial_guess = np.array([0.0, 0.0])

        # Optimize ph0 and ph1
        result = minimize(objective, initial_guess, method="Nelder-Mead")
        ph0_opt, ph1_opt = result.x

        return ph0_opt, ph1_opt

    def process_spectrum(
        self, intensity_data: np.ndarray, ppm_shift_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Phase correction, baseline correction, denoise.
        No FT.

        @param intensity_data: numpy.ndarray
            1D numpy array of the spectral intensities (Y axis values)
        @param ppm_shift_data: np.ndarray
            1D numpy array of the chemical shift in ppm (X axis values)
        @return: tuple[np.ndarray, np.ndarray]
            intensity_data and ppm_shift_data after processing.
        """
        guess_initial_phase_prams = False
        phase_funct = "peak_minima"  # Can be 'acme' or 'peak_minima'

        # Correct the spectra phase
        if guess_initial_phase_prams:
            largest_point = max(abs(intensity_data))
            p0, p1 = self.optimize_phase(
                data=intensity_data,
                ref_peak_ppm=largest_point,
                ppm_scale=ppm_shift_data,
            )
            self.log(f"Initial phase parameters estimated: p0={p0:.2f}, p1={p1:.2f}")
            intensity_data = ng.process.proc_autophase.autops(
                intensity_data, fn=phase_funct, disp=False, p0=p0, p1=p1
            )
        else:
            intensity_data = ng.process.proc_autophase.autops(
                intensity_data, fn=phase_funct, disp=False
            )
        self.log(f"Phase correction applied using model: {phase_funct}.")

        # Remove the imaginary component from the data
        intensity_data = ng.process.proc_base.di(intensity_data)
        self.log("Imaginary components removed")

        # Adjust if phase has made negative peaks rather than positive (i.e. 180 out of phase)
        if abs(min(intensity_data)) > max(intensity_data):
            intensity_data = intensity_data * -1

        # Apply a Gaussian filter to help denoise the spectra
        sigma = 5
        intensity_data = scipy.ndimage.gaussian_filter1d(intensity_data, sigma=sigma)
        self.log(f"Gaussian filter applied with kernel standard deviation of: {sigma}.")

        # Apply baseline correction
        # Available methods are: 'als', 'whittaker', 'polynomial', 'savgol', 'flatfit', 'median'
        method = "median"
        intensity_data, _ = baseline(data=intensity_data, method=method)
        self.log(f"Baseline correction applied ({method}).")

        return intensity_data, ppm_shift_data

    @staticmethod
    def noise_level(intensity_data: np.ndarray, sampling_percent: float = 75) -> float:
        """
        Removes the lowest 5% intensity data and then takes a reading window of 'sampling_percent' size
        and removes all data of higher intesity. This should remove negative and positive peaks
        leaving mostly noisy data. We then calculate the Median Absolute Deviation of this noise data.

        @param intensity_data: np.ndarray
            The processed spectral intensity data
        @param sampling_percent: float
            How much of the spectrum should be used to calculate the noise
        @return: float
            noise_limit: The distance from the centerline (0) covered by the noise
        """
        # Takes the middle x% of the intensity data to remove real peaks from the estimation
        min_bound = 5
        max_bound = min(
            sampling_percent + min_bound, 100
        )  # prevent from exceeding 100%
        noise_data = intensity_data[
            (intensity_data >= np.percentile(intensity_data, min_bound))
            & (intensity_data <= np.percentile(intensity_data, max_bound))
        ]

        # Calculate the Median Absolute Deviation (MAD) and estimate noise
        mad = np.median(np.abs(noise_data - np.median(noise_data)))
        noise_mad = mad * 1.4826  # Scaling factor to approximate standard deviation

        # multiply by 3 as we have a standard deviation and want the limit instead (3 std = 99% coverage in a gaussian)
        return noise_mad * 3

    @classmethod
    def get_peaks(
        cls,
        intensity: np.ndarray,
        shift_ppm: np.ndarray,
        min_sn_ratio: float = 2.0,
        min_distance: float = 0.01,
        plot_results: bool = False,
    ) -> pd.DataFrame:
        """
        Process the spectral data to find peaks and calculate integrals.

        @param intensity: numpy.ndarray
            Array of signal intensity for the spectrum.
        @param shift_ppm: numpy.ndarray
            Array of chemical shift in ppm for the spectrum.
        @param min_sn_ratio: float = 2.0
            Minimum signal to noise ratio for detected peaks.
            Min peak height is calculated based on this.
        @param min_distance: float = 0.01
            Mininum distance between different peaks in ppm.
        @param plot_results: bool = False
            For debugging only!
            Lets you visualize the integrals.
            Note: execution will stop while the window is open.
        @return: pandas.DataFrame
            List of
                'shift': chemical shift of the peak in ppm.
                'integral': peak integral
                'width_at_base': (at 0.50 height) in ppm.
        """
        # Find the peaks in the processed spectrum
        min_peak_height = cls.noise_level(intensity) * min_sn_ratio
        ppm_per_datapoint = float(abs(shift_ppm[-1] - shift_ppm[0]) / len(shift_ppm))
        min_distance_datapoints = int(round(min_distance / ppm_per_datapoint, 0))
        peaks, _ = scipy.signal.find_peaks(
            intensity, height=min_peak_height, distance=min_distance_datapoints
        )

        # Find the beginning and end-points of the peaks at 50% of the height.
        # This ensures area is not calculated in noisy region, even for low height peaks.
        peak_widths = scipy.signal.peak_widths(intensity, peaks=peaks, rel_height=0.50)

        # fill dataframe with peak info
        peaks_integrals = pd.DataFrame(
            {
                "index_max": peaks,
                "index_start": peak_widths[2],
                "index_end": peak_widths[3],
            }
        )
        peaks_integrals["height"] = intensity[peaks_integrals["index_max"]]
        peaks_integrals["index_start"] = (
            peaks_integrals["index_start"].round(0).clip(lower=0.0)
        )
        peaks_integrals["index_end"] = (
            peaks_integrals["index_end"].round(0).clip(upper=len(intensity) - 1)
        )
        peaks_integrals["shift"] = shift_ppm[peaks_integrals["index_max"]]
        peaks_integrals["FWHM"] = (
            shift_ppm[peaks_integrals["index_start"].astype(int)]
            - shift_ppm[peaks_integrals["index_end"].astype(int)]
        )

        # calculate integrals
        integrals = []
        for index, row in peaks_integrals.iterrows():
            integrals.append(
                -scipy.integrate.simpson(
                    intensity[int(row["index_start"]) : int(row["index_end"])],
                    x=shift_ppm[int(row["index_start"]) : int(row["index_end"])],
                )
            )
        peaks_integrals["integral"] = integrals

        if plot_results:
            cls.plot(shift_ppm, intensity, peaks_integrals)

        peaks_integrals.drop(
            labels=["index_start", "index_end", "index_max"], axis=1, inplace=True
        )

        return peaks_integrals

    @classmethod
    def plot(cls, shift_ppm: np.array, intensity: np.array, peaks: pd.DataFrame):
        """This is for debugging mostly."""
        plt.figure(figsize=(10, 6))
        plt.plot(shift_ppm, intensity, label="Spectrum")
        for index, row in peaks.iterrows():
            plt.plot([row["shift"]], [row["height"] * 1.1], "r*")
            plt.fill_between(
                shift_ppm[int(row["index_start"]) : int(row["index_end"])],
                intensity[int(row["index_start"]) : int(row["index_end"])],
                alpha=0.4,
            )
            # plt.gca().set_xlim([0, 10.0])
        plt.gca().invert_xaxis()
        plt.show()

    def analyse(
        self,
        conditions: dict[
            str, ExperimentalParameter | NumericalParameter | AnalyticalParameter
        ],
        recipe: list[RecipeComponent],
        process_only: bool = False,
    ) -> dict:
        """
        Run analysis.

        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run are given here.
        @param recipe: list[RecipeComponent]
            Chemical conditions, reagents and their concentrations, are given here.
        @param process_only: bool = False
            If true, do not run the analysis on the device. Existing result files are analyzed.
            This is handy for building calibration curves from samples, ensuring the same method is used as in the
            automated analysis.
        @return:
        """
        # get all parameters
        all_parameters = self.validate_parameters(conditions)
        sample_name = all_parameters["sample_name"].value

        # start analysis
        result_files = None
        if not process_only:
            if isinstance(self._device, SpinsolveClient):
                result_files = self.run_analysis_spinsolve(all_parameters)
            else:
                error = ValueError(
                    f"Invalid device type for {self.__class__.__name__} analysis: '{type(self._device)}'."
                )
                self.log(error)
                raise error
        else:
            # get expected filename
            result_files = str(
                os.path.join(
                    all_parameters["data_folder"].value,
                    all_parameters["sample_name"].value,
                )
            )

        # process result
        self.log(f"Processing {sample_name}...", indent="enter")
        spectrum_intensity, spectrum_shift, measurement_info = self.open_spinsolve(
            result_files
        )
        if self._storage_root is not None:
            self.save_files(path=result_files)
        self.log(f"Read result file:\n{dict_to_str(measurement_info)}")
        spectrum_intensity, spectrum_shift = self.process_spectrum(
            spectrum_intensity, spectrum_shift
        )
        peaks = self.get_peaks(
            spectrum_intensity,
            spectrum_shift,
            min_sn_ratio=all_parameters["min_SN_ratio"].value,
            min_distance=all_parameters["peak_resolution"].value,
            plot_results=False,
        )
        self.log(f"Extrapolated peaks:\n{peaks.to_string()}")

        # get concentration
        result = {name: 0.0 for name in self.result_metrics}
        result["chemical_shift"] = np.nan
        result["pass"] = True

        # find target concentration
        target_concentration = self.get_reference_concentration(
            reference_compound=all_parameters["yield_calculation_chemical"].value,
            recipe=recipe,
        )

        best_peak = self.get_matching_peak(
            peaks,
            target_column="shift",
            target=all_parameters["target_peak"].value,
            max_deviation=all_parameters["target_peak_deviation"].value,
        )

        if best_peak is not None:
            result.update(
                {
                    "integral": best_peak["integral"],
                    "width": best_peak["FWHM"],
                    "chemical_shift": best_peak["shift"],
                    "pass": best_peak["FWHM"] < all_parameters["max_peak_width"].value,
                }
            )

            concentration = self.get_concentration(
                best_peak["integral"],
                coefficients=[
                    all_parameters["target_peak_calibration_coeff_0"].value,
                    all_parameters["target_peak_calibration_coeff_1"].value,
                ],
            )
            result["concentration"] = concentration

            if target_concentration is not None:
                chemical_yield = concentration / target_concentration * 100
                result["yield"] = chemical_yield

        self.log(
            f"Processed {sample_name}:\n{dict_to_str(result)}.",
            level="ok",
            indent="exit",
        )
        return result


import random


class DummyNMRAnalysis(NMRAnalysis):
    """
    For testing only.
    """

    random.seed()

    def analyse(
        self,
        conditions: dict[
            str, ExperimentalParameter | NumericalParameter | AnalyticalParameter
        ],
        recipe: list[RecipeComponent],
        process_only: bool = False,
    ) -> dict:
        return {"yield": random.randrange(101)}
