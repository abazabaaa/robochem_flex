"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Template for analytics classes

"""

import copy
from abc import ABC, abstractmethod
import os.path
import shutil
import pandas as pd

from omniplatypus.utilities.logger import Logger
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    RecipeComponent,
)
from omniplatypus.procedures.experiments.experiment_parameters import (
    ExperimentalParameter,
    NumericalParameter,
)
from omniplatypus.procedures.analytics.analytics_parameters import AnalyticalParameter
from omniplatypus.devices.base.device import BaseDevice
from typing import Any, Dict
from lamas.lama_sheperd import LamaSheperd
from lamas.alpaca import Alpaca
from lamas.guanaco import Guanaco
from lamas.lama import Lama
from lamas.vicuna import Vicuna


class AnalysisError(Exception):
    """Exception raised when an error occurs during the analysis."""

    pass


class AnalyticsTemplate(ABC):
    """Template for the analytics class, here we are relatively flexible depending on how we plan to
    implement analytics in the future, The general gist of this is we need a method to
    initialise with the option of a few different internal methods. We need a couple of methods to set parameters
    and a method to run the analysis and save the data (spectra)

    """

    _required_parameters: list[
        AnalyticalParameter
    ] = []  # These parameters must be included for each run
    _optional_parameters: list[
        AnalyticalParameter
    ] = []  # These parameters are optionals and can be included if needed

    _processing_methods: list[str] = []  # Available processing methods
    _lama_parameters: Dict[str, Any] = LamaSheperd.structure.copy()

    @classmethod
    def get_processing_method_names(cls) -> list[str]:
        """
        List of supported processing method names.
        Each processing method requires different parameters, as specified by their tags.

        @return: list[str]
            Names of the processing methods which are supported by the analysis.
        """
        return copy.deepcopy(cls._processing_methods)

    @classmethod
    def get_required_parameters(cls) -> list[AnalyticalParameter]:
        """
        List of parameters which must be set for every analysis.

        @return: list[AnalyticalParameter]
            List of parameters which must be set for every analysis.
        """
        return copy.deepcopy(cls._required_parameters)

    @classmethod
    def get_optional_parameters(cls) -> list[AnalyticalParameter]:
        """
        List of parameters which can be set for every analysis.
        If unset, these parameters default to the values listed here.

        @return: list[AnalyticalParameter]
            List of parameters which can be set for every analysis.
        """
        return copy.deepcopy(cls._optional_parameters)

    @classmethod
    def get_lama_parameters(cls) -> Dict[str, Any]:
        """
        Returns the parameters for the lama shepherd, these are used to set the parameters for the lama shepherd

        """
        return cls._lama_parameters

    @classmethod
    def get_all_parameters(cls) -> list[AnalyticalParameter]:
        """
        List of all parameters which can be set for every analysis.
        If unset, some of these parameters default to the values listed here.

        @return: list[AnalyticalParameter]
            List of all parameters which can be set for every analysis.
        """
        return copy.deepcopy(cls._required_parameters + cls._optional_parameters)

    def __init__(
        self,
        analytical_device: BaseDevice | None,
        processing_method: str | None = None,
        storage_root: str | None = None,
    ) -> None:
        """
        Constructor.

        @param analytical_device: BaseDevice | None
            The device which will be used to run the analysis.
        @param processing_method: str | None = None
            If more than one option is available, select the algorithm for processing the data.
        @param storage_root: str | None = None
            If selected, copies the raw result files to this location.
        """
        self._device = analytical_device
        self._processing_method = processing_method
        if self._processing_method is None:
            try:
                self._processing_method = self._processing_methods[0]
            except IndexError:
                pass
        self._storage_root = storage_root

    def save_files(self, path: str) -> None:
        """
        Copies the specified folder into the analysis storage folder.
        In case of failure it does not raise.

        @param path: str
            Path to the folder or file with the results which need to be copied.
        """
        try:
            base_name = os.path.basename(path)
            destination_folder = os.path.join(self._storage_root, "raw_data_analysis")
            if not os.path.isdir(destination_folder):
                os.mkdir(destination_folder)
            destination = os.path.join(destination_folder, base_name)

            if os.path.isdir(path):
                shutil.copytree(path, destination)
            elif os.path.isfile(path):
                shutil.copy2(path, destination)
            else:
                raise FileNotFoundError(
                    f" {path}:  WTF is this, not a file not a directory?"
                )

        except (
            FileNotFoundError,
            TypeError,
            ValueError,
            shutil.Error,
            Exception,
        ) as error:
            self.log(f"Failed to save raw data '{path}', non-fatal.", level="error")
            self.log(error)
        else:
            self.log(f"Saved raw data '{path}'.")

    def validate_parameters(
        self,
        conditions: dict[
            str, ExperimentalParameter | NumericalParameter | AnalyticalParameter
        ],
    ) -> dict[str, AnalyticalParameter]:
        """
        Validates the analytical experiment parameters.
        Check if any required parameter is missing.
        Add all optional parameters with their default values. We can treat the result as
        if the user provided all parameters.
        Parameters which do not belong to analysis are assumed to be experiment parameters and
        are not touched.

        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run are given here.
        @return: dict
            A similar dictionary to the input one, but including all optional parameters as well.
        """
        # Check if required parameters are missing
        required = {parm.name for parm in self._required_parameters}
        given = set(conditions.keys())
        missing = required.difference(given)
        if len(missing) > 0:
            error = ValueError(
                f"{self.__class__.__name__} missing required parameters '{missing}'."
            )
            self.log(error)
            raise error

        # Add optional parameters with default value, if they were not provided by caller.
        optional = {parm.name: parm for parm in self._optional_parameters}
        missing = set(optional.keys()).difference(given)
        for name in missing:
            conditions[name] = copy.deepcopy(optional[name])
        return conditions

    def sample_loading(self, enable=False) -> None:
        """
        Procedure to enable or disable sample loading, for those devices which implement such a system (e.g.: 6-way
        valve for sampling loop). If not implemented, leave this empty.

        @param enable: bool = False
            When set to true, should connect the analysis device to the flow system and allow the platform to load the
            sample. When set to False, the sampling loop is bypassed.
        """
        pass

    def get_reference_concentration(
        self, reference_compound: str, recipe: list[RecipeComponent]
    ) -> float | None:
        """
        Find reference concentration (e.g.: for calculating yields and conversions).

        @param reference_compound: str
            The name of the limiting reagent compound as found in the recipe.
        @param recipe: list[RecipeComponent]
            Recipe for the experiment.
        @return: float | None
            Concentration of the reference limiting compound in mM, or None if this is not specified, not found or 0.
        """
        target_concentration = None
        for compound in recipe:
            if compound.name == reference_compound:
                target_concentration = compound.concentration  # [mM]
                break
        if target_concentration is None:
            error_message = (
                f"Compound '{reference_compound}' not found in recipe '{recipe}'."
                " Cannot calculate yield based on this compound!"
            )
            self.log(error_message, level="error")
        elif target_concentration == 0.0:
            error_message = (
                f"Compound '{reference_compound}' concentration is 0."
                " Cannot calculate yield based on this compound!"
            )
            self.log(error_message, level="error")
            target_concentration = None
        return target_concentration

    def get_matching_peak(
        self,
        peaks: pd.DataFrame,
        target_column: str,
        target: float,
        max_deviation: float,
        reference_spectrum: str | None = None,
        min_spectral_correlation: float = 95.0,
    ) -> pd.Series | None:
        """
        Find the best fit for a peak out of a list of peaks.

        @param peaks: pd.DataFrame
            DataFrame of peaks data.
        @param target_column: str
            Name of the column in the dataframe holding the peak position data.
        @param target: float
            Target value for the peak position.
        @param max_deviation: float
            Max deviation for the position value of the peak.
        @return: pd.Series | None
            The row corresponding to the best matching peak, or None if no suitable peak is found.
        """
        if len(peaks.index) == 0:
            self.log("No valid peaks detected!", level="warning")
            return None
        suitable_peaks = peaks.copy()
        if reference_spectrum is not None:
            if reference_spectrum not in suitable_peaks.columns:
                self.log(
                    f"No spectral correlation data found for '{reference_spectrum}'.\n",
                    level="warning",
                )
                return None
            suitable_peaks.drop(
                suitable_peaks[
                    suitable_peaks[reference_spectrum] < min_spectral_correlation
                ].index,
                inplace=True,
            )
            if suitable_peaks.empty:
                self.log(
                    f"No spectral correlation match found for '{reference_spectrum}'.\n",
                    level="warning",
                )
                return None

        suitable_peaks["delta"] = (suitable_peaks[target_column] - target).abs()
        suitable_peaks.sort_values(by="delta", ignore_index=True, inplace=True)
        best_peak = suitable_peaks.loc[0]
        if best_peak["delta"] > max_deviation:
            self.log(
                "No matching peak found.\n"
                f"Best match at {best_peak[target_column]:.3} (target: {target:.3} ppm).",
                level="warning",
            )
            return None
        return best_peak

    def get_concentration(self, integral: float, coefficients: list[float]) -> float:
        """
        Calculate concentration based on integral using polynomial regression coefficients.

        @param integral: float
            The raw integral value.
        @param coefficients: list[float]
            Polynomial coefficients for the regression starting from index 0 (x^0) and increasing exponents of x.
            Example for linear regression: provide 2 coefficients c = m*integral + q as [q, m].
        @return: float
            Concentration of the compound.
        """
        concentration = 0.0
        for exponent in range(len(coefficients)):
            concentration += coefficients[exponent] * (integral**exponent)
        return concentration

    @abstractmethod
    def analyse(
        self,
        conditions: dict[
            str, ExperimentalParameter | NumericalParameter | AnalyticalParameter
        ],
        recipe: list[RecipeComponent],
    ) -> dict:
        """
        Run analysis.

        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run are given here.
        @param recipe: list[RecipeComponent]
            Chemical conditions, reagents and their concentrations, are given here.
        @return:
        """
        pass

    def log(self, message: str | Exception, **kwargs) -> None:
        """
        Log a message.

        @param message: str | Exception
            The message to log, or an exception.
        """
        kwargs["subfolder"] = "analysis"
        if "origin" in kwargs.keys():
            kwargs.pop("origin")
        kwargs["priority"] = 4  # corresponds to medium verbosity
        Logger.log_message(message, origin=self.__class__.__name__, **kwargs)

    def _clean_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Clean data performs baseline correction, smoothing and other data preprocessing operations

        Since data cleaning works beautifully with lama, we are going to use that. Lama needs to be installed.


        :param data: pd.DataFrame
            The data to be cleaned
        :param kwargs:
            Additional arguments to be passed to lamas.alpacas:
            - "data_type": str = "rama_berry", "nmr_lama"
            - "perform_baseline_correction": bool = True
            - "perform_smoothing": bool = True
            - "normalise": bool = True

        :return: pd.DataFrame
        """
        # for now pass:
        pass

    def _run_analytics(self):
        """finction runs all the required analysis, mostly finding peaks modelling of peaks and calculating integrals"""
        pass
