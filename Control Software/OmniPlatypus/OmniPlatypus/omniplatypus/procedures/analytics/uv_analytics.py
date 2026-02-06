'''
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

'''

import os.path
from itertools import chain
import pandas as pd
import numpy as np
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import os
from omniplatypus.utilities.general import dict_to_str, none_if_empty
from omniplatypus.devices.nrg.rama_berry import RamaBerry
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    RecipeComponent,
)
from omniplatypus.procedures.experiments.experiment_parameters import (
    ExperimentalParameter,
    NumericalParameter,
)
from omniplatypus.procedures.analytics.analytics_parameters import AnalyticalParameter
from omniplatypus.procedures.analytics.analytics_template import (
    AnalyticsTemplate,
    AnalysisError,
)
from typing import Any, Dict, List

from omniplatypus.procedures.analytics.lama_analytics import *
from omniplatypus.utilities.general import (
    get_function_from_globals,
    get_ast_structure,
    get_imported_structure,
)


class AnalyticsUV(AnalyticsTemplate):
    """Raman Analysis Class,
    This class will be used to run the raman analysis on the data from the spectrometer

    #### For now this is a stub, will implement this later after i give a look at the lamas code

    """

    _base_data = os.path.join("Path", "To", "Data", "Folder")
    result_metrics: list[str] = [
        "yield",
        "integral",
        "absorbance_at_wlen",
        "pass",
    ]
    _required_parameters: list[AnalyticalParameter] = [
        AnalyticalParameter(
            name="yield_calculation_chemical",
            value="",
            tag="all",
        ),
        AnalyticalParameter(
            name="integration_time",
            value=1000,
            min_value=0,
            max_value=50000,
            units="ms",
            tag="all",
        ),
        AnalyticalParameter(
            name="n_averages",
            value=1,
            min_value=0,
            max_value=100,
            units="",
            tag="all",
        ),
        AnalyticalParameter(
            name="n_scans",
            value=1,
            min_value=0,
            max_value=10000,
            units="",
            tag="all",
        ),
        AnalyticalParameter(
            name="delay",
            value=0,
            min_value=0,
            max_value=10000,
            units="ms",
            tag="all",
        ),
        AnalyticalParameter(
            name="save_as",
            value="SERVER_SINGLE",
            discrete_values=["SERVER_SINGLE", "SERVER_KINETIC"],
            tag="all",
        ),
        AnalyticalParameter(name="sample_name", value="test"),
        AnalyticalParameter(
            name="data_folder",
            value=os.path.join(
                "\\\\10.10.29.250", "hims-nrg-robochem", "raman_spectra"
            ),
        ),
        AnalyticalParameter(name="dark_correction", value=True, tag="all"),
    ]
    _optional_parameters: list[AnalyticalParameter] = [
        AnalyticalParameter(
            name="integration_lower_bound",
            value=200,
            min_value=190,
            max_value=1100,
            units="cm^-1",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="integration_upper_bound",
            value=250,
            min_value=190,
            max_value=11000,
            units="cm^-1",
            tag="simple_integration",
        ),
        # AnalyticalParameter(
        #     name="peak_of_product",
        #     value=1000,
        #     min_value=0,
        #     max_value=4000,
        #     units="cm^-1",
        #     tag="lamas",
        # ),
        # AnalyticalParameter(
        #     name="Peak_of_interest_Sideproduct_A",
        #     value=1000,
        #     min_value=0,
        #     max_value=4000,
        #     units="cm^-1",
        #     tag="lamas",
        # ),
        # AnalyticalParameter(
        #     name="Peak_of_interest_Sideproduct_B",
        #     value=1000,
        #     min_value=0,
        #     max_value=4000,
        #     units="cm^-1",
        #     tag="lamas",
        # ),
        # AnalyticalParameter(
        #     name="peak_of_starting_material",
        #     value=1000,
        #     min_value=0,
        #     max_value=4000,
        #     units="cm^-1",
        #     tag="lamas",
        # ),
        AnalyticalParameter(name="aggregate", value=False, units="", tag="all"),
        AnalyticalParameter(
            name="molar_extinction_coefficient",
            value=1.0,
            min_value=0.0,
            max_value=1.0e50,
            units="",
            tag="all",
        ),
        AnalyticalParameter(name="path_to_calibration_file", tag="all", value=""),
        # AnalyticalParameter(
        #     name="processing_function",
        #     tag="all",
        #     discrete_values=get_imported_structure(
        #         "omniplatypus.procedures.analytics.lama_analytics",
        #         "functions",
        #         globals(),
        #     ),
        #     value="processing_function_integrated_isotope_exchange",
        # ),
        # AnalyticalParameter(
        #     name="crop_range", value=(0, 4000), tag="all", units="cm^-1"
        # ),
        # AnalyticalParameter(
        #     name="normalise_range", value=(0, 4000), tag="all", units="cm^-1"
        # ),
        # AnalyticalParameter(
        #     name="crop_zoom", value=(0, 4000), tag="all", units="cm^-1"
        # ),
        # AnalyticalParameter(
        #     name="normalise_zoom", value=(0, 4000), tag="all", units="cm^-1"
        # ),
        # AnalyticalParameter(
        #     name="peak_of_product_boundary",
        #     value=1640,
        #     min_value=0,
        #     max_value=4000,
        #     units="cm^-1",
        #     tag="all",
        # ),
        # AnalyticalParameter(
        #     name="peak_of_starting_material_boundary",
        #     value=1720,
        #     min_value=0,
        #     max_value=4000,
        #     units="cm^-1",
        #     tag="all",
        # ),
        # AnalyticalParameter(
        #     name="isosbestic_point",
        #     value=1687,
        #     min_value=0,
        #     max_value=4000,
        #     tag="all",
        # ),
    ]

    _processing_methods = ["single_point_absorbance", "integrated_absorbance"]

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
        # 1. Set the parameters in the spectrometer
        _parameters = self.validate_parameters(conditions)
        non_spectrometer_parameters = self._split_parameters(_parameters)

        # 2. Run the spectrometer
        if not process_only:
            fluorescence_check = self._fluorescence_check(
                do_it=non_spectrometer_parameters["all"]["fluorescence_check"].value,
                treshold=non_spectrometer_parameters["all"][
                    "fluorescence_threshold"
                ].value,
                conditions=conditions,
            )
            if not fluorescence_check:
                self.log("Fluorescence check failed", level="error")
                results = {metric: None for metric in self.result_metrics}
                results["pass"] = False
                return results
            delay = (
                conditions["integration_time"].value * conditions["n_averages"].value
            ) + conditions["delay"].value
            self._set_parameters(conditions)
            aggregate = non_spectrometer_parameters["all"]["aggregate"].value
            data = self._spectrometer_run(delay_time=delay, aggregate=aggregate)
        else:
            data_file = os.path.join(
                non_spectrometer_parameters["experiment_path"].value,
                non_spectrometer_parameters["experiment_name"].value,
            )
            data = self.read_data(data_file)

        metrics = self._process_analytics(
            data=data,
            recipe=recipe,
            non_spectrometer_parameters=non_spectrometer_parameters,
            conditions=conditions,
        )

        results = self._make_results(
            metrics=metrics, recipe=recipe, conditions=conditions
        )
        what_to_copy = str(
            os.path.join(
                conditions["data_folder"].value,
                conditions["sample_name"].value + ".csv",
            )
        )
        self.save_files(path=what_to_copy)
        # 7. return the metrics
        return results

    def _fluorescence_check(
        self, do_it: bool, treshold: int, conditions: dict[str, AnalyticalParameter]
    ):
        """
        runs a short acquisition of 1 second and 1 average to check if there is fluorescence

        :param do_it: bool, if true the check is run
        :param treshold: int, the treshold for the check
        """

        if not do_it:
            return True

        self.log("Running fluorescence check", indent="enter")
        self._device["integration_time"] = 1000
        self._device["n_averages"] = 1
        self._device["n_scans"] = 1
        self._device["delay"] = 0
        self._device["save_as"] = "SERVER_SINGLE"
        self._device["dark_correction"] = True
        data_path = conditions["data_folder"].value.replace(self._base_data, "")
        self._device["save_path"] = data_path
        self._device[
            "save_moniker"
        ] = f"{conditions['sample_name'].value}_fluorescence_check"
        data = self._spectrometer_run(delay_time=1000, aggregate=False)

        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            self.log("No data was returned, not good", level="error", indent="exit")
            raise AnalysisError("No data was returned")

        if data.iloc[:, -1].max() > treshold:
            self.log(
                "Fluorescence detected, something is red and we don't like communism",
                level="warning",
            )
            self.log("Fluorescence check failed", level="error", indent="reset")
            return False

        self.log("Fluorescence check passed", level="ok", indent="exit")
        return True

    def _set_parameters(
        self,
        conditions: dict[
            str, ExperimentalParameter | NumericalParameter | AnalyticalParameter
        ],
    ):
        """
        Makes a dictionary of spectrometer parameters and runs the raman_spec.SetRamanParameters unit task

        :param conditions: dict[str, ExperimentalParameter | NumericalParameter | AnalyticalParameter]
        """
        self.log("Setting raman parameters", indent="enter")

        self._device["integration_time"] = conditions["integration_time"].value
        self._device["n_averages"] = conditions["n_averages"].value
        self._device["n_scans"] = conditions["n_scans"].value
        self._device["delay"] = conditions["delay"].value
        self._device["save_as"] = conditions["save_as"].value
        self._device["dark_correction"] = conditions["dark_correction"].value
        data_path = conditions["data_folder"].value.replace(self._base_data, "")
        self._device["save_path"] = data_path
        self._device["save_moniker"] = conditions["sample_name"].value
        self.log("Parameters set Successfully!", level="ok")
        self.log("Raman parameters setting finished", indent="exit")

    def _spectrometer_run(self, delay_time: int = 1000, aggregate: bool = False):
        """
        Runs the raman_spec start acquisition, polls the data and returns it, in case of errors
        it also runs the stop acquisition unit task.
        :param delay_time: int the time between spectrometer polls
        :param aggregate: bool if true, the data is aggregated as a big DF
        """
        self.log("Starting raman acquisition process", indent="enter")
        self._device["start_acq"] = True
        self.log("Acquisition started!")
        data = self._spectrometer_poll(aggregate=aggregate)

        self.log("Acquisition finished", indent="exit")
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            self.log("No data was returned, not good", level="error", indent="exit")
            raise AnalysisError("No data was returned")

        return data

    def _spectrometer_poll(self, aggregate: bool = False):
        """
        Polls the raman data from the spectrometer
        :param delay_time: int the time between spectrometer polls
        :param aggregate: bool if true, the data is aggregated as a big DF
        """
        self.log("Polling raman data", indent="enter")
        data = self._device[
            "data"
        ]  # this is blocking, if aggregate we want to poll until no more data is returned
        if aggregate:
            while True:
                new_data = self._device["data"]
                if isinstance(new_data, str) and new_data == "done":
                    break
                # append the last column of the new data to the last column of the old data
                data = pd.concat([data, new_data.iloc[:, -1]], axis=1)

        self.log("Data polled successfully", indent="exit")
        return data

    def read_data(self):
        """
        Reads the data from a file, for now this is a stub
        """
        pass

    def _split_parameters(self, _parameters: dict[str, AnalyticalParameter]):
        """
        Splits the parameters in spectrometer and non spectrometer parameters
        :param all_parameters: dict[str, AnalyticalParameter]
        :return: dict[str, AnalyticalParameter], dict[str, AnalyticalParameter]
        """
        self.log("Splitting parameters", indent="enter")
        all_parameters = {}
        integration_parameters = {}
        lama_parameters = {}
        _lama_parameters = [
            param.name
            for param in (self._required_parameters + self._optional_parameters)
            if param.tag == "lamas"
        ]
        _integral_parameters = [
            param.name
            for param in (self._required_parameters + self._optional_parameters)
            if param.tag == "simple_integration"
        ]
        _all_parameters = [
            param.name
            for param in (self._required_parameters + self._optional_parameters)
            if param.tag == "all"
        ]

        for param_key, param_value in _parameters.items():
            if not isinstance(param_value, AnalyticalParameter):
                continue

            if param_key in _all_parameters:
                all_parameters[param_key] = param_value
            elif param_key in _integral_parameters:
                integration_parameters[param_key] = param_value
            elif param_key in _lama_parameters:
                lama_parameters[param_key] = param_value
            else:
                self.log(f"Parameter {param_key} has no tag, skipping", level="warning")

        self.log("Parameters split successfully!", indent="exit", level="ok")
        return {
            "all": all_parameters,
            "simple_integration": integration_parameters,
            "lamas": lama_parameters,
        }

    def _process_analytics(
        self,
        data: pd.DataFrame,
        recipe: list[RecipeComponent],
        non_spectrometer_parameters: dict[str, AnalyticalParameter],
        conditions: dict[
            str, ExperimentalParameter | NumericalParameter | AnalyticalParameter
        ],
    ):
        """
        Processes the data using the processing function defined in the parameters
        """
        self.log("Processing data", indent="enter")
        try:
            processing_function = get_function_from_globals(
                function_name=non_spectrometer_parameters["all"][
                    "processing_function"
                ].value,
                globals=globals(),
            )
        except Exception as e:
            self.log(
                f"Exception raised when getting global function: {e}",
                level="error",
                indent="reset",
            )
            raise e

        metrics = processing_function(
            data=data,
            recipe=recipe,
            non_spectrometer_parameters=non_spectrometer_parameters,
            conditions=conditions,
        )
        self.log("Data processed successfully", indent="exit")
        return metrics

    def _make_results(
        self, metrics: dict, recipe: List[AnalyticalParameter], conditions: dict
    ):
        """
        Processes metrics from the processing function and creates a results dictionary.

        :param metrics: dict, the dictionary output from the processing function
        :param recipe: dict, the recipe for the experiment
        :param conditions: dict, the conditions for the experiment
        :return: dict, the results dictionary containing yield, integral, integral_sideproduct_A, and pass
        """
        self.log("Making results", indent="enter")

        results = {}

        # Extract necessary values from metrics
        pi_conc = metrics.get("PI_conc", None)  # Product concentration
        sm_conc = metrics.get("SM_conc", None)  # Starting material concentration
        pi_area = metrics.get("PI_area", 0)  # Product integral (area)
        sm_area = metrics.get("SM_area", 0)  # Starting material integral (area)
        sm_var = metrics.get("SM_var", 0)  # Variance of the starting material
        pi_var = metrics.get("PI_var", 0)  # Variance of the product

        self.log(f"Metrics received: {metrics}", indent="enter")
        self.log(
            f"Extracted - PI_conc: {pi_conc}, SM_conc: {sm_conc}, PI_area: {pi_area}, SM_area: {sm_area}",
            indent="exit",
        )

        # If both concentrations are missing, set yield to None and continue
        if pi_conc is None or sm_conc is None:
            self.log("Concentrations missing, setting yield to None.", level="warning")
            results.update(
                {
                    "yield": None,
                    "yield_variance": None,
                    "integral": pi_area,
                    "integral_starting_material": sm_area,
                    "concentration_starting_material": sm_conc,
                    "concentration_starting_material_variance": sm_var,
                    "concentration_product": pi_conc,
                    "concentration_product_variance": pi_var,
                    "pass": True,  # No yield calculation, but reaction isn't necessarily failed
                }
            )
            return results
        # Reference concentration retrieval
        try:
            yield_calculation_chemical = conditions["yield_calculation_chemical"].value
            reference_concentration = self.get_reference_concentration(
                yield_calculation_chemical, recipe
            )
            self.log(
                f"Reference concentration for yield calculation: {reference_concentration}"
            )
        except Exception as e:
            self.log(f"Error getting reference concentration: {str(e)}", level="error")
            results["pass"] = False
            return results

        # Ensure reference concentration is valid
        if reference_concentration <= 0:
            self.log(
                "Invalid reference concentration, setting yield to None.", level="error"
            )
            results["yield"] = None
            results["conversion"] = None
            results["pass"] = False
            return results
        # Calculate yield heads up here we gonna do smth a bit dirty. we don't really calculate yield.
        yield_value = pi_conc
        yield_bottom = pi_conc - np.sqrt(pi_var)
        yield_top = pi_conc + np.sqrt(pi_var)
        yield_std_dev = np.abs(yield_top - yield_bottom)
        yield_variance = yield_std_dev**2

        conversion_value = -1*sm_conc
        conversion_bottom = (
           (-1*sm_conc - np.sqrt(sm_var))
        )
        conversion_top = (
           (-1*sm_conc + np.sqrt(sm_var))
        )
        conversion_std_dev = np.abs(conversion_top - conversion_bottom)
        conversion_variance = conversion_std_dev**2
        self.log(f"Calculated yield: {yield_value} ± {yield_std_dev}")
        self.log(f"Calculated conversion: {conversion_value} ± {conversion_std_dev}")
        # Handle yield out of bounds with standard deviation adjustment
        # if yield_value > 100:
        #     if yield_bottom <= 150:
        #         self.log(
        #             "Yield above 100 but within uncertainty range.",
        #             level="warning",
        #         )
        #     else:
        #         self.log(
        #             "Yield is unreasonably high, setting to None and marking as failed.",
        #             level="error",
        #         )
        #         yield_value = None
        #
        # elif yield_value < 0:
        #     if yield_top >= -50:
        #         self.log(
        #             "Yield below 0 but within uncertainty range.",
        #             level="warning",
        #         )
        #
        #     else:
        #         self.log(
        #             "Yield is unreasonably low, setting to None and marking as failed.",
        #             level="error",
        #         )
        #         yield_value = None
        #
        # if conversion_value > 100:
        #     if conversion_bottom <= 150:
        #         self.log(
        #             "Conversion above 100 but within uncertainty range.",
        #             level="warning",
        #         )
        #
        #     else:
        #         self.log(
        #             "Conversion is unreasonably high, setting to None and marking as failed.",
        #             level="error",
        #         )
        #         conversion_value = None
        #
        # elif conversion_value < 0:
        #     if conversion_top >= -50:
        #         self.log(
        #             "Conversion below 0 but within uncertainty range.",
        #             level="warning",
        #         )
        #
        #     else:
        #         self.log(
        #             "Conversion is unreasonably low, setting to None and marking as failed.",
        #             level="error",
        #         )
        #         conversion_value = None

        # Populate results
        results.update(
            {
                "yield": yield_value,
                "yield_variance": yield_variance,
                "conversion": conversion_value,
                "conversion_variance": conversion_variance,
                "integral": pi_area,
                "integral_starting_material": sm_area,
                "concentration_starting_material": sm_conc,
                "concentration_starting_material_variance": sm_var,
                "concentration_product": pi_conc,
                "concentration_product_variance": pi_var,
            }
        )

        # Pass criteria
        pass_criteria = {
            "yield_valid": yield_value is not None,
            "conversion_valid": conversion_value is not None,
            "area_positive": pi_area > 0
            or sm_area > 0,  # At least one integral must be positive
            "concentration_valid": pi_conc is None
            or (isinstance(pi_conc, (int, float)) and pi_conc + pi_var**0.5 >= 0),
            "reference_concentration_valid": reference_concentration > 0,
            "starting_material_concentration_valid": reference_concentration > 0
            and sm_conc is not None,
            "product_variance_not_excessive": (
                pi_conc is None or np.sqrt(pi_var) / reference_concentration < 0.50
            ),
        }
        results["pass"] = all(pass_criteria.values())

        # Log pass criteria evaluation
        self.log(f"Pass criteria evaluation: {pass_criteria}")
        self.log(
            f"Pass status: {'PASS' if results['pass'] else 'FAIL'}",
            level="ok" if results["pass"] else "warning",
        )

        self.log("Results successfully generated", indent="exit")
        return results


class dummyRamanAnalytics(AnalyticsUV):
    def analyse(
        self,
        conditions: dict[
            str, ExperimentalParameter | NumericalParameter | AnalyticalParameter
        ],
        recipe: list[RecipeComponent],
        process_only: bool = False,
    ) -> dict:
        pass
