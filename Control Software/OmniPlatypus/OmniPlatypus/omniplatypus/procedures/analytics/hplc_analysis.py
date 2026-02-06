"""
File: hplc_analysis.py
Author: Oliver Bayley, Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/ombayley, https://github.com/simone16

Description: Standardized analytical class for HPLC analysis.
"""

import time
from typing import Literal, get_origin, get_args
import numpy as np
import pandas as pd

from omniplatypus.utilities.general import dict_to_str
from omniplatypus.devices.nrg.chromtroller import HPLCProcessingSettings
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    RecipeComponent,
)
from omniplatypus.procedures.experiments.experiment_parameters import (
    ExperimentalParameter,
    NumericalParameter,
)
from omniplatypus.procedures.analytics.analytics_parameters import AnalyticalParameter
from omniplatypus.procedures.analytics.analytics_template import AnalyticsTemplate


class HPLCAnalysis(AnalyticsTemplate):
    result_metrics: list[str] = [
        "yield",
        "rt",
        "concentration",
        "integral",
        "yield_sideproduct_A",
        "rt_sideproduct_A",
        "concentration_sideproduct_A",
        "integral_sideproduct_A",
        "yield_sideproduct_B",
        "rt_sideproduct_B",
        "concentration_sideproduct_B",
        "integral_sideproduct_B",
        "pass",
        "info",
    ]

    _required_parameters: list[AnalyticalParameter] = [
        AnalyticalParameter(name="sample_name", value="test_1"),
        AnalyticalParameter(
            name="target_rt",
            value=0.0,
            units="min",
            tag="all",
        ),
        AnalyticalParameter(
            name="yield_calculation_chemical",
            value="",
            tag="all",
        ),
    ]

    _optional_parameters: list[AnalyticalParameter] = [
        AnalyticalParameter(
            name="sideproduct_A_rt",
            value=0.0,
            units="min",
            tag="all",
        ),
        AnalyticalParameter(
            name="sideproduct_B_rt",
            value=0.0,
            units="min",
            tag="all",
        ),
        AnalyticalParameter(
            name="sideproduct_C_rt",
            value=0.0,
            units="min",
            tag="all",
        ),
        AnalyticalParameter(
            name="sideproduct_D_rt",
            value=0.0,
            units="min",
            tag="all",
        ),
        AnalyticalParameter(
            name="max_peak_deviation",
            value=0.1,
            units="min",
            tag="all",
        ),
        AnalyticalParameter(
            name="target_peak_calibration_coeff_0",
            value=0.0,
            units="mM",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="target_peak_calibration_coeff_1",
            value=1000.00,
            units="mM/AU",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_A_peak_calibration_coeff_0",
            value=0.0,
            units="mM",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_A_peak_calibration_coeff_1",
            value=1000.00,
            units="mM/AU",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_B_peak_calibration_coeff_0",
            value=0.0,
            units="mM",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_B_peak_calibration_coeff_1",
            value=1000.00,
            units="mM/AU",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_C_peak_calibration_coeff_0",
            value=0.0,
            units="mM",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_C_peak_calibration_coeff_1",
            value=1000.00,
            units="mM/AU",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_D_peak_calibration_coeff_0",
            value=0.0,
            units="mM",
            tag="simple_integration",
        ),
        AnalyticalParameter(
            name="sideproduct_D_peak_calibration_coeff_1",
            value=1000.00,
            units="mM/AU",
            tag="simple_integration",
        ),
        *[
            AnalyticalParameter(
                name=attribute_name,
                value=HPLCProcessingSettings().__getattribute__(attribute_name),
                discrete_values=(
                    None
                    if get_origin(attribute_type) is not Literal
                    else get_args(attribute_type)
                ),
                units="",
                tag="all",
            )
            for attribute_name, attribute_type in HPLCProcessingSettings.__annotations__.items()
        ],
    ]

    _processing_methods: list[str] = ["simple_integration"]

    def sample_loading(self, enable=False) -> None:
        """
        Procedure to enable or disable sample loading, for those devices which implement such a system (e.g.: 6-way
        valve for sampling loop). If not implemented, leave this empty.

        @param enable: bool = False
            When set to true, should connect the analysis device to the flow system and allow the platform to load the
            sample. When set to False, the sampling loop is bypassed.
        """
        if enable:
            self._device["valve_position"] = "fill"
        else:
            self._device["valve_position"] = "inject"
        time.sleep(2.0)  # not sure this is needed, but let's be safe

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

        # Translate the analysis parameters to the HPLC settings.
        analysis_parameters = HPLCProcessingSettings()
        for attribute_name in HPLCProcessingSettings.__annotations__.keys():
            analysis_parameters.__setattr__(
                attribute_name, conditions[attribute_name].value
            )
        # Run info is for logging and tracking of samples.
        self._device.reopen()
        try:
            self._device["run_info"] = {"name": sample_name, "recipe": str(recipe)}
            # perform the actual analysis
            self._device["analysis_parameters"] = analysis_parameters
            self._device["acquisition"] = "run"
            raw_result: dict = self._device["analysis_result"]
            self.log("Received analysis data:\n" + dict_to_str(raw_result))
        finally:
            self._device.close()

        # create result dictionary
        result = {name: 0.0 for name in self.result_metrics}
        result["rt"] = np.nan
        result["rt_sideproduct_A"] = np.nan
        result["rt_sideproduct_B"] = np.nan
        result["rt_sideproduct_C"] = np.nan
        result["rt_sideproduct_D"] = np.nan
        result["pass"] = True

        # update result file name
        conditions["sample_name"].value = raw_result["file_name"]

        # find target concentration
        target_concentration = self.get_reference_concentration(
            reference_compound=all_parameters["yield_calculation_chemical"].value,
            recipe=recipe,
        )

        # Get peak data
        peaks: dict = raw_result["data"]
        if peaks is None:
            self.log(
                f"Analysis returned no peak data. Check Chromtroller log for errors.",
                level="error",
                indent="exit",
            )
            result["pass"] = False
            return result
        peaks: pd.DataFrame = pd.DataFrame.from_dict(peaks)

        # desired product
        best_peak = self.get_matching_peak(
            peaks,
            target_column="peak_rt",
            target=all_parameters["target_rt"].value,
            max_deviation=all_parameters["max_peak_deviation"].value,
            reference_spectrum="target",
            min_spectral_correlation=60.0,
        )

        if best_peak is not None:
            result.update(
                {
                    "integral": best_peak["integral"],
                    "rt": best_peak["peak_rt"],
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

        # sideproducts
        for sideproduct_id in ("A", "B", "C", "D"):
            best_peak = self.get_matching_peak(
                peaks,
                target_column="peak_rt",
                target=all_parameters[f"sideproduct_{sideproduct_id}_rt"].value,
                max_deviation=all_parameters["max_peak_deviation"].value,
                reference_spectrum=f"sideproduct_{sideproduct_id}",
                min_spectral_correlation=60.0,
            )

            if best_peak is not None:
                result.update(
                    {
                        f"integral_sideproduct_{sideproduct_id}": best_peak["integral"],
                        f"rt_sideproduct_{sideproduct_id}": best_peak["peak_rt"],
                    }
                )

                concentration = self.get_concentration(
                    best_peak["integral"],
                    coefficients=[
                        all_parameters[
                            f"sideproduct_{sideproduct_id}_peak_calibration_coeff_0"
                        ].value,
                        all_parameters[
                            f"sideproduct_{sideproduct_id}_peak_calibration_coeff_1"
                        ].value,
                    ],
                )
                result[f"concentration_sideproduct_{sideproduct_id}"] = concentration

                if target_concentration is not None:
                    chemical_yield = concentration / target_concentration * 100
                    result[f"yield_sideproduct_{sideproduct_id}"] = chemical_yield

                    chemical_conversion = 100 - chemical_yield
                    result[
                        f"conversion_sideproduct_{sideproduct_id}"
                    ] = chemical_conversion

        self.log(
            f"Processed {sample_name}:\n{dict_to_str(result)}.",
            level="ok",
            indent="exit",
        )
        return result


import random


class dummyHPLCAnalysis(HPLCAnalysis):
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
        return {
            "integral": random.randrange(100000000000),
            "integral_sideproduct_A": random.randrange(1000000000),
            "integral_sideproduct_B": random.randrange(10000000000),
        }
