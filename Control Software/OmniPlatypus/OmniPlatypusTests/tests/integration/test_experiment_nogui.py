"""
File: test_experiment_nogui.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for running experiments without the GUI.
"""

import copy
import unittest
import pandas as pd

from omniplatypus.procedures.experiments.experiment_parameters import (
    ExperimentalParameter,
    NumericalParameter,
    ChemicalParameter,
)
from omniplatypus.procedures.experiments.photochemistry import (
    PhotochemicalReaction,
    PhotochemicalReactionDryRun,
)


class ExperimentTest(unittest.TestCase):
    """
    Test for experiments without using the GUI.
    """

    experiment: PhotochemicalReaction | PhotochemicalReactionDryRun
    _samples: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        cls._samples = pd.DataFrame(
            {
                "VialID": [
                    "N2",
                    "waste",
                    "cleaning",
                    "mixing",
                    "SM",
                    "catalyst",
                    "reagent",
                    "solvent",
                    "premix",
                    "collect",
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                    "A5",
                    "A6",
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "C_A1",
                    "C_A2",
                    "C_A3",
                    "C_A4",
                    "C_A5",
                    "C_A6",
                    "C_B1",
                    "C_B2",
                    "C_B3",
                    "C_B4",
                    "C_B5",
                    "C_B6",
                    "C_C1",
                    "C_C2",
                    "C_C3",
                    "C_C4",
                    "C_C5",
                    "C_C6",
                ],
                "VialName": [
                    "N2",
                    "waste",
                    "cleaning",
                    "mixing",
                    "SM",
                    "catalyst",
                    "reagent",
                    "solvent",
                    "premix",
                    "collect",
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                    "A5",
                    "A6",
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "C_A1",
                    "C_A2",
                    "C_A3",
                    "C_A4",
                    "C_A5",
                    "C_A6",
                    "C_B1",
                    "C_B2",
                    "C_B3",
                    "C_B4",
                    "C_B5",
                    "C_B6",
                    "C_C1",
                    "C_C2",
                    "C_C3",
                    "C_C4",
                    "C_C5",
                    "C_C6",
                ],
                "Conc_SM": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1000.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "Conc_catalyst": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    17.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "Conc_reagent": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    8000.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "Conc_premix": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1000.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "Volume": [
                    0.0,
                    0.0,
                    10000.0,
                    0.0,
                    4000.0,
                    1500.0,
                    4000.0,
                    10000.0,
                    4000.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "Type": [
                    "Gas",
                    "Waste",
                    "Cleaning",
                    "Mixing",
                    "Stock",
                    "Stock",
                    "Stock",
                    "Solvent",
                    "Stock",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                ],
                "Sampler": [
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                ],
                "Holder": [
                    "holder_A",
                    "holder_F",
                    "holder_F",
                    "holder_A",
                    "holder_D",
                    "holder_E",
                    "holder_D",
                    "holder_F",
                    "holder_D",
                    "holder_B",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_E",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                    "holder_B",
                ],
                "Position": [
                    "D1",
                    "A1",
                    "B1",
                    "D2",
                    "A1",
                    "C1",
                    "A3",
                    "C1",
                    "B1",
                    "A4",
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                    "A5",
                    "A6",
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                    "A5",
                    "A6",
                    "B1",
                    "B2",
                    "B3",
                    "B4",
                    "B5",
                    "B6",
                    "C1",
                    "C2",
                    "C3",
                    "C4",
                    "C5",
                    "C6",
                ],
            }
        )
        cls._samples.set_index("VialID", inplace=True, verify_integrity=True)
        # todo this is to test vial updates
        # cls._samples.drop("N2", inplace=True)
        cls.experiment = PhotochemicalReaction()
        # cls.experiment = PhotochemicalReactionDryRun()

    @classmethod
    def tearDownClass(cls):
        cls.experiment.stop()

    def test_run(self):
        self.experiment.start("Perry", self._samples)
        run_base_id = "test"
        run_index = 0
        exiting = False
        while True:
            run_id = run_base_id + "_" + str(run_index)
            recipe = [
                NumericalParameter("residence_time", 30.0, "S"),
                NumericalParameter("light_intensity", 0.0, "%"),
                NumericalParameter("slug_volume", 500.0, "uL"),
                NumericalParameter("rinse", 1.0, "bool"),
                NumericalParameter("bubble_volume", 50.0, "uL"),
                NumericalParameter("intercomponent_bubble_volume", 0.2, "uL"),
                NumericalParameter("mix_before_injection", 6.0, "int"),
                NumericalParameter("mix_after_injection", 0.0, "int"),
                NumericalParameter("flowrate_sampling", 1.0, "mL/min"),
                ExperimentalParameter("collection_vial", ""),
                ExperimentalParameter("slug_preparation", "middle"),
                ExperimentalParameter("sampling", "direct"),
                ChemicalParameter("SM", 0.185, "M", sampling_priority=5000),
                ChemicalParameter("reagent", 18.0, "eq", sampling_priority=1000),
                ChemicalParameter("catalyst", 0.005, "eq", sampling_priority=3000),
            ]
            print("Run " + run_id + ": (type 'exit' in any field to stop)")

            while True:
                status = self.experiment.get_action_requests(block=False)
                if status is None:
                    break
                print(status)
                print("Add a vial:")
                index = input(" id:")
                vial_row = {}
                for col in self._samples.columns:
                    value = input(f"    {col}:")
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    vial_row[col] = value
                self._samples.loc[index] = vial_row
                status_resolution = UserAction(
                    status.error_id,
                    status.error_keyword,
                    "update samples",
                    self._samples,
                )
                self.experiment.submit_user_action(status_resolution)

            value = input("Use premix (y/n/exit, leave blank to refresh)?")
            if value.lower() == "exit":
                break
            if value == "":
                continue
            if value.lower() in ("y", "yes", "yeah", "sure"):
                recipe = [
                    NumericalParameter("residence_time", 30.0, "S"),
                    NumericalParameter("light_intensity", 0.0, "%"),
                    NumericalParameter("slug_volume", 500.0, "uL"),
                    NumericalParameter("rinse", 1.0, "bool"),
                    NumericalParameter("bubble_volume", 50.0, "uL"),
                    NumericalParameter("intercomponent_bubble_volume", 0.2, "uL"),
                    NumericalParameter("mix_before_injection", 6.0, "int"),
                    NumericalParameter("mix_after_injection", 0.0, "int"),
                    ExperimentalParameter("collection_vial", ""),
                    NumericalParameter("flowrate_sampling", 1.0, "mL/min"),
                    ExperimentalParameter("slug_preparation", "middle"),
                    ExperimentalParameter("sampling", "direct"),
                    ChemicalParameter("premix", 1.00, "M", sampling_priority=5000),
                ]

                value = input("Load from file?")
                if value.lower() in ("y", "yes", "yeah", "sure"):
                    runs_df = pd.read_csv(
                        r".\OmniPlatypusTests\tests\integration\TestRunsDF.csv"
                    )
                    _recipe = copy.deepcopy(recipe)
                    for index, row in runs_df.iterrows():
                        run_id = run_base_id + "_" + str(run_index)
                        for parameter in _recipe:
                            try:
                                parameter.value = row[parameter.name]
                            except (KeyError, ValueError, TypeError) as e:
                                print(f"Error parsing {parameter.name} (line {index}).")
                                print(e)
                                continue
                        self.experiment.submit_run(run_id, _recipe)
                        run_index += 1
                        print("submitted.")
                        for parameter in _recipe:
                            print(parameter)
                    continue

            for parameter in recipe:
                units = f"{parameter.units}" if hasattr(parameter, "units") else ""
                value = input(
                    f"Set {parameter.name} (default = {parameter.value}{units}):"
                )
                if value.lower() == "exit":
                    exiting = True
                    break
                if not value == "":
                    if parameter.name in (
                        "collection_vial",
                        "slug_preparation",
                        "sampling",
                    ):
                        parameter.value = value
                    else:
                        try:
                            parameter.value = float(value)
                        except (ValueError, TypeError):
                            print("invalid, using default.")
            if exiting:
                break
            print(f"Review run {run_id}:")
            for parameter in recipe:
                print(parameter)
            value = input("Confirm? (y/N/exit):")
            if value.lower() in ("y", "yes", "yeah", "sure"):
                self.experiment.submit_run(run_id, recipe)
                run_index += 1
                print("submitted.")
            else:
                if value.lower() == "exit":
                    break
                print("scrapped.")

        print("exiting, running experiments will complete...")
        self.experiment.stop()
