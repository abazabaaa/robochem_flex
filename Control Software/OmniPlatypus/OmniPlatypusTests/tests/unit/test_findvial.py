"""
File: test_findvial.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for FindVial unit task
"""

import unittest
import pandas as pd

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.sampler import Sampler, GrblPosition
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    DataFrameTask,
    FindVial,
)


class DummySampler(Sampler):
    def __getitem__(self, item):
        if item == "position":
            return GrblPosition(x=0.0, y=0.0, z=0.0)


class TestFindVial(unittest.TestCase):
    def setUp(self):
        self.samples = pd.DataFrame(
            {
                DataFrameTask.index_name: [
                    "N2",
                    "waste",
                    "solvent_1",
                    "solvent_2",
                    "sample_1",
                    "sample_2",
                    "sample_3",
                    "sample_4",
                ],
                DataFrameTask.vial_name: [
                    "N2",
                    "waste",
                    "solvent_1",
                    "solvent_2",
                    "sample_1",
                    "sample_2",
                    "sample_3",
                    "sample_4",
                ],
                "Volume": [
                    0.0,
                    0.0,
                    8000.0,
                    9000.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "Volume_min": [
                    500.0,
                    500.0,
                    1000.0,
                    1000.0,
                    500.0,
                    500.0,
                    100.0,
                    100.0,
                ],
                "Volume_max": [
                    4000.0,
                    4000.0,
                    10000.0,
                    10000.0,
                    4000.0,
                    4000.0,
                    2000.0,
                    2000.0,
                ],
                "Type": [
                    "Gas",
                    "Waste",
                    "Solvent",
                    "Solvent",
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
                ],
                "X": [
                    0.0,
                    10.0,
                    0.0,
                    10.0,
                    20.0,
                    30.0,
                    20.0,
                    30.0,
                ],
                "Y": [
                    0.0,
                    0.0,
                    10.0,
                    10.0,
                    20.0,
                    20.0,
                    30.0,
                    30.0,
                ],
                "Viable": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            }
        )
        self.samples.set_index(
            DataFrameTask.index_name, inplace=True, verify_integrity=True
        )
        self.sampler = DummySampler()
        self.platform = Platform()
        self.platform.samples = self.samples
        self.platform.devices = {"Sampler_cnc": self.sampler}

    def test_find_by_type(self):
        vial_id = FindVial.run(
            platform=self.platform, sampler=self.sampler, vial_type="Gas"
        )
        self.assertEqual(vial_id, "N2")
        vial_id = FindVial.run(
            platform=self.platform, sampler=self.sampler, vial_type="Waste"
        )
        self.assertEqual(vial_id, "waste")

    def test_find_by_distance(self):
        vial_id = FindVial.run(
            platform=self.platform, sampler=self.sampler, vial_type="Solvent"
        )
        self.assertEqual(vial_id, "solvent_1")

    def test_find_by_available_volume(self):
        vial_id = FindVial.run(
            platform=self.platform,
            sampler=self.sampler,
            vial_type="Solvent",
            ensure_volume=-8000.0,
        )
        self.assertEqual(vial_id, "solvent_2")

    def test_find_exclude_non_viable(self):
        self.samples.loc["solvent_1", "Viable"] = False
        vial_id = FindVial.run(
            platform=self.platform,
            sampler=self.sampler,
            vial_type="Solvent",
        )
        self.assertEqual(vial_id, "solvent_2")

    def test_find_by_nominal_volume(self):
        vial_id = FindVial.run(
            platform=self.platform,
            sampler=self.sampler,
            vial_type="Sample",
            nominal_volume=2000.0,
        )
        self.assertEqual(vial_id, "sample_3")


if __name__ == "__main__":
    unittest.main()
