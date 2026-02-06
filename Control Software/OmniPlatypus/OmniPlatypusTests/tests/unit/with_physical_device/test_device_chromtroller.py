#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: unit test for the chromtroller device
"""
import unittest

import time

import pandas as pd

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.chromtroller import HPLCClient, HPLCProcessingSettings


class ChromtrollerTest(unittest.TestCase):
    device: HPLCClient = None
    platform = None

    @classmethod
    def setUpClass(cls):
        cls.platform = Platform()
        cls.platform.build(
            platform_name="dummy",
            devices=["UPLC"],
        )
        cls.device = cls.platform["UPLC"]

    @classmethod
    def tearDownClass(cls):
        cls.device.close()

    def test_initialization(self):
        self.assertEqual(self.device.generic_name, "Chromtroller HPLC Client")
        # Test that parameters are added correctly
        parameters = self.device._parameters
        self.assertIn("run_info", parameters)
        self.assertIn("analysis_parameters", parameters)
        self.assertIn("rt_target", parameters)
        self.assertIn("rt_tolerance", parameters)
        self.assertIn("valve_position", parameters)
        self.assertIn("acquisition", parameters)
        self.assertIn("analysis_result", parameters)

    def test_add_run_info(self):
        print("test")
        self.device["run_info"] = {
            "name": 0,
            "conc": 0.1,
            "reag_list": ["SM"],
            "condit_dict": {"A": 1, "B": 2},
        }

    def test_set_analysis_parameters(self):
        self.device["analysis_parameters"] = HPLCProcessingSettings()

    def test_set_analysis_target(self):
        self.device["rt_target"] = 2.5

    def test_set_analysis_tolerance(self):
        self.device["rt_tolerance"] = 0.1

    def test_valve_to_fill(self):
        self.device["valve_position"] = "fill"

    def test_valve_to_inject(self):
        self.device["valve_position"] = "inject"

    def test_start_acquisition(self):
        self.device["acquisition"] = "run"

    def test_get_result(self):
        result: dict = self.device["analysis_result"]["data"]
        result_df = pd.DataFrame(result)
        print(result_df)


# Run the tests
if __name__ == "__main__":
    unittest.main()
