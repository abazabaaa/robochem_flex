"""
File: test_device_spinsolve.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for spinsolve device using mock interface.
"""

import unittest

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.magritek.spinsolve import SpinsolveClient
from omniplatypus.procedures.analytics.analytics_parameters import AnalyticalParameter
from omniplatypus.procedures.analytics.nmr_analysis import NMRAnalysis


class TestSpinsolve(unittest.TestCase):
    def setUp(self):
        self.platform = Platform()
        self.platform.build(
            platform_name="Perry",
            devices=["NMR"],
            update_docs=False,
            open_gui=True,
        )
        # noinspection PyTypeChecker
        self.device: SpinsolveClient = self.platform["NMR"]

    def tearDown(self):
        self.platform.clear()

    def test_acquisition(self):
        analysis = NMRAnalysis(analytical_device=self.device)

        parameters = [
            AnalyticalParameter(name="sample_name", value="test_16"),
            AnalyticalParameter(name="protocol", value="1D FLUORINE HDEC"),
            AnalyticalParameter(name="Number", value="16"),
            AnalyticalParameter(name="RepetitionTime", value="15"),
            AnalyticalParameter(name="centerFrequency", value="-60"),
            AnalyticalParameter(name="target_peak", value=-61.72),
        ]

        print(analysis.analyse(parameters))


if __name__ == "__main__":
    unittest.main()
