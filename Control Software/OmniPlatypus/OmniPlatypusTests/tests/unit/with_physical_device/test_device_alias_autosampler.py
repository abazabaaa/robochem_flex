"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Test script for the autosampler class

"""

import unittest
from unittest import TestCase
from omniplatypus.devices.knauer.autosampler_alias import AutosamplerAlias
from omniplatypus.devices.platform import Platform
from omniplatypus.procedures.unit_tasks.sampling.alias_control import *


class TestAutosampler(TestCase):
    def setUp(self):
        self.port = "/dev/tty.usbserial-FTALDLQA"
        self.autosampler = AutosamplerAlias()

        self.autosampler.open(self.port)
        print("Autosampler opened")

    def test_zero(self):
        # self.autosampler.initialize()
        # self.autosampler['position'] = AliasPosition(special='HOME_ALL')
        # # let's see if it moves:
        # self.autosampler['position'] = AliasPosition(X=1, Y=1, T='L')
        # self.autosampler['position'] = AliasPosition(X=1, Y=1, T='R')
        # self.autosampler['position'] = AliasPosition(X=1, Y=1, T='L')
        #
        #
        # # move the needle up and down:
        # self.autosampler['needle_vertical_movement'] = 'DOWN'
        # self.autosampler['needle_vertical_movement'] = 'UP'
        #
        # # Home the needle:
        # self.autosampler['position'] = AliasPosition(special='HOME_ALL')

        # run an initial wash:
        self.autosampler["initial_wash"] = "START"

        # move the syringe:
        self.autosampler["move_syringe"] = "HOME"
        self.autosampler["move_syringe"] = "END"
        self.autosampler["move_syringe"] = "HOME"

        # try to aspirate and dispense 100 ul
        self.autosampler["aspirate"] = 100
        self.autosampler["dispense"] = 100


class TestRoutines(TestCase):
    def setUp(self):
        self.platform = Platform()
        self.platform.build(
            platform_name="Sparkie",
            devices=["Sampler_cnc"],
            open_gui=False,
        )

        self.sampler = self.platform["Sampler_cnc"]

        # build a df with a couple samples:
        self.df = pd.DataFrame(
            {
                "SampleID": [
                    "A1",
                    "A2",
                    "A3",
                    "S1",
                    "S2",
                    "B1",
                    "B2",
                ],
                "Sampler": [
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                ],
                "Holder": [
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                ],
                "Position": [
                    "A1",
                    "A2",
                    "A3",
                    "B1",
                    "B2",
                    "C1",
                    "C2",
                ],
                "Volume": [
                    1000.0,
                    1000.0,
                    1000.0,
                    1000.0,
                    1000.0,
                    1000.0,
                    1000.0,
                ],
                "Type": [
                    "Stock",
                    "Stock",
                    "Stock",
                    "Solvent",
                    "Solvent",
                    "Empty",
                    "Empty",
                ],
            }
        )
        # initialise all the unit tasks:
        print("ready to go")

    def test_1(self):
        self.sampler.initialize()


if __name__ == "__main__":
    # only run the first test case:
    unittest.main()
