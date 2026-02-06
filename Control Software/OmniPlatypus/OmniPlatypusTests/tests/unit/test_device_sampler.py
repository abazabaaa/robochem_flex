"""
File: test_device_sampler.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for LiquidHandlerSampler using mock interface.
"""

import unittest

import re
import pandas as pd
from time import sleep

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.sampler import Sampler
from omniplatypus.devices import testing
from omniplatypus.devices.errors import ParameterCommError
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    GenerateSampleDataframe,
    FindVial,
)


class MockCnc(Sampler):
    @property
    def serial_iface(self):
        return self._serial_iface

    def open(self, comport: str) -> None:
        dummy_serial_iface = testing.DummySerial()
        self._serial_iface = dummy_serial_iface
        self._serial_iface.open()

        dummy_serial_iface.read_delay = 0.1
        dummy_serial_iface.reply(
            "Grbl 0.9j ['$' for help]\r\n",
            "",
            "$0=10 (step pulse, usec)\r\n",
            "$1=25 (step idle delay, msec)\r\n",
            "$2=0 (step port invert mask:00000000)\r\n",
            "$3=6 (dir port invert mask:00000110)\r\n",
            "$4=0 (step enable invert, bool)\r\n",
            "$5=0 (limit pins invert, bool)\r\n",
            "$6=0 (probe pin invert, bool)\r\n",
            "$10=3 (status report mask:00000011)\r\n",
            "$11=0.020 (junction deviation, mm)\r\n",
            "$12=0.002 (arc tolerance, mm)\r\n",
            "$13=0 (report inches, bool)\r\n",
            "$20=0 (soft limits, bool)\r\n",
            "$21=0 (hard limits, bool)\r\n",
            "$22=0 (homing cycle, bool)\r\n",
            "$23=1 (homing dir invert mask:00000001)\r\n",
            "$24=50.000 (homing feed, mm/min)\r\n",
            "$25=635.000 (homing seek, mm/min)\r\n",
            "$26=250 (homing debounce, msec)\r\n",
            "$27=1.000 (homing pull-off, mm)\r\n",
            "$100=314.961 (x, step/mm)\r\n",
            "$101=314.961 (y, step/mm)\r\n",
            "$102=314.961 (z, step/mm)\r\n",
            "$110=10000.000 (x max rate, mm/min)\r\n",
            "$111=10000.000 (y max rate, mm/min)\r\n",
            "$112=10000.000 (z max rate, mm/min)\r\n",
            "$120=50.000 (x accel, mm/sec^2)\r\n",
            "$121=50.000 (y accel, mm/sec^2)\r\n",
            "$122=50.000 (z accel, mm/sec^2)\r\n",
            "$130=225.000 (x max travel, mm)\r\n",
            "$131=125.000 (y max travel, mm)\r\n",
            "$132=170.000 (z max travel, mm)\r\n",
            "ok\r\n",
            "",
            "ok\r\n",
            "",
            "ok\r\n",
            "",
            "ok\r\n",
        )
        welcome_msg = self.flush_buffer_in().replace("\n", "").replace("\r", "")
        if not welcome_msg == "":
            pattern = re.compile(r"^Grbl (?P<version>\d+\.\d+[a-zA-Z])")
            match = pattern.search(welcome_msg)
            if match:
                self.log(welcome_msg, level="ok")
                if not match.group("version") == "0.9j":
                    self.log(
                        "This software was tested on grbl v0.9j, but a different version was detected.",
                        level="warning",
                    )
            else:
                self.log(
                    f"Unexpected welcome message: '{welcome_msg}'", level="warning"
                )


class TestSampler(unittest.TestCase):
    def test_build(self):
        self.platform = Platform()
        self.platform.device_constructors["liquid_handler_sampler"] = MockCnc
        self.platform._known_devices = {"LH_cnc": "COM1"}
        try:
            self.platform.build(
                platform_name="Perry",
                devices=["Sampler_cnc"],
                update_docs=False,
                open_gui=True,
            )
        except ParameterCommError:
            pass
        self.cnc = self.platform["Sampler_cnc"]
        self.cnc: MockCnc
        self.dummy_serial_iface = self.cnc.serial_iface

        self.dummy_serial_iface.reply(
            "",
            "<Idle,MPos:0.000,0.000,0.000,WPos:0.000,0.000,0.000>\r\n",
            "",
            "ok\r\n",
            "",
            "<Idle,MPos:0.000,0.000,0.000,WPos:0.000,0.000,0.000>\r\n",
            "",
            "ok\r\n",
            "",
            "ok\r\n",
        )
        self.platform.clear()
        sleep(10)

    @unittest.skip("no")
    def test_write(self):
        self.dummy_serial_iface.reply("", "ok\r\n")
        self.cnc["feed"] = 10.0

        self.dummy_serial_iface.reply("", "ok\r\n")
        self.cnc["units_mm"] = "run"

    @unittest.skip("no")
    def test_read(self):
        # not needed, cache is used.
        # dummy_serial_iface.reply("", "1\r\n")
        d = self.cnc["soft_limits"]

    @unittest.skip("no")
    def test_dataframe(self):
        samples = pd.DataFrame(
            {
                "VialID": [
                    "N2",
                    "Waste_1",
                    "Waste_2",
                    "Waste_3",
                    "Sample_1",
                    "Sample_2",
                    "Sample_3",
                    "Sample_4",
                ],
                "Conc_A": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "Volume": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "Type": [
                    "Gas",
                    "Waste",
                    "Waste",
                    "Waste",
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
                "Sampler_pump": [
                    "Sampler_pump",
                    "Sampler_pump",
                    "Sampler_pump",
                    "Sampler_pump",
                    "Sampler_pump",
                    "Sampler_pump",
                    "Sampler_pump",
                    "Sampler_pump",
                ],
                "Holder": [
                    "holder_B",
                    "holder_F",
                    "holder_F",
                    "holder_F",
                    "holder_C",
                    "holder_C",
                    "holder_C",
                    "holder_C",
                ],
                "Position": [
                    "A1",
                    "A1",
                    "A2",
                    "A3",
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                ],
            }
        )
        samples.set_index("VialID", inplace=True, verify_integrity=True)
        samples = GenerateSampleDataframe.run(platform=self.platform, samples=samples)

        self.dummy_serial_iface.reply(
            "", "<Idle,MPos:0.000,0.000,0.000,WPos:0.000,0.000,0.000>\r\n", "", "ok\r\n"
        )
        waste = FindVial.run(
            platform=self.platform,
            samples=samples,
            sampler=self.cnc,
            vial_type="Waste",
            ensure_volume=5000.0,
        )
        self.assertEqual("Waste_1", waste)

        samples.loc["Waste_1", "Volume"] = 5500.0
        self.dummy_serial_iface.reply(
            "", "<Idle,MPos:0.000,0.000,0.000,WPos:0.000,0.000,0.000>\r\n", "", "ok\r\n"
        )
        waste = FindVial.run(
            platform=self.platform,
            samples=samples,
            sampler=self.cnc,
            vial_type="Waste",
            ensure_volume=5000.0,
        )
        self.assertEqual("Waste_2", waste)

        samples.loc[samples["Type"] == "Waste", "Volume"] = 5500.0
        waste = FindVial.run(
            platform=self.platform,
            samples=samples,
            sampler=self.cnc,
            vial_type="Waste",
            ensure_volume=5000.0,
        )
        self.assertTrue(waste is None)


if __name__ == "__main__":
    unittest.main()
