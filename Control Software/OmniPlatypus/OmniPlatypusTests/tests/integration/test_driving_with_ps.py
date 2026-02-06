"""
File: test_driving_with_ps.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: unit test for the liquid handler CNC and pump (requires connection to the physical platform).
"""

import unittest

from time import sleep

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.syringe_pump import SyringePump
from omniplatypus.devices.nrg.sampler import Sampler
from omniplatypus.devices.nrg.gpio import SolenoidValve
from omniplatypus.devices.nrg.phase_sensor import PhaseSensor

from omniplatypus.procedures.unit_tasks.sensing.phase_sensors import (
    OpenValveUntilPhaseChange,
    PumpUntilPhaseChange,
)


class DrivingTest(unittest.TestCase):
    platform: Platform = None
    main_pump: SyringePump = None
    cnc: Sampler = None
    sampler_pump: SyringePump = None
    nitrogen: SolenoidValve = None
    phase_sensor: PhaseSensor = None

    # noinspection PyTypeChecker
    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="Perry",
            devices=[
                "Main_Pump_1",
                "Sampler_cnc",
                "Sampler_pump",
                "Gpio_Array_1",
                "Phase_Sensor_Array_1",
            ],
        )
        cls.platform = test_platform
        cls.main_pump = test_platform["Main_Pump_2"]
        cls.cnc = test_platform["Sampler_cnc"]
        cls.sampler_pump = test_platform["Sampler_pump"]
        cls.nitrogen = test_platform["sv_injection_n2"]
        cls.phase_sensor = test_platform["ps_handler_out"]

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    def test_builder(self):
        print(self.platform.keys())

    def test_ps(self):
        self.sampler_pump["aux_valve_setpoint"] = "OFF"
        self.nitrogen["valve"] = "OPEN"
        print("Opened valve")
        sleep(10.0)
        self.nitrogen["valve"] = "CLOSE"
        print("Closed valve")
        for i in range(10):
            print("Pumping...")
            PumpUntilPhaseChange.run(
                pump=self.main_pump,
                phase_sensor=self.phase_sensor,
                flowrate=1.0,
                max_volume=2000.0,
                timeout=20.0,
                wait_for_gas=False,
            )
            print("Valving...")
            OpenValveUntilPhaseChange.run(
                valve=self.nitrogen,
                phase_sensor=self.phase_sensor,
                timeout=20.0,
                wait_for_gas=True,
            )
        print("Done")


if __name__ == "__main__":
    unittest.main()
