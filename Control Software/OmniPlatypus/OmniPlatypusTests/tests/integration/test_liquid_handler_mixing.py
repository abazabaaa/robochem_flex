"""
File: test_liquid_handler_mixing.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for the sampling and mixing unit tasks.
"""

import unittest

import pandas as pd
from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.syringe_pump import SyringePump
from omniplatypus.devices.nrg.sampler import Sampler, GrblPosition
from omniplatypus.devices.nrg.gpio import SolenoidValve
from omniplatypus.devices.nrg.phase_sensor import PhaseSensor

from omniplatypus.procedures.unit_tasks.driving.driving_pumps import (
    FillPump,
    PrimePump,
    PumpVolume,
)
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    GenerateSampleDataframe,
    PumpSample,
)
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_injecting import Inject


class LiquidHandlerSamplerTest(unittest.TestCase):
    platform: Platform = None
    main_pump: SyringePump = None
    sampler: Sampler = None
    sampler_pump: SyringePump = None
    n2_valve: SolenoidValve = None
    phase_sensor: PhaseSensor = None
    samples: pd.DataFrame = None

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
        cls.main_pump = test_platform["Main_Pump_1"]
        cls.sampler = test_platform["Sampler_cnc"]
        cls.sampler_pump = test_platform["Sampler_pump"]
        cls.n2_valve = test_platform["sv_injection_n2"]
        cls.phase_sensor = test_platform["ps_handler_out"]

        cls.samples = pd.DataFrame(
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
                    "Cleaning",
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
                    "A2",
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
        cls.samples.set_index("VialID", inplace=True, verify_integrity=True)
        print("Before:")
        print(cls.samples.to_string())
        cls.samples = GenerateSampleDataframe.run(
            platform=cls.platform, samples=cls.samples
        )
        print("\nAfter:")
        print(cls.samples.to_string())

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    def test_sample_mixing(self):
        Inject.run(
            platform=self.platform,
            sampler=self.sampler,
            injection_port_name="injection_flow",
            volume=2000.0,
            flowrate=2.0,
            sampling_pump=self.sampler_pump,
        )
        FillPump.run(pump=self.sampler_pump, fill=False)
        volume = input("pump much?")
        while not volume == "exit":
            PumpSample.run(
                platform=self.platform,
                samples=self.samples,
                sample_id="N2",
                volume=-float(volume),
                needle_position="top",
                retract=False,
            )
            volume = input("pump much?")
        self.sampler["move"] = self.sampler.travel_position

    @unittest.skip("no need")
    def test_find_positions(self):
        msg = ""
        while not msg == "exit":
            msg = input("x y z >")
            pos = msg.split(" ")
            pos = GrblPosition(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            self.sampler["move"] = pos

    @unittest.skip("no need")
    def test_builder(self):
        print(self.platform.keys())
        holders = ["holder_F", "holder_E"]
        user = ""
        for holder in holders:
            samples = ["A1", "A2", "A3", "B1", "B2", "B3"]
            for sample_name in samples:
                coords = GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name=holder,
                    position_name=sample_name,
                )
                pos = GrblPosition(x=coords[0], y=coords[1])
                self.sampler["move"] = pos
                pos = GrblPosition(z=coords[2])
                self.sampler["move"] = pos
                user = input(f"{sample_name} >")
                if user == "exit":
                    break
                self.sampler["move"] = self.sampler.travel_position
            if user == "exit":
                break
        self.sampler["home"] = "RUN"

    def move_to(self, coords):
        self.sampler["move"] = self.sampler.travel_position
        pos = GrblPosition(x=coords[0], y=coords[1])
        self.sampler["move"] = pos
        pos = GrblPosition(z=coords[2])
        self.sampler["move"] = pos

    def do_op(self, vial, ops):
        solvent_holder = "holder_F"
        sample_holder = "holder_C"
        for op in ops:
            s_vial = op[0]
            vol = op[1]
            PumpVolume.run(pump=self.sampler_pump, volume=-100, flowrate=2.0)
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name=solvent_holder,
                    position_name=s_vial,
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume=-vol, flowrate=2.0)
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name=sample_holder,
                    position_name=vial,
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume=vol + 50, flowrate=2.0)
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name=solvent_holder,
                    position_name="A1",
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume=50, flowrate=2.0)
            self.sampler["move"] = self.sampler.travel_position

    @unittest.skip("no need")
    def test_bente(self):
        operations = {
            "A1": [
                ("C1", 500),
                ("C1", 500),
                ("C1", 500),
                ("C1", 500),
                ("C1", 500),
                ("C1", 500),
            ],
            "A2": [
                ("B3", 500),
                ("B3", 500),
                ("B3", 500),
                ("B3", 500),
                ("B3", 500),
                ("B3", 500),
            ],
            "A3": [
                ("C1", 500),
                ("C1", 500),
                ("C1", 500),
                ("C1", 500),
                ("C1", 100),
                ("C2", 500),
                ("C2", 400),
            ],
            "A4": [
                ("C3", 500),
                ("C3", 500),
                ("C3", 500),
                ("C3", 500),
                ("C3", 500),
                ("C3", 500),
            ],
        }
        PumpVolume.run(pump=self.sampler_pump, volume=1010, flowrate=2.0, reverse=True)
        PumpVolume.run(
            pump=self.sampler_pump,
            volume=-10,
            flowrate=2.0,
        )
        for sample, ops in operations.items():
            self.do_op(sample, ops)

        self.sampler["move"] = self.sampler.travel_position
        self.sampler["home"] = "RUN"

    @unittest.skip("already tested")
    def test_dispensing(self):
        # this all works fine
        user_input = input("Prime pumps? (y/n):")
        if user_input == "y":
            PrimePump.run(pump=self.sampler_pump, flowrate=5.0, cycles=2)
            PrimePump.run(pump=self.main_pump, flowrate=20.0, cycles=1)

        sample_data_a = {
            "vial": ["A1", "A2", "A3", "A4", "A5", "A6"],
            "holder": [
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
            ],
            "volume": [250, 100, 50, 25, 10, 5],
        }
        sample_data_a = pd.DataFrame(sample_data_a)
        sample_data_b = {
            "vial": ["B1", "B2", "B3", "B4", "B5", "B6"],
            "holder": [
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
            ],
            "volume": [250, 100, 50, 25, 10, 5],
        }
        sample_data_b = pd.DataFrame(sample_data_b)
        sample_data_c = {
            "vial": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "holder": [
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
            ],
            "volume": [250, 100, 50, 25, 10, 5],
        }
        sample_data_c = pd.DataFrame(sample_data_c)
        sample_data_d = {
            "vial": ["D1", "D2", "D3", "D4", "D5", "D6"],
            "holder": [
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
            ],
            "volume": [250, 100, 50, 25, 10, 5],
        }
        sample_data_d = pd.DataFrame(sample_data_d)
        sample_data = pd.concat(
            [sample_data_a, sample_data_b, sample_data_c, sample_data_d],
            ignore_index=True,
            sort=False,
        )
        df = pd.DataFrame(sample_data_a)

        self.move_to(
            GenerateSampleDataframe.vial_parameters(
                platform=self.platform,
                sampler_name="Sampler_cnc",
                holder_name="holder_F",
                position_name="A1",
            )
        )
        if user_input == "y":
            PumpVolume.run(pump=self.sampler_pump, volume="ALL", flowrate=5.0)
            FillPump.run(pump=self.sampler_pump, flowrate=8.0)
            PumpVolume.run(pump=self.sampler_pump, volume="ALL", flowrate=5.0)
            FillPump.run(pump=self.sampler_pump, flowrate=8.0)
            PumpVolume.run(pump=self.sampler_pump, volume=50.0, flowrate=1.0)
        else:
            FillPump.run(pump=self.sampler_pump, flowrate=8.0)
            PumpVolume.run(pump=self.sampler_pump, volume=50.0, flowrate=1.0)

        for i in range(len(df)):
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name=df.loc[i].holder,
                    position_name=df.loc[i].vial,
                )
            )
            # df.loc[i].volume
            PumpVolume.run(
                pump=self.sampler_pump, volume=df.loc[i].volume, flowrate=1.0
            )
        self.sampler["move"] = self.sampler.travel_position
        self.sampler["home"] = "RUN"

    @unittest.skip("already tested")
    def test_mixing(self):
        user_input = input("Prime pumps? (y/n):")
        if user_input == "y":
            PrimePump.run(pump=self.sampler_pump, flowrate=5.0, cycles=2)
            PrimePump.run(pump=self.main_pump, flowrate=20.0, cycles=1)
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name="holder_F",
                    position_name="A1",
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume="all", flowrate=5.0)
            FillPump.run(pump=self.sampler_pump, flowrate=5.0)
            PumpVolume.run(pump=self.sampler_pump, volume=900.0, flowrate=5.0)
        else:
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name="holder_F",
                    position_name="A1",
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume=900.0, flowrate=5.0)
        self.sampler["move"] = self.sampler.travel_position
        PumpVolume.run(pump=self.sampler_pump, volume=-100.0, flowrate=5.0)

        sample_data_a = {
            "vial": ["A1", "A2", "A3", "A4", "A5", "A6"],
            "holder": [
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
                "holder_E",
            ],
            "volume": [250, 100, 50, 25, 10, 5],
        }
        sample_data_a = pd.DataFrame(sample_data_a)

        for i in range(len(sample_data_a)):
            volume_analyte = sample_data_a.loc[i].volume

            # Take solvent
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name="holder_F",
                    position_name="A2",
                )
            )
            PumpVolume.run(
                pump=self.sampler_pump, volume=-500.0 + volume_analyte, flowrate=1.0
            )

            # take analyte
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name="holder_F",
                    position_name="A3",
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume=-volume_analyte, flowrate=1.0)

            # dump in vial
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name=sample_data_a.loc[i].holder,
                    position_name=sample_data_a.loc[i].vial,
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume=550.0, flowrate=1.0)

            # cleanup and prepare new
            self.move_to(
                GenerateSampleDataframe.vial_parameters(
                    platform=self.platform,
                    sampler_name="Sampler_cnc",
                    holder_name="holder_F",
                    position_name="A1",
                )
            )
            PumpVolume.run(pump=self.sampler_pump, volume="all", flowrate=5.0)
            PumpVolume.run(
                pump=self.sampler_pump,
                volume=-100.0,
                flowrate=5.0,
                reservoir_valve_position="ON",
            )
            self.sampler["move"] = self.sampler.travel_position
            PumpVolume.run(
                pump=self.sampler_pump,
                volume=-100.0,
                flowrate=5.0,
            )

    @unittest.skip("no")
    def test_inject(self):
        self.sampler_pump["enable"] = "ON"
        prime_pump(pump=self.sampler_pump, flowrate=2.0, cycles=1)
        prepare_injection(
            main_pump=self.main_pump,
            sampler=self.sampler,
            sampler_pump=self.sampler_pump,
            gas_valve=self.n2_valve,
            sampler_phase_sensor=self.phase_sensor,
        )
        self.sampler_pump["pump"] = 200.0
        self.sampler_pump["enable"] = "OFF"
        complete_injection(sampler=self.sampler, sampler_pump=self.sampler_pump)
        self.main_pump["flowrate"] = 2.0
        self.main_pump["enable"] = "on"
        self.main_pump["pump"] = 500.0
        self.main_pump["enable"] = "off"

    @unittest.skip("no")
    def test_async_pumping(self):
        self.sampler_pump["aux_valve_setpoint"] = "OFF"
        self.n2_valve["valve"] = "OPEN"
        message = input("Close valve?")
        print()
        self.n2_valve["valve"] = "CLOSE"
        pump_until_phase_detection(
            pump=self.main_pump,
            phase_sensor=self.phase_sensor,
            flowrate=1.0,
            timeout=120.0,
            max_volume=2000.0,
        )

    @unittest.skip("no")
    def adjust_flow(self):
        message = "y"
        self.sampler_pump["aux_valve_setpoint"] = "OFF"
        self.main_pump["enable"] = "on"
        fill_pump(self.main_pump, 10.0)
        self.main_pump["pump"] = 100.0
        while message == "y":
            open_valve_until_phase_detection(
                valve=self.n2_valve,
                phase_sensor=self.phase_sensor,
                timeout=60.0,
            )
            self.main_pump["flowrate"] = 2.0
            self.main_pump["pump"] = 500.0
            message = input("continue?")
        self.main_pump["enable"] = "off"


if __name__ == "__main__":
    unittest.main()
