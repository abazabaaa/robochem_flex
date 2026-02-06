"""
File: test_device_liquid_handler_sampler.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: unit test for the liquid handler CNC sampler.
"""

import unittest
import pandas as pd

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.sampler import Sampler, GrblPosition, InjectionPort
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    GenerateSampleDataframe,
)


class LiquidHandlerSamplerTest(unittest.TestCase):
    device = None
    platform = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="Perry",
            devices=["Collector_cnc"],
        )
        cls.platform = test_platform
        cls.device = test_platform["Collector_cnc"]

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    @unittest.skip("Unnecessary")
    def test_move_x(self):
        destination = GrblPosition(x=50.0, y=0.0, z=0.0)
        self.device["feed"] = 500.0
        self.device["move"] = destination
        self.assertTrue(self.device["position"] == destination)
        destination = GrblPosition(x=0.0, y=0.0, z=0.0)
        self.device["move"] = destination
        self.assertTrue(self.device["position"] == destination)

    @unittest.skip("Unnecessary")
    def test_move_y(self):
        destination = GrblPosition(x=0.0, y=50.0, z=0.0)
        self.device["feed"] = 500.0
        self.device["move"] = destination
        self.assertTrue(self.device["position"] == destination)
        destination = GrblPosition(x=0.0, y=0.0, z=0.0)
        self.device["move"] = destination
        self.assertTrue(self.device["position"] == destination)

    @unittest.skip("Unnecessary")
    def test_move_z(self):
        destination = GrblPosition(x=0.0, y=0.0, z=-10.0)
        self.device["feed"] = 500.0
        self.device["move"] = destination
        self.assertTrue(self.device["position"] == destination)
        destination = GrblPosition(x=0.0, y=0.0, z=0.0)
        self.device["move"] = destination
        self.assertTrue(self.device["position"] == destination)

    def test_injection_ports(self):
        if input("Calibrate injection ports (y/N)?").lower() not in ("y", "yes"):
            return
        self.device: Sampler
        self.device["feed"] = 2000.0
        injection_ports = {
            name: location
            for name, location in self.device.locations.items()
            if isinstance(location, InjectionPort)
        }
        positions = {}
        for name, port in injection_ports.items():
            print(f"Aligning '{name}':")
            destination = self.device.locations["injection_waste"].move
            self.device["move"] = destination
            needle_down = GrblPosition(z=-33.0)
            needle_up = GrblPosition(z=-25.0)
            self.device["feed"] = 500
            position = ""
            while not position == "exit":
                self.device["move"] = needle_down
                print(self.device["position"])
                position = input("adjust?>")
                self.device["move"] = needle_up
                if position == "exit":
                    break
                position = position.split()
                coords = {}
                for p in position:
                    coords[p[0].lower()] = float(p[1:])
                destination.x += coords.get("x", 0.0)
                destination.y += coords.get("y", 0.0)
                self.device["move"] = destination
            prompt = input("Plunge (y/N)?")
            if prompt.lower() in ("y", "yes"):
                destination_2 = self.device.locations["injection_waste"].plunge
                while not prompt.lower() == "exit":
                    self.device["move"] = destination_2
                    prompt = input("adjust z (or exit)>")
                    try:
                        destination_2.z += float(prompt)
                    except ValueError:
                        pass
                destination.z = destination_2.z
            positions[name] = str(destination)
            self.device["feed"] = 2000.0
        print("Positions:")
        for name, position in positions.items():
            print(f"{name}: {position}")

    def test_holders(self):
        samples = pd.DataFrame(
            {
                "VialID": [
                    "HB_A1",
                    "HB_D4",
                    "HD_A1",
                    "HD_D4",
                    "HF_A1",
                    "HF_D4",
                ],
                "Sampler": [
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                    "Collector_cnc",
                ],
                "Holder": [
                    "holder_B",
                    "holder_B",
                    "holder_D",
                    "holder_D",
                    "holder_F",
                    "holder_F",
                ],
                "Position": [
                    "A1",
                    "F6",
                    "A1",
                    "F6",
                    "A1",
                    "F6",
                ],
                "Volume": [
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
                    "Stock",
                    "Stock",
                    "Stock",
                ],
            }
        )
        samples.set_index("VialID", inplace=True, verify_integrity=True)
        samples = GenerateSampleDataframe.run(platform=self.platform, samples=samples)
        needle_down = GrblPosition(z=-33.0)
        needle_up = GrblPosition(z=-25.0)
        self.device: Sampler
        self.device["feed"] = 500
        self.device["move"] = needle_up
        prompt = input("Calibrate holders (y/N)?")
        if prompt.lower() in ("y", "yes"):
            for vial_id, row in samples.iterrows():
                position = GrblPosition(x=row["X"], y=row["Y"])
                print(f"{vial_id} ({position.x}, {position.y})")
                self.device["feed"] = 2000.0
                self.device["move"] = position
                self.device["feed"] = 500
                self.device["move"] = needle_down
                user_prompt = input("proceed (<enter> to continue, exit to finish)?")
                self.device["move"] = needle_up
                if user_prompt.lower() == "exit":
                    break
        self.device["feed"] = 500
        self.device["move"] = needle_up


if __name__ == "__main__":
    unittest.main()
