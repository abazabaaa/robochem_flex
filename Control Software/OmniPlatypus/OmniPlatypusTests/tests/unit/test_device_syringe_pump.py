"""
File: test_device_syringe_pump.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for SyringePumpNrg using mock interface.
"""

import unittest

from threading import Thread
from time import sleep

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.syringe_pump import SyringePump
from omniplatypus.devices import testing


class MockPump(SyringePump):
    @property
    def serial_iface(self):
        return self._serial_iface

    def open(self, comport: str) -> None:
        dummy_serial_iface = testing.DummySerial()
        self._serial_iface = dummy_serial_iface
        self._serial_iface.mute = False
        self._serial_iface.open()
        dummy_serial_iface.reply("", "", "0-0")
        dummy_serial_iface.reply("", "1")
        dummy_serial_iface.reply("", "0-0")
        dummy_serial_iface.reply("", "1")
        dummy_serial_iface.reply("", "0-0")
        dummy_serial_iface.reply("", "v\r\n")
        dummy_serial_iface.reply("", "0-0")
        dummy_serial_iface.reply("", "", "0-0")
        dummy_serial_iface.reply("", "", "0-0")
        dummy_serial_iface.reply("", "10.0\r\n")
        dummy_serial_iface.reply("", "0-0")
        dummy_serial_iface.reply("", "100.0\r\n")
        dummy_serial_iface.reply("", "0-0")
        dummy_serial_iface.reply("", "k\r\n")
        dummy_serial_iface.reply("", "0-0")
        dummy_serial_iface.reply("", "", "0-0")
        dummy_serial_iface.reply("", "", "0-0")


class TestPump(unittest.TestCase):
    def setUp(self):
        self.platform = Platform()
        self.platform.device_constructors["syringe_pump_nrg"] = MockPump
        self.platform._known_devices = {"main_pump_op1": "COM1"}
        self.platform.build(
            platform_name="Perry",
            devices=["Main_Pump_1"],
            update_docs=False,
            open_gui=False,
        )
        self.device = self.platform["Main_Pump_1"]
        self.device: MockPump
        self.dummy_serial_iface = self.device.serial_iface

    def tearDown(self):
        self.dummy_serial_iface.reply("", "k\r\n", "", "0-0\r\n")
        self.dummy_serial_iface.reply("", "", "0-0")
        self.platform.clear()

    def test_write(self):
        self.dummy_serial_iface.reply("", "", "0-0\r\n")
        self.device["diameter"] = 10.0

        self.dummy_serial_iface.reply("", "", "0-0\r\n")
        self.device["flowrate"] = 1.2345

    def test_read(self):
        self.dummy_serial_iface.reply("", "11.0\r\n")
        self.dummy_serial_iface.reply("", "0-0\r\n")
        d = self.device["diameter"]

    def test_pump(self):
        self.dummy_serial_iface.reply("", "1000.0\r\n")
        self.dummy_serial_iface.reply("", "0-0\r\n")
        self.dummy_serial_iface.reply("", "k\r\n")
        self.dummy_serial_iface.reply("", "0-0\r\n")
        self.device["pump"] = 100.0

    def pumping_task(self):
        self.dummy_serial_iface.reply("", "1000.0\r\n")
        self.dummy_serial_iface.reply("", "0-0\r\n")
        self.dummy_serial_iface.reply("", ("k\r\n", 2.0))
        self.dummy_serial_iface.reply("", "0-0\r\n")
        self.device["pump"] = 100.0

    def test_async_stop(self):
        self.device: MockPump
        pumping_thread = Thread(
            target=self.pumping_task, name="pumping thread", daemon=True
        )
        pumping_thread.start()
        sleep(0.1)
        self.device.stop_parameter("pump")
        pumping_thread.join()

    def test_async_stop_2(self):
        self.device: MockPump
        pumping_thread = Thread(
            target=self.pumping_task, name="pumping thread", daemon=True
        )
        pumping_thread.start()
        sleep(0.1)
        # self.device.stop_parameter("pump")
        pumping_thread.join()


if __name__ == "__main__":
    unittest.main()
