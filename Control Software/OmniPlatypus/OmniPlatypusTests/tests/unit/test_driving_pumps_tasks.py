"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

TEST DESCR: test case for the driving_pumps unit tasks functions
"""

import unittest
from unittest.mock import patch, call
from omniplatypus.devices.nrg.syringe_pump import SyringePump

# functions to write tests for:
# fill_to, dispense_volume, driving_valve_position, refill, setup_pump, disable_pump, start_pump, drive_until_stopped_twin


class mock_pump(SyringePump):
    def __init__(self, min_value, max_value, volume_left):
        self.parameter_by_name = {"pump": mock_pump_parameter(min_value, max_value)}
        self.volume_left = volume_left
        self.valve_setpoint = "OFF"
        self.valve_actual = "OFF"
        self.volume = volume_left
        self.enable = 0
        self.acknowledge = 0
        self.max_syringe_volume = max_value
        self._max_syringe_volume = max_value
        self.min_syringe_volume = min_value
        self.flowrate = 0
        self.pump = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        if key == "pump":
            self.volume_left += -value
            self.volume += -value
            if self.volume > self.max_syringe_volume or self.volume < 0:
                raise ValueError("Cannot pump that much.")
        elif key == "zero":
            if value.upper() == "FILL":
                self.volume_left = self.max_syringe_volume
                self.volume = self.max_syringe_volume
            else:
                self.volume = 0.0
                self.volume_left = 0.0


class mock_pump_parameter:
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class TestDrivingPumps(unittest.TestCase):
    # add assertion here
    def setUp(self):
        self.driving_pump = mock_pump(min_value=0, max_value=100, volume_left=50)

    def test_fill_to_within_range(self):
        """tests the single filling function"""
        pump_volume(self.driving_pump)  # Need to make sure it starts from 0
        pump_volume(pump=self.driving_pump, volume=-70.0)
        self.assertEqual(int(self.driving_pump.volume_left), 70)

    def test_fill_wrong_volume(self):
        """tests the single filling function"""
        with self.assertRaises(ValueError):
            pump_volume(self.driving_pump, -120.0)

        with self.assertRaises(ValueError):
            pump_volume(self.driving_pump, -10.0)

        with self.assertRaises(ValueError):
            pump_volume(self.driving_pump, "a")

    def test_fill_all(self):
        """test complete refill"""
        fill_pump(self.driving_pump)
        self.assertEqual(
            int(self.driving_pump.volume_left),
            int(self.driving_pump.max_syringe_volume * 0.95),
        )

    def test_dispense_volume_within_range(self):
        """tests the dispense function"""
        pump_volume(self.driving_pump, 10.0)
        self.assertEqual(int(self.driving_pump.volume_left), int(40.0))

    def test_dispense_volume_out_of_range(self):
        """tests the dispense function"""
        with self.assertRaises(ValueError):
            pump_volume(self.driving_pump, volume=120.0)

        with self.assertRaises(ValueError):
            pump_volume(self.driving_pump, -10.0)

        with self.assertRaises(ValueError):
            pump_volume(self.driving_pump, "a")

    def test_dispense_all(self):
        """tests the dispense function"""
        pump_volume(self.driving_pump)
        self.assertEqual(int(self.driving_pump.volume_left), 0)

    def test_driving_valve_position(self):
        driving_valve_position(self.driving_pump, "Reservoir")
        self.assertEqual(self.driving_pump.valve_setpoint, "ON")

    def test_driving_valve_position_wrong_position(self):
        with self.assertRaises(ValueError):
            driving_valve_position(self.driving_pump, "Needle")

        with self.assertRaises(ValueError):
            driving_valve_position(self.driving_pump, "a")

    def test_setup_pumps(self):
        setup_pump(self.driving_pump, flowrate=10)
        self.assertEqual(self.driving_pump["flowrate"], 10)
        self.assertEqual(self.driving_pump["enable"], 1)
        self.assertEqual(self.driving_pump["ack_pump"], 0)

    def test_disable_pumps(self):
        disable_pumps(self.driving_pump)
        self.assertEqual(self.driving_pump["enable"], 0)
        self.assertEqual(self.driving_pump["ack_pump"], 1)

    def test_start_pump(self):
        initial_volume = start_pump(self.driving_pump)
        self.assertEqual(0, int(self.driving_pump.volume))
        self.assertEqual(int(self.driving_pump["pump"]), initial_volume)


@patch("procedures.unit_tasks.driving.driving_pumps.disable_pumps")
@patch("procedures.unit_tasks.driving.driving_pumps.setup_pump")
@patch("procedures.unit_tasks.driving.driving_pumps.start_pump")
class TestAsyncDriving(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.driving_pumpA = mock_pump(min_value=0, max_value=100, volume_left=50)
        self.driving_pumpB = mock_pump(min_value=0, max_value=100, volume_left=40)
        self.driving_pump_ensemble = [self.driving_pumpA, self.driving_pumpB]
        self.sensor_trigger = asyncio.Event()
        self.sensor_trigger.set()
        self.max_volume = 500
        self.rate = 50

    async def test_refill(self, *args):
        # Mocking the pump
        previous_flowrate = self.driving_pumpA["flowrate"]
        await refill(self.driving_pumpA)
        # assert refill has been called:
        self.assertTrue(self.driving_pumpA["ack_pump"] == 0)
        # assert the volume has been refilled:
        self.assertEqual(
            int(self.driving_pumpA.volume_left),
            int(self.driving_pumpA.max_syringe_volume * 0.95),
        )
        self.assertEqual(self.driving_pumpA["flowrate"], previous_flowrate)

    @patch("procedures.unit_tasks.driving.driving_pumps.refill")
    @patch("asyncio.sleep")
    async def test_pump_switch_on_low_volume(
        self, mock_sleep, mock_refill, mock_start, *args
    ):
        mock_refill.return_value = asyncio.Future()
        mock_sleep.side_effect = lambda x: self.sensor_trigger.set()
        mock_refill.return_value.set_result(True)
        mock_start.return_value = 80  # Initial volume for the pump
        self.driving_pumpA["volume"] = 20  # Simulate low volume in active pump

        await drive_until_stopped_twin(
            self.driving_pump_ensemble, self.rate, self.sensor_trigger, self.max_volume
        )

        mock_start.assert_has_calls(
            [call(self.driving_pumpB)]
        )  # Ensure pumps were started
        mock_refill.assert_called_with(self.driving_pumpA)

    @patch("procedures.unit_tasks.driving.driving_pumps.refill")
    @patch("asyncio.sleep")
    async def test_stop_pumping_when_sensor_triggered(
        self, mock_sleep, mock_refill, mock_start, mock_setup, mock_stop
    ):
        # mocking the sleep to set the trigger if it has been called 3 times:
        async def sleep_side_effect(x):
            if mock_sleep.call_count == 3:
                self.sensor_trigger.set()
            else:
                asyncio.sleep(0.1)

        mock_sleep.side_effect = sleep_side_effect
        mock_refill.return_value = asyncio.Future()
        mock_refill.return_value.set_result(True)
        mock_start.return_value = (
            100  # Initial volume for the pump  # Simulate sensor being triggered
        )
        self.sensor_trigger.clear()

        await drive_until_stopped_twin(
            self.driving_pump_ensemble, self.rate, self.sensor_trigger, self.max_volume
        )

        # check that the sensor has been triggered:
        self.assertTrue(self.sensor_trigger.is_set())
        # check that all the calls have been done:
        # mock_start.assert_has_calls(
        #     [call(self.driving_pumpA), call(self.driving_pumpB)]
        # )
        mock_stop.assert_has_calls([call(self.driving_pumpA), call(self.driving_pumpB)])

    @patch("procedures.unit_tasks.driving.driving_pumps.refill")
    @patch("asyncio.sleep")
    async def test_stop_pumping_when_sensor_triggered_single(
        self, mock_sleep, mock_refill, mock_start, mock_setup, mock_stop
    ):
        async def sleep_side_effect(x):
            if mock_sleep.call_count == 3:
                self.sensor_trigger.set()

        mock_sleep.side_effect = sleep_side_effect
        mock_refill.return_value = asyncio.Future()
        mock_refill.return_value.set_result(True)
        mock_start.return_value = 100  # Initial volume for the pump
        self.sensor_trigger.clear()

        await drive_until_stopped_single(
            self.driving_pumpA, self.rate, self.sensor_trigger, self.max_volume
        )

        self.assertTrue(self.sensor_trigger.is_set())
        mock_stop.assert_called_once_with(self.driving_pumpA)


# more tests to be added
# @patch('procedures.unit_tasks.driving.driving_pumps.refill')
# async def test_drive_until_stopped_twin(self, patch_refill):
#     # Mocking the pump
#     with patch('procedures.unit_tasks.driving.driving_pumps.refill') as mock_refill:
#         patch_refill.return_value = asyncio.Future()
#         patch_refill.return_value.set_result(True)
#
#         await drive_until_stopped_twin(self.driving_pump_ensemble, self.rate, self.sensor_trigger, self.max_volume)
#         mock_refill.assert_called()
#         self.assertTrue(self.driving_pump['acknowledge'] == 1)


if __name__ == "__main__":
    unittest.main()
