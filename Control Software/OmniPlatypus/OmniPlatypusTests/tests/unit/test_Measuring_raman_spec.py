"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

TEST DESCR:
"""

import unittest
from unittest.mock import MagicMock, patch
from io import StringIO
import pandas as pd
from omniplatypus.procedures.unit_tasks.measuring.raman_spec import (
    set_raman_parameters,
    get_data,
)  # Adjust import as necessary


class TestRamanFunctions(unittest.TestCase):
    def setUp(self):
        # Mock the raman object
        self.raman = MagicMock()

    def test_set_raman_parameters_valid(self):
        parameters = {
            "integration_time": 100,
            "n_averages": 10,
            "delay": 5,
            "n_scans": 1,
            "return_as": "something",
            "dark_correction": True,
        }
        set_raman_parameters(self.raman, parameters)
        for param in parameters:
            self.raman.__setitem__.assert_any_call(param, parameters[param])

    def test_set_raman_parameters_invalid(self):
        parameters = {"invalid_param": 123}
        with patch("sys.stdout", new=StringIO()) as fake_out:
            set_raman_parameters(self.raman, parameters)
            self.assertIn("Parameter invalid_param not valid", fake_out.getvalue())

    @patch(
        "procedures.unit_tasks.measuring.raman_spec.time.sleep", return_value=None
    )  # Mock time.sleep
    def test_get_data(self, mocked_sleep):
        # Mocking data returned by the raman object
        self.raman.__getitem__.return_value = [
            {"Rshift": [1, 2], "Intensity": [3, 4]},
            "done",
        ]
        result = get_data(self.raman)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("Rshift" in result.columns and "Intensity" in result.columns)

    #
    # def test_start_acquisition(self):
    #     start_acquisition(self.raman)
    #     self.assertTrue(self.raman.__setitem__.called)
    #     self.assertEqual(self.raman.__setitem__.call_args[0], ("acq_data", True))
    #
    # def test_stop_acquisition(self):
    #     stop_acquisition(self.raman)
    #     self.assertTrue(self.raman.__setitem__.called)
    #     self.assertEqual(self.raman.__setitem__.call_args[0], ("stop_acq", True))


if __name__ == "__main__":
    unittest.main()
