import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import numpy as np
import time

# for this we will assume that avaspec has been tested by the avaspec team so we can mock it


class Mock_avaspec:
    def __init__(self):
        self.initialise = MagicMock(return_value=None)
        self.AVS_Init = MagicMock(return_value=1)
        self.AVS_GetList = MagicMock(return_value=["R2-D2"])
        self.AVS_Activate = MagicMock(return_value=Mock_R2D2())
        self.AVS_Measure = MagicMock(return_value=None)
        self.AVS_PollScan = MagicMock(return_value=True)
        self.AVS_GetLambda = MagicMock(return_value=[1, 2, 3])
        self.AVS_GetScopeData = MagicMock(return_value=[0, [3, 3, 3]])
        self.AVS_PrepareMeasure = MagicMock(return_value=None)
        self.MeasConfigType = MagicMock(return_value=Mock_MeasConfigType())


class Mock_MeasConfigType(MagicMock):
    pass


class Mock_R2D2(MagicMock):
    pass


class TestAVAlanCHE(unittest.TestCase):
    """Test the AVAlanCHE class."""

    @patch("AVAlanCHE.AVAlanCHE.ava", new_callable=Mock_avaspec)
    def setUp(self, mock_ava):
        self.mock_ava = mock_ava

        from AVAlanCHE.AVAlanCHE_class import AVAlanCHE

        """Prepare resources for testing."""
        self.spectrometer = AVAlanCHE()

    # Mock wavelength

    def tearDown(self):
        """stop the patcher"""
        patch.stopall()

    def test_init(self):
        """Test the initialization of the class."""
        self.assertEqual(self.spectrometer.n_devices, 1)
        self.assertEqual(self.spectrometer.devices, ["R2-D2"])
        self.assertTrue(isinstance(self.spectrometer.spectrometer, Mock_R2D2))
        self.assertTrue(isinstance(self.spectrometer.meas_config, Mock_MeasConfigType))

    def test_set_parameters(self):
        """Test setting the parameters."""
        parameters = {
            "integration_time": 2000.0,
            "n_averages": 2,
            "n_scans": 100,
            "save_each_n": 10,
            "correct_dark": True,
            "save_as": "single",
            "save_path": "test/path",
        }

        self.spectrometer.set_parameters(parameters)

        self.assertEqual(
            self.spectrometer.current_parameters["integration_time"], 2000.0
        )
        self.assertEqual(self.spectrometer.current_parameters["n_averages"], 2)
        self.assertEqual(self.spectrometer.current_parameters["n_scans"], 100)
        self.assertEqual(self.spectrometer.current_parameters["save_each_n"], 10)
        self.assertEqual(self.spectrometer.current_parameters["correct_dark"], True)
        self.assertEqual(self.spectrometer.current_parameters["save_as"], "single")
        self.assertEqual(self.spectrometer.meas_config.m_IntegrationTime, 2000.0)
        self.assertEqual(self.spectrometer.meas_config.m_NrAverages, 2)

    @patch("AVAlanCHE.AVAlanCHE.ava", new_callable=Mock_avaspec)
    def test_take_spectrum(self, mock_ava):
        """Test that the spectrum is taken and saved."""
        # self.mock_ava.AVS_Measure.return_value = None
        # self.mock_ava.AVS_PollScan.return_value = True
        # self.mock_ava.AVS_GetLambda.return_value = [1, 2, 3]
        # self.mock_ava.AVS_GetScopeData.return_value = [0, [1, 2, 3]]
        self.spectrometer.current_dark = np.array([0, 0, 0])
        self.spectrometer.ready_to_measure = True

        self.spectrometer.take_spectrum()

        self.assertTrue(
            np.array_equal(self.spectrometer.wavelength, np.array([1, 2, 3]))
        )
        self.assertTrue(
            np.array_equal(self.spectrometer.current_spectrum, np.array([1, 2, 3]))
        )

    def test_setup_measurement(self):
        """Test the setup_measurement function."""
        self.mock_ava.AVS_PrepareMeasure.return_value = None
        self.spectrometer.setup_measurement()

        self.mock_ava.AVS_PrepareMeasure.assert_called_with(
            self.spectrometer.spectrometer, self.spectrometer.current_parameters
        )
        self.assertTrue(self.spectrometer.ready_to_measure)

    def test_get_dark(self):
        """Test that the dark spectrum is taken and saved."""
        self.mock_ava.AVS_Measure.return_value = None
        self.mock_ava.AVS_PollScan.return_value = True
        self.mock_ava.AVS_GetLambda.return_value = [1, 2, 3]
        self.mock_ava.AVS_GetScopeData.return_value = [0, [1, 2, 3]]
        self.spectrometer.ready_to_measure = True

        self.spectrometer.get_dark()

        self.assertTrue(hasattr(self.spectrometer, "current_dark"))
        self.assertTrue(
            np.array_equal(self.spectrometer.current_dark, np.array([1, 2, 3]))
        )

    def test_full_measurement_routine(
        self,
    ):
        """Test the full measurement routine."""
        # Setup mock responses and side effects
        self.mock_ava.AVS_Measure.return_value = None
        self.mock_ava.AVS_PollScan.side_effect = [False, True]  # Simulate some delay
        self.mock_ava.AVS_GetLambda.return_value = [1, 2, 3]
        self.mock_ava.AVS_GetScopeData.return_value = [0, [100, 200, 300]]
        self.spectrometer.current_dark = np.array([0, 0, 0])
        self.spectrometer.ready_to_measure = True

        # Run measurement
        self.spectrometer.setup_measurement()
        self.spectrometer.run_measurement()

        # Assertions to ensure the routine was executed correctly
        self.mock_ava.AVS_Measure.assert_called()
        self.assertEqual(
            self.mock_ava.AVS_PollScan.call_count, 2
        )  # Ensure PollScan was called twice due to the side effect
        self.assertTrue(
            np.array_equal(
                self.spectrometer.current_spectrum, np.array([100, 200, 300])
            )
        )
        # Add more assertions based on your specific requirements

    @patch("AVAlanCHE.AVAlanCHE.os")
    def test_saving(self, mock_os):
        """Test saving the data."""
        # Setup
        self.spectrometer.current_parameters["save_path"] = "test/path"
        self.spectrometer.name = "test_name"
        self.spectrometer.data_df = MagicMock()
        self.spectrometer.save_data()

        # Assertions
        mock_os.path.join.assert_called_with("test/path", "test_name.csv")
        self.spectrometer.data_df.to_csv.assert_called_once()

    def test_measurement_timing(self):
        """Test the timing of one loop of the measurement routine with a known delay."""
        known_delay = 0.1  # Known delay for AVS_PollScan in seconds
        self.mock_ava.AVS_Measure.return_value = None
        self.mock_ava.AVS_PollScan.side_effect = self._create_delay_side_effect(
            known_delay
        )
        self.mock_ava.AVS_GetScopeData.return_value = [0, [100, 200, 300]]
        self.spectrometer.ready_to_measure = True

        start_time = time.time()
        self.spectrometer.take_spectrum()
        end_time = time.time()

        elapsed_time = end_time - start_time

        # Check if the elapsed time is approximately equal to the known delay
        # Allow a small margin for execution of other instructions
        self.assertAlmostEqual(elapsed_time, known_delay, delta=0.05)

    def _create_delay_side_effect(self, delay):
        """Create a side effect function to simulate a delay in AVS_PollScan."""

        def side_effect(*args, **kwargs):
            time.sleep(delay)
            return True

        return side_effect


if __name__ == "__main__":
    unittest.main()
