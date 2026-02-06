"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from types import MethodType

from lamas.glama import Glama
from lamas.logger import Logger
from lamas.utils import extract_columns


class TestLoadData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="lamas", use_console=True)

    @classmethod
    def tearDownClass(cls):
        cls.logger.stop_logging_thread()

    def setUp(self):
        # Create a mock object for the class containing load_data
        self.glama = Glama(logger=self.logger)  # Replace with the actual class name
        self.glama._load_ramaberry = MagicMock(autospec=True)
        self.glama._load_lama_nmr = MagicMock(autospec=True)
        self.glama._load_lama_raman = MagicMock(autospec=True)
        self.glama._load_bruker = MagicMock(autospec=True)
        self.glama._load_spinsolve = MagicMock(autospec=True)
        self.glama._load_jcampdx = MagicMock(autospec=True)
        self.glama._validate_nmr_type = MagicMock(autospec=True)
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_load_data_from_dataframe(self):
        df = self.df.copy()
        self.glama.load_data(input_data=df, x_col="x", y_cols=["y"])
        self.assertEqual(self.glama.data.equals(df), True)

    def test_load_data_from_ramaberry(self):
        path = "data.csv"
        self.glama.data = self.df.copy()
        self.glama.load_data(input_data=path, from_filetype="ramaberry", x_col="x")
        self.glama._load_ramaberry.assert_called_once_with(path)
        self.assertEqual(self.glama.file_type, "ramaberry")

    def test_load_data_from_lama_nmr(self):
        path = "data.csv"
        self.glama.data = self.df.copy()
        self.glama.load_data(input_data=path, from_filetype="lama_nmr", x_col="x")
        self.glama._load_lama_nmr.assert_called_once_with(path)
        self.assertEqual(self.glama.file_type, "lama_nmr")

    def test_load_data_from_lama_raman(self):
        path = "data.csv"
        self.glama.data = self.df.copy()
        self.glama.load_data(input_data=path, from_filetype="lama_raman", x_col="x")
        self.glama._load_lama_raman.assert_called_once_with(path)
        self.assertEqual(self.glama.file_type, "lama_raman")

    def test_load_data_from_bruker(self):
        path = "data.csv"
        self.glama.data = self.df.copy()
        self.glama.load_data(input_data=path, from_filetype="bruker", x_col="x")
        self.glama._validate_nmr_type.assert_called_once()
        self.glama._load_bruker.assert_called_once_with(path)
        self.assertEqual(self.glama.file_type, "bruker")

    def test_load_data_from_spinsolve(self):
        path = "data.csv"
        self.glama.data = self.df.copy()
        self.glama.load_data(input_data=path, from_filetype="spinsolve", x_col="x")
        self.glama._validate_nmr_type.assert_called_once()
        self.glama._load_spinsolve.assert_called_once_with(path)
        self.assertEqual(self.glama.file_type, "spinsolve")

    def test_load_data_from_jdx(self):
        path = "data.jdx"
        self.glama.data = self.df.copy()
        self.glama.load_data(input_data=path, from_filetype="jdx", x_col="x")
        self.glama._validate_nmr_type.assert_called_once()
        self.glama._load_jcampdx.assert_called_once_with(path)
        self.assertEqual(self.glama.file_type, "jdx")

    def test_load_data_invalid_filetype(self):
        path = "data.csv"
        self.glama.data = self.df.copy()
        with self.assertRaises(ValueError):
            self.glama.load_data(input_data=path, from_filetype="invalid_type")

    def test_load_data_invalid_input_type(self):
        with self.assertRaises(TypeError):
            self.glama.load_data(input_data=12345)

    def test_data_sorted_by_x_col(self):
        # Testing if data is sorted by x_col
        df = pd.DataFrame({"x": [3, 1, 2], "y": [6, 4, 5]})
        self.glama.load_data(input_data=df, x_col="x", y_cols=["y"])
        self.assertTrue(self.glama.data["x"].is_monotonic_increasing)


class TestLoadMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logger = Logger
        cls.logger.start_logging_thread(platform="lamas", use_console=True)

    @classmethod
    def tearDownClass(cls):
        cls.logger.stop_logging_thread()

    def setUp(self):
        self.loader = Glama(logger=self.logger)
        self.loader.file_path = "dummy.csv"

    @patch("pandas.read_csv")
    def test_load_ramaberry(self, mock_read_csv):
        # Mock CSV data
        mock_read_csv.return_value = pd.DataFrame(
            {"R_shift": [100, 200, 300], "20231015_120000": [1.0, 1.5, 2.0]}
        )
        self.loader._load_ramaberry("dummy.csv")

        # Test assertions
        self.assertEqual(self.loader.x_col, "R_shift")
        self.assertEqual(self.loader.y_cols, ["Intensity"])
        self.assertEqual(self.loader.timestamp, "20231015_120000")
        pd.testing.assert_frame_equal(
            self.loader.data,
            pd.DataFrame({"R_shift": [100, 200, 300], "Intensity": [1.0, 1.5, 2.0]}),
        )

    @patch("pandas.read_csv")
    def test_load_lama_nmr(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame(
            {"PPM": [7.5, 8.0, 8.5], "20231015_120000": [0.5, 1.0, 1.5]}
        )
        self.loader._load_lama_nmr("dummy.csv")

        self.assertEqual(self.loader.x_col, "PPM")
        self.assertEqual(self.loader.y_cols, ["Intensity"])
        self.assertEqual(self.loader.timestamp, "20231015_120000")
        print(
            self.loader.data,
            pd.DataFrame({"PPM": [7.5, 8.0, 8.5], "Intensity": [0.5, 1.0, 1.5]}),
        )
        pd.testing.assert_frame_equal(
            self.loader.data,
            pd.DataFrame({"PPM": [7.5, 8.0, 8.5], "20231015_120000": [0.5, 1.0, 1.5]}),
        )

    @patch("pandas.read_csv")
    def test_load_lama_raman(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame(
            {"R_shift": [100, 200, 300], "20231015_120000": [0.2, 0.4, 0.6]}
        )
        self.loader._load_lama_raman("dummy.csv")

        self.assertEqual(self.loader.x_col, "R_shift")
        self.assertEqual(self.loader.y_cols, ["Intensity"])
        self.assertEqual(self.loader.timestamp, "20231015_120000")
        pd.testing.assert_frame_equal(
            self.loader.data,
            pd.DataFrame({"R_shift": [100, 200, 300], "Intensity": [0.2, 0.4, 0.6]}),
        )

    @patch("nmrglue.jcampdx.read")
    @patch("nmrglue.fileiobase.uc_from_udic")
    def test_load_jcampdx(self, mock_unit_conversion, mock_read):
        mock_read.return_value = (
            {"acqus": {"time": (2023, 10, 15, 12, 0, 0, 0, 0, 0)}},
            np.array([0.1, 0.2, 0.3]),
        )
        mock_unit_conversion.return_value.ppm_scale.return_value = np.array(
            [7.5, 8.0, 8.5]
        )

        self.loader._load_jcampdx("dummy.jdx")

        self.assertEqual(self.loader.x_col, "PPM")
        self.assertEqual(self.loader.y_cols, ["Intensity"])
        self.assertTrue(self.loader.timestamp.startswith("20231015_120000"))
        pd.testing.assert_frame_equal(
            self.loader.data,
            pd.DataFrame({"PPM": [7.5, 8.0, 8.5], "Intensity": [0.1, 0.2, 0.3]}),
        )

    def test_validate_nmr_type(self):
        self.loader.spec_type = "NMR"
        try:
            self.loader._validate_nmr_type()
        except ValueError:
            self.fail(
                "_validate_nmr_type() raised ValueError unexpectedly for NMR type."
            )

        self.loader.spec_type = "NonNMR"
        with self.assertRaises(ValueError):
            self.loader._validate_nmr_type()

    @patch("nmrglue.bruker.read")
    @patch("nmrglue.bruker.remove_digital_filter")
    @patch("nmrglue.proc_base.zf_size")
    @patch("nmrglue.proc_base.fft")
    def test_load_bruker(
        self, mock_fft, mock_zf_size, mock_remove_digital_filter, mock_read
    ):
        # Set up the mocked return values
        mock_read.return_value = (
            {"acqus": {"time": (2023, 10, 15, 12, 0, 0, 0, 0, 0)}},
            np.array([1.0, 2.0, 3.0]),
        )
        mock_remove_digital_filter.return_value = np.array([1.0, 2.0, 3.0])
        mock_zf_size.return_value = np.array([1.0, 2.0, 3.0])
        mock_fft.return_value = np.array([10.0, 20.0, 30.0])
        self.loader._load_bruker()

        self.assertEqual(self.loader.x_col, "PPM")
        self.assertEqual(self.loader.y_cols, ["Intensity"])
        self.assertTrue(self.loader.timestamp.startswith("20231015_120000"))

    @patch("nmrglue.spinsolve.read")
    @patch("nmrglue.proc_base.fft")
    @patch("nmrglue.spinsolve.guess_udic")
    def test_load_spinsolve(self, mock_udic, mock_fft, mock_read):
        # Mock spinsolve read and FFT output
        mock_read.return_value = (
            {"acqu": {"bandwidth": 7, "time": (2023, 10, 15, 12, 0, 0, 0, 0, 0)}},
            np.array([1.0, 2.0, 3.0]),
        )
        mock_fft.return_value = np.array([10.0, 20.0, 30.0])
        mock_udic.return_value = {
            "ndim": 1,
            0: {
                "size": 3,  # Number of data points in this dimension
                "complex": True,  # Indicates complex data
                "encoding": "direct",  # Encoding type
                "sw": 10000.0,  # Spectral width in Hz
                "obs": 400.13,  # Observation frequency in MHz
                "car": 4.7,  # Carrier frequency in ppm
                "label": "1H",  # Nucleus label
            },
        }
        self.loader._load_spinsolve()

        self.assertEqual(self.loader.x_col, "PPM")
        self.assertEqual(self.loader.y_cols, ["Intensity"])
        self.assertTrue(self.loader.timestamp.startswith("20231015_120000"))


class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.logger = Logger
        self.logger.start_logging_thread(platform="lamas", use_console=True)

    @classmethod
    def tearDownClass(self):
        self.logger.stop_logging_thread()

    def setUp(self):
        # Replace YourClass with the actual class name containing _apply_function and save_data
        self.processor = Glama(logger=self.logger)

    @patch("pandas.DataFrame.to_csv")
    def test_apply_function_multiple_y_cols(self, mock_to_csv):
        # Setup sample data
        self.processor.data = pd.DataFrame(
            {"x": [1, 2, 3], "y1": [4, 5, 6], "y2": [7, 8, 9]}
        )
        self.processor.y_cols = ["y1", "y2"]

        # Mock function to apply
        @extract_columns
        def mock_function_2(y, factor=1):
            return y * factor

        self.processor.mock_function = MethodType(mock_function_2, self.processor)

        # Apply function with an argument
        self.processor._apply_function(self.processor.mock_function, factor=2)

        # Verify that function applied correctly
        self.assertListEqual(self.processor.data["y1"].tolist(), [8, 10, 12])
        self.assertListEqual(self.processor.data["y2"].tolist(), [14, 16, 18])
        # pd.testing.assert_series_equal(self.processor.data["y1"], pd.Series([8, 10, 12]))
        # # pd.testing.assert_series_equal(self.processor.data["y2"], pd.Series([14, 16, 18]))

    def test_apply_function_single_y_col(self):
        # Setup sample data
        self.processor.data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        self.processor.y_cols = ["y"]

        # Mock function to apply
        @extract_columns
        def mock_function_1(y):
            return y + 10

        self.processor.mock_function = MethodType(mock_function_1, self.processor)

        # Apply function
        self.processor._apply_function(self.processor.mock_function)

        # Verify that function applied correctly
        # pd.testing.assert_series_equal(self.processor.data["y"], pd.Series([14, 15, 16]))
        self.assertListEqual(self.processor.data["y"].tolist(), [14, 15, 16])

    def test_apply_function_no_data(self):
        # Set data to None to simulate missing data
        self.processor.data = None
        with self.assertRaises(ValueError):
            self.processor._apply_function(lambda x: x * 2)

    @patch("pandas.DataFrame.to_csv")
    def test_save_data_with_output_path(self, mock_to_csv):
        # Setup sample data
        self.processor.data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        self.processor.x_col = "x"
        self.processor.timestamp = "20231015_120000"
        output_path = "/path/to/output.csv"

        # Call save_data with an explicit output_path
        self.processor.save_data(output_path=output_path)

        # Verify that to_csv was called with the correct output path
        mock_to_csv.assert_called_once_with(output_path, index=False)

    @patch("pandas.DataFrame.to_csv")
    def test_save_data_with_generated_path(self, mock_to_csv):
        # Setup sample data and data_parent_dict
        self.processor.data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        self.processor.x_col = "x"
        self.processor.timestamp = "20231015_120000"
        self.processor.spec_type = "test_type"
        self.processor.data_parent_dir = "/path/to"
        expected_output_path = "/path/to/20231015_120000_test_type.csv"

        # Call save_data without an output_path
        self.processor.save_data()

        # Verify that to_csv was called with the generated path
        mock_to_csv.assert_called_once_with(expected_output_path, index=False)

    def test_save_data_no_output_path_or_data_parent_dict(self):
        # Setup sample data without data_parent_dict or output_path
        self.processor.data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        self.processor.x_col = "x"
        self.processor.timestamp = "20231015_120000"
        self.processor.spec_type = "test_type"
        self.processor.data_parent_dir = None  # No data_parent_dict

        # Expect ValueError since no output path or data_parent_dict is provided
        with self.assertRaises(ValueError):
            self.processor.save_data()


if __name__ == "__main__":
    unittest.main()
