"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from unittest.mock import patch
from unittest import TestCase
from lamas.alpaca import Alpaca
import matplotlib.pyplot as plt
from Tests.unit.test_sub_scripts import plotting_function as pf
import sys


class TestAlpaca(TestCase):
    """Test case for alpaca class, since a lot of stuff has been tested in te glama
    class we will just test the normalise method and the actual performance on data"""

    @classmethod
    def setUpClass(cls):
        """Set up the class and load the data"""
        cls.alpaca = Alpaca()
        cls.alpaca.load_data(
            "Tests/Test_data/raman_test.csv", from_filetype="ramaberry"
        )
        cls.fig = pf(
            cls.alpaca.data[cls.alpaca.x_col],
            cls.alpaca.data[cls.alpaca.y_cols[0]],
            label="Original Data",
        )

    def test_preprocess(self):
        """Test the denoise method"""
        with patch.object(
            self.alpaca, "denoise", wraps=self.alpaca.denoise
        ) as mock_denoise:
            self.alpaca.denoise(method="savgol")
            mock_denoise.assert_called_once()

        with patch.object(
            self.alpaca, "baseline", wraps=self.alpaca.baseline
        ) as mock_baseline:
            self.alpaca.baseline(method="als")
            mock_baseline.assert_called_once()

    def test_normalise(self):
        """Test the normalise method"""
        self.test_preprocess()

        with patch.object(
            self.alpaca, "normalise_to", wraps=self.alpaca.normalise_to
        ) as mock_normalise:
            self.alpaca.normalise_to(method="peak", range_vals=(500, 1200))
            mock_normalise.assert_called_once()

        pf(
            self.alpaca.data[self.alpaca.x_col],
            self.alpaca.data[self.alpaca.y_cols[0]],
            label="Normalised Data",
            fig=self.fig,
        )

        plt.show()

        # pas = input("Did the plot look good? (y/n): ")
        # self.assertEqual(pas, "y")
        # save = input("Do you want to save the cleaned data? (y/n): ")
        save = "y"
        if save == "y":
            self.alpaca.save_data("Tests/Test_data/raman_test_cleaned.csv")

        print("done")
        sys.exit()
