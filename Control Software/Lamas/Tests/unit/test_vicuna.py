"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from unittest import skip
from unittest import TestCase

from Tests.unit.test_sub_scripts import plotting_function as pf
from lamas.vicuna import Vicuna
import matplotlib.pyplot as plt


class TestVicuna(TestCase):
    """Testing case for vicuna, the goal here is to test how well
    the class is able to model a cleaned spectrum."""

    @classmethod
    def setUpClass(cls):
        """load cleaned data and start logger"""

        # cls.cuna = Vicuna(use_console = False)
        cls.cuna = Vicuna()
        cls.cuna.load_data(
            "Tests/Test_data/raman_test_cleaned.csv", from_filetype="lama_raman"
        )
        # cls.cuna.data = cls.cuna.data[
        #     (cls.cuna.data[cls.cuna.x_col] > 600)
        #     & (cls.cuna.data[cls.cuna.x_col] < 4000)
        # ]
        # cls.cuna.data.reset_index(drop=True, inplace=True)
        # x_vals = cls.cuna.data[cls.cuna.x_col].to_numpy()
        # y_vals = cls.cuna.data[cls.cuna.y_cols[0]].to_numpy()
        #
        # # Interpolate the data to have a more continuous spectrum
        # f = interp1d(x_vals, y_vals, kind="cubic")
        # x_vals = np.linspace(x_vals[0], x_vals[-1], 6000)
        # y_vals = f(x_vals)
        # cls.cuna.data = pd.DataFrame({cls.cuna.x_col: x_vals, cls.cuna.y_cols[0]: y_vals})
        #

        # cls.fig = pf(
        #     cls.cuna.data[cls.cuna.x_col],
        #     cls.cuna.data[cls.cuna.y_cols[0]],
        #     label="Original Data",
        # )

    @skip("Not ready yet")
    def test_generate_baseline(self):
        self.cuna._generate_baseline(lam=1e10, p=0.05, n_iter=10)
        pf(
            self.cuna.baseline[self.cuna.x_col],
            self.cuna.baseline[self.cuna.y_cols[0]],
            label="Baseline",
            fig=self.fig,
        )
        pf(
            self.cuna.data[self.cuna.x_col],
            self.cuna.data[self.cuna.y_cols[0]]
            - self.cuna.baseline[self.cuna.y_cols[0]],
            label="Baseline Removed",
            fig=self.fig,
        )
        plt.legend()
        plt.show()
        did_work = input("Did the baseline look good? (y/n): ")
        self.assertEqual(did_work, "y")

    @skip("Not ready yet")
    def test_generate_m_spec(self):
        self.cuna._generate_baseline(lam=1e9, p=0.05, n_iter=10)
        self.cuna._generate_model_spectrum()
        self.fig = pf(
            self.cuna.data[self.cuna.x_col],
            self.cuna.data[self.cuna.y_cols[0]],
            label="baselinerd Data",
        )
        pf(
            self.cuna.model_spectrum[self.cuna.x_col],
            self.cuna.model_spectrum[self.cuna.y_cols[0]],
            label="Model Spectrum",
            fig=self.fig,
        )
        plt.legend()
        plt.show()
        did_work = input("Did the model spectrum look good? (y/n): ")
        self.assertEqual(did_work, "y")

    def test_fit_loop(self):
        self.cuna.fit_all(lam=1e15, p=0.05, n_iter=10)
        self.fig = pf(
            self.cuna.data[self.cuna.x_col],
            self.cuna.data[self.cuna.y_cols[0]],
            label="baselinerd Data",
        )
        pf(
            self.cuna.model_spectrum[self.cuna.x_col],
            self.cuna.model_spectrum[self.cuna.y_cols[0]],
            label="Model Spectrum",
            fig=self.fig,
        )
        plt.legend()
        plt.show()
        did_work = input("Did the fitted spectrum look good? (y/n): ")
        self.assertEqual(did_work, "y")
        # pass
