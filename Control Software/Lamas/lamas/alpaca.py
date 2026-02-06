"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:
Alpacas are a subspecies of Llamas, and they are known for their grace and tidy appearence. This is a class that
aims at tidying up the raw Raman data by performing the preprocessing necessary to make the data ready for analysis.


"""

from lamas.glama import Glama
from lamas.utils import initkwargs, LamaParameter, LamaMethod
from typing import Tuple, Literal
import numpy as np


class Alpaca(Glama):
    """
    class handles data preprocessing for both raman, and NMR spectra
    (support for other types of spectra is planned)
    """

    structure: dict = {
        "normalise_to": LamaMethod(
            method_name="normalise_to",
            parameters=[
                LamaParameter(
                    name="to",
                    type=float,
                    default=1000.0,
                    description="value to normalise to (default 1000)",
                    required=False,
                ),
                LamaParameter(
                    name="peak",
                    type=float,
                    default=None,
                    description="peak value to normalise against",
                    required=False,
                ),
                LamaParameter(
                    name="range_vals",
                    type=Tuple[float, float],
                    default=None,
                    description="range of values to normalise against",
                    required=False,
                ),
                LamaParameter(
                    name="method",
                    type=Literal["area", "peak"],
                    default="area",
                    description="method to use for normalisation (default 'area')",
                    required=False,
                ),
            ],
        ),
        "baseline": LamaMethod(
            method_name="baseline",
            parameters=[
                LamaParameter(
                    name="method",
                    type=Literal[
                        "cholesky",
                        "linear_fit",
                        "als",
                        "rubber_band",
                        "polyfit",
                        "wavelet",
                    ],
                    description="method to use for baseline correction (default 'cholesky')",
                    required=False,
                )
            ],
            submethods=[
                LamaMethod(
                    method_name="cholesky",
                    parameters=[
                        LamaParameter(
                            name="p",
                            type=float,
                            default=0.01,
                            description="weight of peak term",
                        ),
                        LamaParameter(
                            name="lam",
                            type=float,
                            default=10**2,
                            description="weight of smoothness term",
                        ),
                        LamaParameter(
                            name="niter",
                            type=int,
                            default=100,
                            description="number of iterations",
                        ),
                    ],
                    other_parameters={"method": "cholesky"},
                ),
                LamaMethod(
                    method_name="linear_fit",
                    parameters=[
                        LamaParameter(
                            name="flat_range",
                            type=Tuple[float, float],
                            default=(None, None),
                            description="range of values to use for the linear fit",
                        )
                    ],
                    other_parameters={"method": "linear_fit"},
                ),
                LamaMethod(
                    method_name="als",
                    parameters=[
                        LamaParameter(
                            name="lam",
                            type=float,
                            default=10**6,
                            description="weight of the smoothness term",
                        ),
                        LamaParameter(
                            name="p",
                            type=float,
                            default=0.01,
                            description="weight of the asymmetry term",
                        ),
                        LamaParameter(
                            name="niter",
                            type=int,
                            default=100,
                            description="number of iterations",
                        ),
                    ],
                    other_parameters={"method": "als"},
                ),
                LamaMethod(
                    method_name="rubberband",
                    parameters=[
                        LamaParameter(
                            name="window",
                            type=int,
                            default=50,
                            description="window size, in points",
                        )
                    ],
                    other_parameters={"method": "rubberband"},
                ),
                LamaMethod(
                    method_name="polyfit",
                    parameters=[
                        LamaParameter(
                            name="order",
                            type=int,
                            default=3,
                            description="order of the polynomial to fit",
                        )
                    ],
                    other_parameters={"method": "polyfit"},
                ),
                LamaMethod(
                    method_name="wavelet",
                    parameters=[
                        LamaParameter(
                            name="wavelet",
                            type=str,
                            default="db4",
                            description="wavelet to use",
                        ),
                        LamaParameter(
                            name="level",
                            type=int,
                            default=1,
                            description="level of wavelet transform",
                        ),
                    ],
                    other_parameters={"method": "wavelet"},
                ),
            ],
        ),
        "denoise": LamaMethod(
            method_name="denoise",
            parameters=[
                LamaParameter(
                    name="method",
                    type=Literal[
                        "fft", "moving_average", "wavelet", "median", "savgol"
                    ],
                    description="method to use for noise reduction (default 'fft')",
                    required=False,
                )
            ],
            submethods=[
                LamaMethod(
                    method_name="fft",
                    parameters=[
                        LamaParameter(
                            name="threshold",
                            type=float,
                            default=0.1,
                            description="fraction of frequencies to retain (lower value means more aggressive denoising)",
                        )
                    ],
                    other_parameters={"method": "fft"},
                ),
                LamaMethod(
                    method_name="moving_average",
                    parameters=[
                        LamaParameter(
                            name="window_size",
                            type=int,
                            default=5,
                            description="size of the moving window for averaging",
                        )
                    ],
                    other_parameters={"method": "moving_average"},
                ),
                LamaMethod(
                    method_name="wavelet",
                    parameters=[
                        LamaParameter(
                            name="wavelet",
                            type=str,
                            default="db1",
                            description="type of wavelet to use",
                        ),
                        LamaParameter(
                            name="level",
                            type=int,
                            default=1,
                            description="decomposition level for wavelet transform",
                        ),
                        LamaParameter(
                            name="threshold",
                            type=float,
                            default=0.2,
                            description="threshold for coefficient filtering",
                        ),
                    ],
                    other_parameters={"method": "wavelet"},
                ),
                LamaMethod(
                    method_name="median",
                    parameters=[
                        LamaParameter(
                            name="window_size",
                            type=int,
                            default=3,
                            description="size of the filter window for median calculation",
                        )
                    ],
                    other_parameters={"method": "median"},
                ),
                LamaMethod(
                    method_name="savgol",
                    parameters=[
                        LamaParameter(
                            name="window_size",
                            type=float,
                            default=0.05,
                            description="window size as fraction of the data length",
                        ),
                        LamaParameter(
                            name="order",
                            type=int,
                            default=3,
                            description="order of the polynomial to fit",
                        ),
                    ],
                    other_parameters={"method": "savgol"},
                ),
            ],
        ),
    }

    raman_required_kwargs = [
        initkwargs(
            "ex_wlen",
            "excitation wavelength of the laser used to generate the data (used to calculate shift)",
            "nm",
            785,
        ),
        initkwargs(
            "norm_range",
            "normalisation range (chosen solvent peak)",
            "cm^-1",
            [1000, 1020],
        ),
    ]

    nmr_required_kwargs = [
        initkwargs(
            "solvent",
            "solvent used in the NMR experiment (available values are 'CDCl3','DMSO', 'D2O', 'CD3OD' or non deuterated counterparts)",
            "string",
            "CDCl3",
        ),
        initkwargs("file_type", "type of file to load", "string", "bruker"),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = "alpaca"
        self._description = "A class that preprocesses Raman and NMR data"
        self.data = None
        self.x_col = None
        self.y_cols = None

    def normalise_to(
        self,
        to: float = 1000,
        peak: float = None,
        range_vals: Tuple[float, float] = None,
        method: Literal["area", "peak"] = "area",
    ) -> None:
        """
        Normalises the data in self.data to a specific value.
        method is area: integrates by calculating the area under the curve.
        method is peak: integrates by calculating the peak.

        :param to: float: value to normalise to (default 1000)
        :param peak: float: peak value to normalise against
        :param range_vals: Tuple[float,float]: range of values to normalise against
        :param method: str: method to use for normalisation (default 'area')

        returns None

        if peak and range_vals are not provided it will use the max value in the data and find
        the range of values to normalise against
        peak and range_vals are mutually exclusive and if both are provided range_vals will be used
        """
        # first identify the method;
        match method:
            case "area":
                self._normalise_area(to=to, peak=peak, range_vals=range_vals)
            case "peak":
                self._normalise_peak(to=to, peak=peak, range_vals=range_vals)
            case _:
                raise ValueError("method must be either 'area' or 'peak'")

    def _normalise_area(
        self,
        to: float = 1000,
        peak: float = None,
        range_vals: Tuple[float, float] = None,
    ) -> None:
        """
        integrates by calculating the area under the curve. Normalises the data in self.data to a specific value.

        @param to: float: value to normalise to (default 1000)
        @param peak: float: peak value to normalise against (default None), will find the peak shoulders automatically
        @param range_vals: Tuple[float,float]: range of values to normalise against (default None)

        if both peak and range_vals are provided range_vals will be used

        returns None
        """
        if range_vals != None:
            if len(self.y_cols) > 1:
                for y in self.y_cols:
                    int_val = self._integrate_basic(
                        y=self.data[y].to_numpy(), bounds=range_vals
                    )
                    self.data[y] = self.data[y] * (to / int_val)
            elif len(self.y_cols) == 1:
                int_val = self._integrate_basic(bounds=range_vals)
                self.data[self.y_cols] = self.data[self.y_cols] * (to / int_val)
            else:
                raise ValueError("No data to process.")
        elif peak != None:
            # find the peak position:
            peak_pos = np.argmin(self.data[self.x_col].to_numpy() - peak)
            # make sure it's a np.array
            peak_pos = np.array([peak_pos])
            if len(self.y_cols) > 1:
                for y in self.y_cols:
                    int_val, _ = self._integrate_peaks(
                        peak_positions=peak_pos, y=self.data[y].to_numpy()
                    )
                    self.data[y] = self.data[y] * (to / int_val)
            elif len(self.y_cols) == 1:
                int_val, _ = self._integrate_peaks(peak_positions=peak_pos)
                self.data[self.y_cols] = self.data[self.y_cols] * (to / int_val)
            else:
                raise ValueError("No data to process.")
        else:
            raise ValueError("either peak or range_vals must be provided")

    def _normalise_peak(
        self,
        to: float = 1000,
        peak: float = None,
        range_vals: Tuple[float, float] = None,
    ) -> None:
        """
        integrates by calculating the peak. Normalises the data in self.data to a specific value.

        @param to: float: value to normalise to (default 1000)
        @param peak: float: peak value to normalise against (default None)
        @param range_vals: Tuple[float,float]: range of values to normalise against (default None), it will find a max value in the data

        if both peak and range_vals are provided peak will be used

        returns None
        """
        if peak is not None:
            # Find the y value at the specified peak position
            peak_pos = np.argmin(self.data[self.x_col].to_numpy() - peak)

            for y_col in self.y_cols:
                peak_val = self.data[y_col].iloc[peak_pos]
                self.data[y_col] = self.data[y_col] * (to / peak_val)
        elif range_vals is not None:
            # Find the maximum value within the range for each y column
            for y_col in self.y_cols:
                max_val = (
                    self.data[y_col]
                    .loc[
                        (self.data[self.x_col] > range_vals[0])
                        & (self.data[self.x_col] < range_vals[1])
                    ]
                    .max()
                )
                self.data[y_col] = self.data[y_col] * (to / max_val)
        else:
            raise ValueError("Either peak or range_vals must be provided")

    def baseline(
        self,
        method: Literal[
            "cholesky", "linear_fit", "als", "rubber_band", "polyfit", "wavelet"
        ] = "cholesky",
        **kwargs,
    ) -> None:
        """
        Baselines the data in self.data.

        :param method: str: method to use for baseline correction (default "cholesky",
        also available:linear_fit, poly_fit, rubberband, als, wavelet)
            kwargs wiki
            cholesky: uses cholesky decomposition to baseline correct the data
                    :param p: float (default 0.01): weight of peak term
                    :param lam: float (default 10**2): weight of smoothness term
                    :param niter: int (default 100): number of iterations
            linear_fit: uses a linear fit to baseline correct the data
                    :flat_range: Tuple[float,float] (default (None,None)): range of values to use for the linear fit
            als: uses the asymmetric least squares method to baseline correct the data
                    :param lam: float(default 10**6): weight of the smoothness term
                    :param p: float(default 0.01): weight of the asymmetry term
                    :param niter: int(default 100): number of iterations
            rubberband: uses the rubberband method to baseline correct the data
                    :param window: int(default 50): window size, in points
            polyfit: uses polynomial fit to baseline correct the data
                    :param order: int(default 3): order of the polynomial to fit
            wavelet: uses wavelet transform to baseline correct the data
                    :param wavelet: str(default 'db4'): wavelet to use
                    :param level: int(default 1): level of wavelet transform

        :param kwargs: dict: additional arguments for the method, see above

        returns None
        """
        match method:
            case "cholesky":
                self._apply_function(self._cholesky_baseline, **kwargs)
            case "linear_fit":
                self._apply_function(self._linear_fit_baseline, **kwargs)
            case "als":
                self._apply_function(self._als_baseline, **kwargs)
            case "rubberband":
                self._apply_function(self._rolling_ball_baseline, **kwargs)
            case "poly_fit":
                self._apply_function(self._polyfit_baseline, **kwargs)
            case "wavelet":
                self._apply_function(self._wavelet_baseline, **kwargs)
            case _:
                raise ValueError("method must be 'cholesky'")

    def denoise(
        self,
        method: Literal["fft", "moving_average", "wavelet", "median", "savgol"] = "fft",
        **kwargs,
    ) -> None:
        """
        Applies noise reduction to the data in self.data.

        :param method: str, method to use for noise reduction (default "fft", options: moving_average, wavelet, median)

        kwargs options:
            fft: Fast Fourier Transform denoising
                :param threshold: float (default 0.1), fraction of frequencies to retain (lower value means more
                aggressive denoising)

            moving_average: Applies a simple moving average filter for noise reduction
                :param window_size: int (default 5), size of the moving window for averaging

            wavelet: Wavelet transform denoising
                :param wavelet: str (default 'db1'), type of wavelet to use
                :param level: int (default 1), decomposition level for wavelet transform
                :param threshold: float (default 0.2), threshold for coefficient filtering

            median: Median filter denoising
                :param window_size: int (default 3), size of the filter window for median calculation

            savgol: Savitzky-Golay filter denoising
                :param window_frac: float (default 0.05), window size as fraction of the data length
                :param order: int (default 3), order of the polynomial to fit

        :param kwargs: dict, additional arguments for the chosen method
        :return: None
        """
        match method:
            case "fft":
                self._apply_function(self._fft_denoise, **kwargs)
            case "moving_average":
                self._apply_function(self._moving_average_denoise, **kwargs)
            case "wavelet":
                self._apply_function(self._wavelet_denoise, **kwargs)
            case "median":
                self._apply_function(self._median_filter_denoise, **kwargs)
            case "savgol":
                self._apply_function(self._savgol_filter_denoise, **kwargs)
            case _:
                raise ValueError(
                    "method must be one of: 'fft', 'moving_average', 'wavelet', 'median'"
                )
