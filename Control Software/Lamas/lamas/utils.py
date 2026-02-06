"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import numpy as np
import pandas as pd
from functools import wraps
from typing import Union, Tuple, List, Any, Callable, Optional, Dict, Literal
import inspect


class basekwargs:
    name: str
    description: str
    units: str

    def __init__(self, name: str, description: str, units: str):
        self.name = name
        self.description = description
        self.units = units


class initkwargs(basekwargs):
    value: Any

    def __init__(self, name: str, description: str, units: str, value: Any):
        super().__init__(name, description, units)
        self.value = value


class PeakIdentification:
    """Peak identification is a quick way to save and snapshot
    a specific peak in a spectrum. This is done by knowing which
    column and which x values the peak is represented by.

    """

    min_x: float
    max_x: float
    y_col: str

    def __init__(self, min_x: float, max_x: float, y_col: str):
        """
        contains important information about the peak
        min_x: float: the minimum x value of the peak
        max_x: float: the maximum x value of the peak
        y_col: str: the column name that contains the y values

        """
        self.min_x = min_x
        self.max_x = max_x
        # check that min_x and max_x are not equal and that min_x is less than max_x
        if min_x == max_x:
            raise ValueError("min_x and max_x cannot be equal.")
        if min_x > max_x:
            self.min_x, self.max_x = self.max_x, self.min_x
        self.y_col = y_col


class VoigtPeakIdentification(PeakIdentification):
    """
    This class is to keep track and identify a voigt peak, it stores the parameters
    of the peak, can be created from a df row and can return a df row.

    """

    def __init__(
        self,
        df_row: pd.Series | None = None,
        y_col: str | None = None,
        min_x: float = 0.0,
        max_x: float = 0.0,
        mu: float = 0.0,
        sigma: float = 0.0,
        gamma: float = 0.0,
        A: float = 0.0,
        area: float = 0.0,
    ):
        """
        :param df_row: pd.Series: the row from a dataframe that contains the peak information
        :param y_col: str: the column name that contains the y values
        :param min_x: float: the minimum x value of the peak
        :param max_x: float: the maximum x value of the peak
        :param mu: float: the center of the peak
        :param sigma: float: the standard deviation of the peak
        :param gamma: float: the gamma value of the peak
        :param A: float: the amplitude of the peak

        Remember that a voigt peak has the formula:
            Amplitude * (exp(-((x - mu) ** 2) / (2 * sigma**2))*gamma / (np.pi * ((x - mu) ** 2 + gamma**2)))
        """

        if df_row is not None:
            super().__init__(df_row["min_x"], df_row["max_x"], y_col)
            self.sigma = df_row["sigma"]
            self.gamma = df_row["gamma"]
            self.A = df_row["A"]
            self.mu = df_row["mu"]
            self.area = df_row["area"]

        else:
            super().__init__(min_x, max_x, y_col)
            self.sigma = sigma
            self.gamma = gamma
            self.A = A
            self.mu = mu
            self.area = area

        self.voigt_parameters = [self.A, self.mu, self.sigma, self.gamma]

    @property
    def peak_to_df_row(self):
        """returns the peak as a df row"""
        return pd.Series(
            {
                "y_col": self.y_col,
                "min_x": self.min_x,
                "max_x": self.max_x,
                "center": self.mu,
                "sigma": self.sigma,
                "gamma": self.gamma,
                "amplitude": self.amplitude,
                "area": self.area,
            }
        )

    def __str__(self):
        return f"VoigtPeak: y_col: {self.y_col}, min_x: {self.min_x}, max_x: {self.max_x}, mu: {self.mu}, sigma: {self.sigma}, gamma: {self.gamma}, amplitude: {self.amplitude}, area: {self.area}"


class VoigtLinearFitResult:
    """
    This class stores the linear fitting results of a Voigt peak across multiple spectra,
    along with references to the original peak objects used for the fitting.

    It includes the linear fit parameters for both amplitude and area with respect to concentration,
    the relevant statistics, and the list of `VoigtPeakIdentification` objects that matched across spectra.
    """

    def __init__(
        self,
        peak_set: list[VoigtPeakIdentification],
        slope_amplitude: float,
        intercept_amplitude: float,
        r_value_amplitude: float,
        p_value_amplitude: float,
        slope_area: float,
        intercept_area: float,
        r_value_area: float,
        p_value_area: float,
    ):
        """
        :param peak_set: list[VoigtPeakIdentification]: The matched set of Voigt peaks across spectra.
        :param slope_amplitude: float: Slope of the linear fit for amplitude vs. concentration.
        :param intercept_amplitude: float: Intercept of the linear fit for amplitude vs. concentration.
        :param r_value_amplitude: float: Correlation coefficient for amplitude fit.
        :param p_value_amplitude: float: p-value for amplitude fit.
        :param slope_area: float: Slope of the linear fit for area vs. concentration.
        :param intercept_area: float: Intercept of the linear fit for area vs. concentration.
        :param r_value_area: float: Correlation coefficient for area fit.
        :param p_value_area: float: p-value for area fit.
        """
        # Store references to the original peak objects for traceability
        self.peak_set = peak_set
        self.mu = np.mean(
            [peak.mu for peak in peak_set]
        )  # Assuming mu is consistent across matched peaks

        # Store fitting results for amplitude vs. concentration
        self.slope_amplitude = slope_amplitude
        self.intercept_amplitude = intercept_amplitude
        self.r_value_amplitude = r_value_amplitude
        self.p_value_amplitude = p_value_amplitude

        # Store fitting results for area vs. concentration
        self.slope_area = slope_area
        self.intercept_area = intercept_area
        self.r_value_area = r_value_area
        self.p_value_area = p_value_area

    @property
    def fit_summary(self):
        """
        Returns a summary of the fit results as a dictionary.

        :return: dict: Summary of fitting results.
        """
        return {
            "peak_mu": self.mu,
            "slope_amplitude": self.slope_amplitude,
            "intercept_amplitude": self.intercept_amplitude,
            "r_value_amplitude": self.r_value_amplitude,
            "p_value_amplitude": self.p_value_amplitude,
            "slope_area": self.slope_area,
            "intercept_area": self.intercept_area,
            "r_value_area": self.r_value_area,
            "p_value_area": self.p_value_area,
        }

    @property
    def fit_to_df_row(self):
        """
        Converts the fit results to a DataFrame row with references to the original peaks.

        :return: pd.Series: A row containing the fit results and peak references.
        """
        return pd.Series(self.fit_summary)

    def check_similarity_with_peaks(
        self,
        other_peak: VoigtPeakIdentification,
        similarity_function: Callable[
            [VoigtPeakIdentification, VoigtPeakIdentification], bool
        ],
        **kwargs,
    ) -> bool:
        """
        Checks similarity with another peak by comparing each peak in `self.peak_set` using a specified similarity function.

        :param other_peak: VoigtPeakIdentification: The peak to compare with each peak in `self.peak_set`.
        :param similarity_function: Callable: A function that checks if two peaks are similar (e.g., `peaks_same` from `glama`).
        :param kwargs: Additional keyword arguments to pass to `similarity_function`.

        :return: bool: True if all peaks in `self.peak_set` are similar to `other_peak` based on the similarity function, False otherwise.
        """

        return all(
            similarity_function(peak, other_peak, **kwargs) >= 0.8
            for peak in self.peak_set
        )

    def __str__(self):
        return f"VoigtLinearFitResult(mu: {self.mu:.2e}, Area_slope: {self.slope_area:.2e}, Area_intercept: {self.intercept_area:.2e}, Amplitude_slope: {self.slope_amplitude:.2e}, Amplitude_intercept: {self.intercept_amplitude:.2e}, n_peaks: {len(self.peak_set)})"


class PeakCluster:
    """class to store a cluster of peaks"""

    mu: float  # the average mu of the peaks in the cluster (can be used as a centroid mu)
    amplitude: float  # the average amplitude of the peaks in the cluster
    area: float  # the sum of the areas of the peaks in the cluster
    min_x: float  # the minimum x value of the peaks in the cluster
    max_x: float  # the maximum x value of the peaks in the cluster
    peaks: List[VoigtPeakIdentification]  # the list of peaks
    centroid_mu: float  # the area weighted average mu of the peaks in the cluster

    def __init__(self, peaks: List[VoigtPeakIdentification]):
        """
        :param peaks: List[VoigtPeakIdentification]: the list of peaks in the cluster
        """

        self.peaks = peaks if peaks is not None else []
        self.update_cluster_parameters()

    def update_cluster_parameters(self):
        """
        updates the cluster parameters, such as the average mu, sigma, gamma, and amplitude
        """
        self.mu = np.mean([peak.mu for peak in self.peaks])
        self.A = np.mean([peak.A for peak in self.peaks])
        self.area = np.sum([peak.area for peak in self.peaks])
        self.min_x = np.min([peak.min_x for peak in self.peaks])
        self.max_x = np.max([peak.max_x for peak in self.peaks])
        self.centroid_mu = np.average(
            [peak.mu for peak in self.peaks], weights=[peak.area for peak in self.peaks]
        )

    def add_peak(self, peak: VoigtPeakIdentification):
        """
        adds a peak to the cluster

        :param peak: VoigtPeakIdentification: the peak to add to the cluster
        """
        self.peaks.append(peak)
        self.update_cluster_parameters()


def extract_columns(method: Callable) -> Callable:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.data is None:
            raise ValueError("No data to process.")
        signature = inspect.signature(method)
        params = signature.parameters

        requires_x = "x" in params
        requires_y = "y" in params
        has_self = "self" in params

        y_col = self.y_cols[0] if len(self.y_cols) > 1 else self.y_cols
        x: Union[np.ndarray, None] = kwargs.pop(
            "x", self.data[self.x_col].to_numpy() if requires_x else None
        )
        y: Union[np.ndarray, None] = kwargs.pop(
            "y", self.data[y_col].to_numpy() if requires_y else None
        )
        valid_kwargs = {k: v for k, v in kwargs.items() if k in params}

        if x is not None and y is not None:
            # ensure that x and y are the same shape (should be (n,))
            x = x.reshape(-1)
            y = y.reshape(-1)

        method_args = []
        if has_self:
            method_args.append(self)
        if requires_x:
            method_args.append(x)
        if requires_y:
            method_args.append(y)
        return method(*method_args, **valid_kwargs)

    return wrapper


class LamaParameter:
    tag: str = "parameter"

    def __init__(
        self,
        name: str,
        type: Any,
        default: Any = None,
        value: Any = None,
        required: Optional[bool] = False,
        description: Optional[str] = None,
        allowed_values: Optional[list] = None,
        show_to_user: Optional[bool] = True,
    ):
        """represents a parameter that is going to be passed to a LamaMethod

        :param name: str: the name of the parameter
        :param type: Any: the type of the parameter
        :param default: Any: the default value of the parameter
        :param required: bool: whether the parameter is required
        :param description: str: the description of the parameter
        :param allowed_values: list: the allowed values for the parameter

        """
        self.name = name
        self.type = type
        self.default = default
        if value is None and default is not None:
            self.value = default
        else:
            self.value = value
        self.required = required
        self.description = description
        self.allowed_values = allowed_values
        self.show_to_user = show_to_user

    def validate(self, value):
        if self.type is not Any and not isinstance(value, self.type):
            raise ValueError(
                f"Invalid type for parameter {self.name}. Expected {self.type}, got {type(value)}"
            )
        if self.allowed_values is not None and value not in self.allowed_values:
            raise ValueError(
                f"Invalid value for parameter {self.name}. Expected one of {self.allowed_values}, got {value}"
            )

    @staticmethod
    def deserialize_type(type_obj: Any):
        """
        Deserializes the type to an actual type object.
        :param type_obj:
        :return:
        """
        type_label = type_obj.get("type", None)

        if type_label == "Literal":
            return Literal[tuple(type_obj.get("args", ()))]
        elif type_label == "Union":
            return Union[tuple(type_obj.get("args", ()))]
        elif type_label == "List":
            return List[type_obj.get("args", (Any,))]
        elif type_label == "Tuple":
            return Tuple[tuple(type_obj.get("args", (Any,)))]
        elif type_label == "Dict":
            return Dict[type_obj.get("args", (Any, Any))]
        elif type_label == "Optional":
            return Optional[type_obj.get("args", (Any,))]
        elif type_label == "<class 'float'>":
            return float
        elif type_label == "<class 'int'>":
            return int
        elif type_label == "<class 'str'>":
            return str
        elif type_label == "<class 'bool'>":
            return bool
        else:
            return Any

    @classmethod
    def from_json(cls, json_dict: dict):
        """
        reloads the LamaParameter class from a json dictionary:
        :param json_dict:
        :return: instance of LamaParameter
        """
        try:

            instance = cls(
                name=json_dict["name"],
                type=LamaParameter.deserialize_type(json_dict["type"]),
                default=json_dict.get("default", None),
                value=json_dict.get("value", None),
                required=json_dict.get("required", False),
                description=json_dict.get("description", None),
                allowed_values=json_dict.get("allowed_values", None),
                show_to_user=json_dict.get("show_to_user", True),
            )
            return instance
        except Exception as e:
            print(f"Error in LamaParameter.from_json: {e}")
            return None

    def return_dict(self):
        if not isinstance(self.value, type(self.default)):
            self.value = self.default
        return {self.name, self.value}


class LamaMethod:
    tag: str = "method"

    def __init__(
        self,
        method_name: str,
        parameters: List[LamaParameter],
        submethods: Optional[List["LamaMethod"]] = None,
        other_parameters: Optional[Dict[str, Any]] = None,
        show_to_user: Optional[bool] = True,
    ):
        """Represents a method that can be called by a Glama class

        :param method_name: The name of the method.
        :param parameters: The parameters of the method.
        :param submethods: The submethods of the method, each being a LamaMethod instance.
        :param other_parameters: Conditions to determine when submethod parameters should be included.
        """
        self.method_name = method_name
        self._parameters = parameters
        self._submethods = submethods if submethods else []
        self._other_parameters = other_parameters if other_parameters else {}
        self.show_to_user = show_to_user

    def get_parameter(self, name: str) -> Optional[LamaParameter]:
        for param in self._parameters:
            if param.name == name:
                return param
        return None

    def get_parameters(
        self, as_dict: bool = False, current_values: Optional[Dict[str, Any]] = None
    ) -> Union[List[LamaParameter], Dict[str, LamaParameter]]:
        """
        Returns the parameters of the method and the parameters of the submethods.
        If submethods have _other_parameters, it checks the conditions based on current_values
        and only includes parameters from submethods that meet the conditions.

        :param as_dict: If True, returns parameters as a dictionary; otherwise, as a list.
        :param current_values: The current parameter values to evaluate conditions.
        :return: A list or dictionary of LamaParameter instances.
        """
        if as_dict:
            params = {param.name: param for param in self._parameters}
        else:
            params = list(self._parameters)

        for submethod in self._submethods:
            if submethod._other_parameters:
                # Check if all conditions in other_parameters are met
                if current_values:
                    match = True
                    for key, value in submethod._other_parameters.items():
                        if key not in current_values or current_values[key] != value:
                            match = False
                            break
                    if match:
                        if as_dict:
                            params.update(
                                {param.name: param for param in submethod._parameters}
                            )
                        else:
                            params.extend(submethod._parameters)
            else:
                # No conditions, include submethod's parameters
                if as_dict:
                    params.update(
                        {param.name: param for param in submethod._parameters}
                    )
                else:
                    params.extend(submethod._parameters)

        if as_dict:
            return params
        return params

    @classmethod
    def from_json(cls, json_dict: dict):
        """
        reloads the LamaMethod class from a json dictionary:
        :param json_dict:
        :return: instance of LamaMethod
        """
        try:
            instance = cls(
                method_name=json_dict["method_name"],
                parameters=[
                    LamaParameter.from_json(json_dict=param)
                    for param in json_dict["_parameters"]
                ],
                submethods=[
                    cls.from_json(submethod) for submethod in json_dict["_submethods"]
                ],
                other_parameters=json_dict.get("_other_parameters", {}),
                show_to_user=json_dict.get("show_to_user", True),
            )
            return instance
        except Exception as e:
            print(f"Error in LamaMethod.from_json: {e}")
            return None
