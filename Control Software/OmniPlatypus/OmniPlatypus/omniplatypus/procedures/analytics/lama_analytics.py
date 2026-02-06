"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Ok the idea is to make a processing function that takes in a spectrum and returns whatever we need
to do our analysis on it. now the tricky bit is to return something that that can be used no matter what the
machine learning requests.

"""

from typing import Tuple

import numpy as np
import pandas as pd
from lamas.alpaca import Alpaca
from lamas.lama import Lama
from lamas.vicuna import Vicuna
from lamas.guanaco import Guanaco
import ast
import os
import joblib
from scipy.stats import t


def processing_function_isotope_exchange(
    data: pd.DataFrame,
    recipe: dict,
    non_spectrometer_parameters: dict,
    conditions: dict,
    logger=None,
):
    """
    This function takes in a spectrum and returns the necessary data to perform the analysis.
    :param data: pd.DataFrame the spectrum to be processed
    :param recipe: dict the recipe for the experiment
    :param non_spectrometer_parameters: dict the parameters for the experiment
    :param conditions: dict the conditions for the experiment

    """
    all_parameters = non_spectrometer_parameters["all"]
    crop_range = ast.literal_eval(all_parameters["crop_range"].value)
    normalise_range = ast.literal_eval(all_parameters["normalise_range"].value)
    crop_zoom = ast.literal_eval(all_parameters["crop_zoom"].value)

    product_boundary = all_parameters["peak_of_product_boundary"].value
    starting_material_boundary = all_parameters[
        "peak_of_starting_material_boundary"
    ].value
    isosbestic_point = all_parameters["isosbestic_point"].value

    # Load the spectrum
    spectrum = Alpaca(logger=logger)
    spectrum.load_data(data, from_filetype="ramaberry")

    # do processing
    spectrum.crop_data(bounds=crop_range)
    spectrum.baseline(method="cholesky", lam=10**6, p=0.03, n_iter=20)
    spectrum.normalise_to(to=1000, range_vals=normalise_range)
    spectrum.crop_data(bounds=(1520, 2000))

    spectrum.denoise(method="median", window=5)
    spectrum.crop_data(bounds=crop_zoom)
    spectrum.baseline(method="als", lam=10**6, p=0.03, n_iter=20)

    bounds_product = tuple(sorted([product_boundary, isosbestic_point]))
    bounds_starting_material = tuple(
        sorted([starting_material_boundary, isosbestic_point])
    )

    # get the integral of the product
    integral_product = spectrum.integrate(bounds=bounds_product, mode="area")
    integral_starting_material = spectrum.integrate(
        bounds=bounds_starting_material, mode="area"
    )

    return integral_product, integral_starting_material


def processing_function_integrated_isotope_exchange(
    data: pd.DataFrame,
    recipe: dict,
    non_spectrometer_parameters: dict,
    conditions: dict,
    logger=None,
):
    """
    This function takes in a spectrum and returns the necessary data to perform the analysis.
    :param data: pd.DataFrame the spectrum to be processed
    :param recipe: dict the recipe for the experiment
    :param non_spectrometer_parameters: dict the parameters for the experiment
    :param conditions: dict the conditions for the experiment
    :param logger: logger instance for logging
    :return: dict with calculated concentrations and areas
    """
    (
        peak_of_product_area,
        peak_of_starting_material_area,
    ) = processing_function_isotope_exchange(
        data=data,
        recipe=recipe,
        non_spectrometer_parameters=non_spectrometer_parameters,
        conditions=conditions,
        logger=logger,
    )

    # Handle None or zero areas
    peak_of_product_area = (
        peak_of_product_area if peak_of_product_area is not None else 0
    )
    peak_of_starting_material_area = (
        peak_of_starting_material_area
        if peak_of_starting_material_area is not None
        else 0
    )

    # Load the linear model
    linear_model_path = (
        non_spectrometer_parameters["all"].get("path_to_calibration_file").value
    )
    if linear_model_path is None:
        return {
            "PI_area": peak_of_product_area,
            "SM_area": peak_of_starting_material_area,
        }

    lm = load_linear_model(linear_model_path)
    if lm is None:
        return {
            "PI_area": peak_of_product_area,
            "SM_area": peak_of_starting_material_area,
        }

    bins_PI = lm["bins_PI"]
    bins_SM = lm["bins_SM"]
    stds_PI = lm["stds_PI"]
    stds_SM = lm["stds_SM"]

    peak_starting_material_var = interpolate_std_for_new_integral(
        peak_of_starting_material_area, bins_SM, stds_SM
    )
    peak_product_var = interpolate_std_for_new_integral(
        peak_of_product_area, bins_PI, stds_PI
    )

    # linear_models_starting_material = lm["model_SM"]
    # X_train_starting_material = lm["X_SM"]
    # y_train_starting_material = lm["y_SM"]
    # linear_models_product = lm["model_PI"]
    # X_train_product = lm["X_PI"]
    # y_train_product = lm["y_PI"]

    # # Calculate prediction intervals
    # intervals_starting_material = prediction_interval(
    #     linear_models_starting_material,
    #     X_train_starting_material,
    #     y_train_starting_material,
    #     [[peak_of_starting_material_area]],
    # )
    # intervals_product = prediction_interval(
    #     linear_models_product,
    #     X_train_product,
    #     y_train_product,
    #     [[peak_of_product_area]],
    # )
    #
    # # Predict concentrations
    # peak_of_product_conc = (
    #     intervals_product["mean"][0].item() if peak_of_product_area else 0
    # )
    # peak_starting_material_conc = (
    #     intervals_starting_material["mean"][0].item()
    #     if peak_of_starting_material_area
    #     else 0
    # )
    # peak_product_var = (
    #     intervals_product["variance"][0].item() if peak_of_product_area else 0
    # )
    # peak_starting_material_var = (
    #     intervals_starting_material["variance"][0].item()
    #     if peak_of_starting_material_area
    #     else 0
    # )

    return {
        "SM_conc": peak_of_starting_material_area,
        "PI_conc": peak_of_product_area,
        "SM_var": peak_starting_material_var**2,
        "PI_var": peak_product_var**2,
        "SM_area": peak_of_starting_material_area,
        "PI_area": peak_of_product_area,
    }


def load_linear_model(path: str):
    """
    Load a linear model from a path
    """
    # check if the path is a file and if it is a linear model and if it exists
    if not os.path.isfile(path):
        return None
    if not path.endswith(".linear_model"):
        return None
    return joblib.load(path)


def prediction_interval(model, X_train, y_train, X_new, confidence=0.95):
    """
    Calculate the prediction interval for new predictions.

    :param model: Trained linear regression model
    :param X_train: Training inputs (numpy array)
    :param y_train: Training targets (numpy array)
    :param X_new: New inputs for prediction (numpy array)
    :param confidence: Confidence level for the interval (default: 0.95)
    :return: Tuple containing predicted values, lower bounds, and upper bounds
    """
    # Predict for new inputs
    y_pred = model.predict(X_new)

    # Residual standard error (RSE)
    y_train_pred = model.predict(X_train)
    residuals = y_train - y_train_pred
    rse = np.sqrt(
        np.sum(residuals**2) / (len(X_train) - 2)
    )  # Degrees of freedom = n - 2

    # Mean and variance of X
    X_mean = np.mean(X_train)
    Sxx = np.sum((X_train - X_mean) ** 2)

    # Critical t-value
    t_value = t.ppf(1 - (1 - confidence) / 2, df=len(X_train) - 2)

    # Prediction interval for each X_new
    intervals = {
        "mean": [],
        "lower": [],
        "upper": [],
        "margin": [],
        "variance": [],
        "std_dev": [],
    }
    for x_new in X_new:
        se_pred = rse * np.sqrt(1 + (1 / len(X_train)) + ((x_new - X_mean) ** 2 / Sxx))
        margin = t_value * se_pred
        variance_pred = se_pred**2  # Variance is the square of the std dev
        intervals["mean"].append(y_pred)
        intervals["lower"].append(y_pred - margin)
        intervals["upper"].append(y_pred + margin)
        intervals["margin"].append(margin)
        intervals["variance"].append(variance_pred)
        intervals["std_dev"].append(se_pred)

    return intervals


def interpolate_std_for_new_integral(
    I_new: float, bin_means: np.array, bin_stds: np.array
):
    """
    Linearly interpolate or extrapolate the standard deviation
    for a new integral value I_new based on bin_means/bin_stds taken from the linear model

    :param I_new: a float value of anew measured integral
    :param bin_means: np.array of the bin positions
    :param bin_stds: np.array of the stdev for each bin

    returns float: the interpolated stdev at the I_new
    """
    if len(bin_means) < 2:
        return float(bin_stds[0])  # trivial case
    # Extrapolate below
    if I_new < bin_means[0]:
        x1, x2 = bin_means[0], bin_means[1]
        y1, y2 = bin_stds[0], bin_stds[1]
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (I_new - x1)
    # Extrapolate above
    if I_new > bin_means[-1]:
        x1, x2 = bin_means[-2], bin_means[-1]
        y1, y2 = bin_stds[-2], bin_stds[-1]
        slope = (y2 - y1) / (x2 - x1)
        return y2 + slope * (I_new - x2)
    # Piecewise linear interpolation
    for i in range(len(bin_means) - 1):
        if bin_means[i] <= I_new <= bin_means[i + 1]:
            x1, x2 = bin_means[i], bin_means[i + 1]
            y1, y2 = bin_stds[i], bin_stds[i + 1]
            slope = (y2 - y1) / (x2 - x1)
            return y1 + slope * (I_new - x1)
    # Fallback (shouldn’t normally happen)
    return float(bin_stds[-1])
