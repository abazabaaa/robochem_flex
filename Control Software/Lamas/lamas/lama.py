"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: The `Lama` class is the last piece of the puzzle. It predicts the concentrations of known compounds in a new spectrum
using models built from previous data. It also identifies newly emerged peaks that cannot be assigned to known compounds.

Given a new spectrum and the models (linear fit results) from a `Guanaco` instance, this class can:
- Predict concentrations of known compounds present in the new spectrum.
- Identify unmatched peaks that may indicate unknown compounds or anomalies.
"""

import numpy as np
from typing import List, Dict, Set, Any, Tuple, Literal

from lamas.utils import (
    VoigtPeakIdentification,
    VoigtLinearFitResult,
    LamaMethod,
    LamaParameter,
)
from lamas.glama import Glama
from lamas.guanaco import Guanaco


class Lama(Glama):
    """
    The `Lama` class predicts concentrations of known compounds in a new spectrum
    and identifies any unassigned peaks.

    Attributes:
        linear_fit_results (Dict[str, List[VoigtLinearFitResult]]): Linear fit results for known compounds.
    """

    linear_fit_results: Dict[str, List[VoigtLinearFitResult]]
    structure: Dict[str, Any] = {
        "predict_concentrations": LamaMethod(
            method_name="predict_concentrations",
            parameters=[
                LamaParameter(
                    name="new_spectrum",
                    type=List[VoigtPeakIdentification],
                    description="A list of Voigt peaks from the new spectrum.",
                    show_to_user=False,
                ),
                LamaParameter(
                    name="known_compounds",
                    type=Set[str],
                    description="Set of compounds expected in the new spectrum.",
                    show_to_user=False,
                ),
                LamaParameter(
                    name="mode",
                    type=Literal[
                        "relative_threshold",
                        "weighted_similarity",
                        "mu_only",
                        "correlation",
                        "fwhm",
                        "overlap_area",
                        "dtw",
                        "cross_correlation",
                    ],
                    default="relative_treshold",
                    description="The mode of the peak matching algorithm, relative_treshold or absolute_treshold",
                ),
            ],
            submethods=[
                LamaMethod(
                    method_name="relative_threshold",
                    parameters=[
                        LamaParameter(
                            name="threshold",
                            type=float,
                            default=0.1,
                            description="The maximum difference in mu values for peaks to be considered matching.",
                        ),
                    ],
                    other_parameters={"mode": "relative_threshold"},
                ),
                LamaMethod(
                    method_name="weighted_similarity",
                    parameters=[
                        LamaParameter(
                            name="threshold",
                            type=float,
                            default=0.1,
                            description="The maximum difference in mu values for peaks to be considered matching.",
                        ),
                        LamaParameter(
                            name="weights",
                            type=Dict[str, float],
                            default=None,
                            description="Dictionary of weights for each parameter",
                        ),
                    ],
                    other_parameters={"mode": "weighted_similarity"},
                ),
                LamaMethod(
                    method_name="mu_only",
                    parameters=[
                        LamaParameter(
                            name="threshold",
                            type=float,
                            default=0.1,
                            description="The maximum difference in mu values for peaks to be considered matching.",
                        ),
                    ],
                    other_parameters={"mode": "mu_only"},
                ),
                LamaMethod(
                    method_name="correlation",
                    parameters=[
                        LamaParameter(
                            name="num_points",
                            type=int,
                            default=100,
                            description="Number of points to use for the correlation",
                        )
                    ],
                    other_parameters={"mode": "correlation"},
                ),
                LamaMethod(
                    method_name="fwhm",
                    parameters=[
                        LamaParameter(
                            name="threshold",
                            type=float,
                            default=0.1,
                            description="The maximum difference in mu values for peaks to be considered matching.",
                        ),
                    ],
                    other_parameters={"mode": "fwhm"},
                ),
                LamaMethod(
                    method_name="dtw",
                    parameters=[
                        LamaParameter(
                            name="threshold",
                            type=float,
                            default=0.1,
                            description="The maximum difference in mu values for peaks to be considered matching.",
                        )
                    ],
                    other_parameters={"mode": "dtw"},
                ),
                LamaMethod(
                    method_name="overlap_area",
                    parameters=[
                        LamaParameter(
                            name="num_points",
                            type=int,
                            default=100,
                            description="Number of points to use for the overlap area",
                        )
                    ],
                    other_parameters={"mode": "overlap_area"},
                ),
                LamaMethod(
                    method_name="cross_correlation",
                    parameters=[
                        LamaParameter(
                            name="num_points",
                            type=int,
                            default=100,
                            description="Number of points to use for the cross correlation",
                        )
                    ],
                    other_parameters={"mode": "cross_correlation"},
                ),
            ],
        ),
    }

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the `Lama` instance, passing any keyword arguments to the superclass initializer.
        """
        super().__init__(**kwargs)

    def absorb_fits_from_class(self, guanaco_class: Guanaco) -> None:
        """
        Absorbs the linear fit results from a `Guanaco` class instance and stores them in the `Lama` instance.

        Args:
            guanaco_class (Guanaco): The `Guanaco` class instance containing the linear fit results.

        Raises:
            AttributeError: If the `Guanaco` class does not have the linear fit results.
        """
        self.log("Absorbing linear fit results from the Guanaco class.", indent="enter")
        if (
            not hasattr(guanaco_class, "filtered_peak_fits")
            or guanaco_class.filtered_peak_fits is None
        ):
            self.log(
                "Guanaco class does not have the linear fit results. Please generate them and try again.",
                level="error",
                indent="exit",
            )
            raise AttributeError(
                "Guanaco class does not have the linear fit results. Please generate them and try again."
            )

        self.linear_fit_results = guanaco_class.filtered_peak_fits
        self.log("Linear fit results absorbed successfully.", level="ok", indent="exit")

    def predict_concentrations(
        self,
        new_spectrum: List[VoigtPeakIdentification],
        known_compounds: Set[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Predicts the concentrations of known compounds in a new spectrum and identifies any unassigned peaks.

        Args:
            new_spectrum (List[VoigtPeakIdentification]): A list of Voigt peaks from the new spectrum.
            known_compounds (Set[str]): Set of compounds expected in the new spectrum.
            **kwargs: Additional keyword arguments passed to the `peaks_same` method for peak similarity checks.

        Raises:
            ValueError: If linear fit results are not loaded.

        Returns:
            Dict[str, Any]: Dictionary containing concentration predictions and unmatched peaks:
                - 'concentration_results': Dict with compound as key and dict with 'predicted_concentration',
                  'error', and 'num_peaks_used' as values.
                - 'unmatched_peaks': List of `VoigtPeakIdentification` objects that could not be assigned.
        """
        self.log(
            "Starting concentration prediction for the new spectrum...", indent="enter"
        )

        # Ensure required data is loaded
        self._ensure_linear_fit_results_loaded()

        # Initialize data structures for storing concentration results and unmatched peaks
        compound_concentrations, unmatched_peaks = self._initialize_concentration_data(
            known_compounds
        )

        # Iterate over each peak in the new spectrum
        for new_peak in new_spectrum:
            peak_assigned = self._assign_peak_to_compounds(
                new_peak, known_compounds, compound_concentrations, **kwargs
            )

            # If no match was found, add the peak to unmatched_peaks
            if not peak_assigned:
                unmatched_peaks.append(new_peak)
                self.log(
                    f"Peak at mu={new_peak.mu:.4f} could not be assigned to any known compound.",
                    level="warning",
                )

        # Calculate the mean concentration and standard deviation for each compound
        concentration_results = self._calculate_compound_concentrations(
            compound_concentrations
        )

        # Log and return results
        self.log("Concentration prediction completed.", level="ok", indent="exit")
        return {
            "concentration_results": concentration_results,
            "unmatched_peaks": unmatched_peaks,
        }

    def _ensure_linear_fit_results_loaded(self) -> None:
        """
        Ensures that linear fit results are loaded into the instance.

        Raises:
            ValueError: If linear fit results are not loaded.
        """
        self.log("Ensuring linear fit results are loaded.", indent="enter")
        if not hasattr(self, "linear_fit_results"):
            self.log(
                "Linear fit results not loaded. Run `absorb_fits_from_class` first.",
                level="error",
                indent="exit",
            )
            raise ValueError(
                "Linear fit results not loaded. Run `absorb_fits_from_class` first."
            )
        self.log("Linear fit results are loaded.", level="ok", indent="exit")

    def _initialize_concentration_data(
        self, known_compounds: Set[str]
    ) -> Tuple[Dict[str, List[float]], List[VoigtPeakIdentification]]:
        """
        Initializes data structures for storing concentration results and unmatched peaks.

        Args:
            known_compounds (Set[str]): Set of compounds expected in the new spectrum.

        Returns:
            Tuple[Dict[str, List[float]], List[VoigtPeakIdentification]]: A dictionary to store concentrations per compound,
                and a list to store unmatched peaks.
        """
        self.log("Initializing concentration data structures.", indent="enter")
        compound_concentrations: Dict[str, List[float]] = {
            compound: [] for compound in known_compounds
        }
        unmatched_peaks: List[VoigtPeakIdentification] = []
        self.log("Initialization complete.", level="ok", indent="exit")
        return compound_concentrations, unmatched_peaks

    def _assign_peak_to_compounds(
        self,
        new_peak: VoigtPeakIdentification,
        known_compounds: Set[str],
        compound_concentrations: Dict[str, List[float]],
        **kwargs: Any,
    ) -> bool:
        """
        Attempts to assign a peak to one of the known compounds and updates the compound concentrations.

        Args:
            new_peak (VoigtPeakIdentification): The peak from the new spectrum to be assigned.
            known_compounds (Set[str]): Set of compounds expected in the new spectrum.
            compound_concentrations (Dict[str, List[float]]): Dictionary to store concentrations per compound.
            **kwargs: Additional keyword arguments passed to the `peaks_same` method for peak similarity checks.

        Returns:
            bool: True if the peak was assigned to a compound, False otherwise.
        """
        self.log(
            f"Attempting to assign peak at mu={new_peak.mu:.4f} to known compounds.",
            indent="enter",
        )
        # Iterate through each known compound
        for compound in known_compounds:
            fit_results = self.linear_fit_results.get(compound, [])
            if not fit_results:
                self.log(
                    f"No fit results available for compound '{compound}'. Skipping.",
                    level="warning",
                )
                continue  # Skip compounds without fit results

            # Go through each fitted peak result for the compound to check for a match with `new_peak`
            for fit_result in fit_results:
                similarity_score = fit_result.check_similarity_with_peaks(
                    new_peak, self.peak_same, **kwargs
                )
                if similarity_score == True:
                    # Calculate concentration using the linear model for amplitude
                    estimated_concentration = (
                        fit_result.slope_amplitude * new_peak.A
                        + fit_result.intercept_amplitude
                    )
                    compound_concentrations[compound].append(estimated_concentration)
                    self.log(
                        f"Peak at mu={new_peak.mu:.4f} assigned to compound '{compound}' with estimated concentration {estimated_concentration:.4f}.",
                    )
                    self.log(
                        f"Peak assignment completed for mu={new_peak.mu:.4f}.",
                        indent="exit",
                    )
                    return True  # Peak assigned

        self.log(
            f"Peak at mu={new_peak.mu:.4f} could not be assigned to any known compound.",
            level="warning",
            indent="exit",
        )
        return False  # Peak not assigned

    def _calculate_compound_concentrations(
        self, compound_concentrations: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates the mean concentration and standard deviation for each compound.

        Args:
            compound_concentrations (Dict[str, List[float]]): Dictionary containing estimated concentrations per compound.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary with compound as key and a dictionary containing 'predicted_concentration',
                'error', and 'num_peaks_used' as values.
        """
        self.log("Calculating compound concentrations.", indent="enter")
        concentration_results: Dict[str, Dict[str, Any]] = {}
        for compound, concentrations in compound_concentrations.items():
            if concentrations:
                mean_concentration = np.mean(concentrations)
                std_dev_concentration = np.std(concentrations)
                concentration_results[compound] = {
                    "predicted_concentration": mean_concentration,
                    "error": std_dev_concentration,
                    "num_peaks_used": len(concentrations),
                }
                self.log(
                    f"Compound '{compound}' predicted concentration: {mean_concentration:.4f} ± {std_dev_concentration:.4f} (from {len(concentrations)} peaks).",
                    level="ok",
                )
            else:
                concentration_results[compound] = {
                    "predicted_concentration": 0.0,
                    "error": 0.0,
                    "num_peaks_used": 0,
                }
                self.log(
                    f"No peaks matched for compound '{compound}'. Predicted concentration set to 0.",
                    level="warning",
                )
        self.log("Compound concentration calculation completed.", indent="exit")
        return concentration_results
