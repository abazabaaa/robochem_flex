"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Guanacos are the buddies of the Vicunas, they live at lower altitudes,
similarly here the Guanaco class is used to generate linearity models for chemicals.
The idea is that given a set of spectra where the concentration of chemicals is known, (after being vicuna modelled)
this class should be able to pick out which peaks are from which chemical and then generate a caliration curve for each chemical


"""

import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Literal
from scipy.stats import linregress

from lamas.utils import (
    VoigtPeakIdentification,
    VoigtLinearFitResult,
    LamaMethod,
    LamaParameter,
    PeakCluster,
)
from lamas.utils import VoigtPeakIdentification, VoigtLinearFitResult, PeakCluster
from lamas.glama import Glama
from lamas.vicuna import Vicuna
import os
import itertools


class Guanaco(Glama):
    """
    The `Guanaco` class isolates sets of peaks related to a single compound
    and generates calibration curves for each compound.

    Attributes:
        conc_df (pd.DataFrame): DataFrame containing the concentration for each compound
            in the corresponding file (or y column in the spectrum).
        sub_df_dict (Dict[str, pd.DataFrame]): Dictionary containing sub-dataframes for each compound,
            where each key is the compound name.
        models (Dict[str, List[VoigtPeakIdentification]]): Dictionary containing the Voigt models for each spectrum.

    Usage:
    first instantiate the class and then load the models and the concentration data
    naco = Guanaco(**kwargs) -> kwargs are passed to the superclass

    naco.load_

    """

    structure: Dict[str, Any] = {
        "peak_matching": LamaMethod(
            method_name="peak_matching",
            parameters=[
                LamaParameter(
                    name="delta_mu",
                    type=float,
                    default=0.1,
                    description="The maximum difference in mu values for peaks to be considered matching.",
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
        "fit_and_filter_peaks": LamaMethod(
            method_name="fit_and_filter_peaks",
            parameters=[
                LamaParameter(
                    name="r2_threshold",
                    type=float,
                    default=0.8,
                    description="Minimum R^2 value to consider the fit acceptable.",
                ),
                LamaParameter(
                    name="p_value_threshold",
                    type=float,
                    default=0.05,
                    description="Maximum p-value to consider the fit significant.",
                ),
            ],
        ),
        "save_data": LamaMethod(
            method_name="save_data",
            parameters=[
                LamaParameter(
                    name="base_directory",
                    type=str,
                    default="filtered_peak_fits",
                    description="Directory to save the data files.",
                )
            ],
        ),
        "load_fits": LamaMethod(
            method_name="load_fits",
            parameters=[
                LamaParameter(
                    name="base_directory",
                    type=str,
                    default="filtered_peak_fits",
                    description="Directory where the data files are saved.",
                )
            ],
        ),
    }

    conc_df: pd.DataFrame
    sub_df_dict: Dict[str, pd.DataFrame]
    models: Dict[str, List[VoigtPeakIdentification]]

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the Guanaco instance, passing any keyword arguments to the superclass initializer.
        """
        super().__init__(**kwargs)

    def load_models_from_dir(self, path_to_models: str) -> None:
        """
        Loads Voigt models from a specified directory.

        This method searches for all files in the given directory that have the extension '.models',
        reads each file as a CSV into a DataFrame, and creates a list of `VoigtPeakIdentification` objects
        for each model. The models are stored in the `self.models` dictionary with the filename (without extension)
        as the key.

        Args:
            path_to_models (str): The path to the directory containing the model files.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        self.log(
            f"Loading Voigt models from directory: {path_to_models}", indent="enter"
        )
        try:
            self.models: Dict[str, List[VoigtPeakIdentification]] = (
                {} if not hasattr(self, "models") else self.models
            )
            for file in os.listdir(path_to_models):
                if file.endswith(".models"):
                    model_path = os.path.join(path_to_models, file)
                    model = pd.read_csv(model_path)
                    self.models[file.split(".")[0]] = [
                        VoigtPeakIdentification(**row) for _, row in model.iterrows()
                    ]
                    self.log(f"Loaded model from {model_path}")
            self.logger.log_message(
                "All models loaded successfully.", level="ok", indent="exit"
            )
        except FileNotFoundError:
            self.log(
                f"Could not find the directory {path_to_models}",
                level="error",
                indent="exit",
            )
            raise FileNotFoundError(f"Could not find the directory {path_to_models}")

    def load_models_from_class(self, vicuna_class: Vicuna) -> None:
        """loads the models from a vicuna class, converting them to VoigtPeakIdentification objects
        and copying them to the self.models attribute
        """
        self.log("Loading models from a Vicuna class.", indent="enter")
        if (
            not hasattr(vicuna_class, "voigt_parameters")
            or vicuna_class.voigt_parameters is None
        ):
            self.log(
                "The provided Vicuna class has no models. Skipping.", level="warning"
            )
            self.log("Finished loading models.", indent="exit")
            return
        self.models: Dict[str, List[VoigtPeakIdentification]] = (
            {} if not hasattr(self, "models") else self.models
        )
        for filename, model in vicuna_class.voigt_parameters.items():
            self.models[filename] = [
                VoigtPeakIdentification(params) for i,params in model.iterrows()
            ]
            self.log(f"Loaded model from {filename}")

        self.log("All models loaded successfully.", level="ok", indent="exit")

    def _check_models_loaded(self) -> None:
        """
        Checks whether the concentration DataFrame and models are loaded, and verifies that
        all spectra present in the concentration DataFrame have corresponding models loaded.

        Raises:
            AssertionError: If not all spectra are present in the models.
            Exception: If concentration DataFrame or models are not loaded.
        """
        self.log(
            "Checking if models and concentration data are loaded.", indent="enter"
        )
        if hasattr(self, "conc_df") and hasattr(self, "models"):
            loaded_models = set(self.models.keys())
            loaded_spectra = set(self.conc_df[self.filename_col].values)
            missing_spectra = loaded_spectra - loaded_models
            if missing_spectra:
                self.log(
                    f"Not all spectra are present in the models. Missing spectra: {missing_spectra}",
                    level="error",
                    indent="exit",
                )
                raise AssertionError("Not all the spectra are present in the models")
            else:
                self.log(
                    "All spectra have corresponding models.", level="ok", indent="exit"
                )
        else:
            self.log(
                "Something went wrong with checking the model loading.",
                level="warning",
                indent="enter",
            )
            if not hasattr(self, "conc_df"):
                self.log(
                    "The concentration DataFrame is not loaded.",
                    level="warning",
                )
            if not hasattr(self, "models"):
                self.log(
                    "The models are not loaded.",
                    level="warning",
                )
            self.log(
                "Please load the concentration DataFrame and the models.",
                level="error",
                indent="exit",
            )
            raise Exception("Concentration DataFrame or models not loaded.")

    def load_concentrations_from_file(
        self, conc_file: str, filename_col: str = "filename"
    ) -> None:
        """
        Loads the concentration DataFrame from a CSV file and stores it in the `conc_df` attribute.

        Args:
            conc_file (str): The path to the concentration CSV file.

        Raises:
            FileNotFoundError: If the specified concentration file does not exist.
        """
        self.log("Loading concentration file.", indent="enter")
        try:
            self.conc_df = pd.read_csv(conc_file)
            self.filename_col = filename_col
            self.log(
                "Concentration file loaded successfully.", level="ok", indent="exit"
            )
        except FileNotFoundError:
            self.log(
                f"Could not find the concentration file: {conc_file}",
                level="error",
                indent="exit",
            )
            raise FileNotFoundError(
                f"Could not find the concentration file: {conc_file}"
            )

    def load_concentrations_from_class(self, vicuna_class: Vicuna) -> None:
        """
        Loads concentration data from a Vicuna class and merges it into `self.conc_df`.

        If `self.conc_df` does not exist or is None, it initializes it with `vicuna_class.concentrations`.
        If `self.conc_df` exists, it performs an outer merge with `vicuna_class.concentrations`
        on the 'filename' column, adding any missing rows and columns with zero-fill for missing values.

        Raises an error if there are conflicts in overlapping cells.

        Args:
            vicuna_class (Vicuna): An instance of the Vicuna class with a `concentrations` attribute.

        Raises:
            TypeError: If `vicuna_class.concentrations` is not a DataFrame.
            ValueError: If there are conflicts in overlapping cells during the merge.
        """
        self.log("Loading concentration data from a Vicuna class.", indent="enter")
        # Check if vicuna_class has a non-empty concentrations DataFrame
        if (
            not hasattr(vicuna_class, "concentrations")
            or vicuna_class.concentrations is None
        ):
            self.log(
                "The provided Vicuna class has no concentration data. Skipping.",
                level="warning",
            )
            self.log("Finished loading concentration data.", indent="exit")
            return

        # Ensure vicuna_class.concentrations is a DataFrame
        if not isinstance(vicuna_class.concentrations, pd.DataFrame):
            self.log(
                "vicuna_class.concentrations must be a DataFrame.",
                level="error",
                indent="exit",
            )
            raise TypeError("vicuna_class.concentrations must be a DataFrame.")

        # If self.conc_df doesn't exist or is None, initialize it with vicuna_class.concentrations
        if not hasattr(self, "conc_df") or self.conc_df is None:
            self.log(
                "Initializing self.conc_df with vicuna_class.concentrations.",
            )
            self.conc_df = vicuna_class.concentrations.copy()
        else:
            # Perform an outer merge on 'filename' without suffixes and check for conflicts
            self.log("Merging concentration data with existing data.", indent="enter")
            merged = pd.merge(
                self.conc_df,
                vicuna_class.concentrations,
                on="filename",
                how="outer",
                suffixes=("_self", "_vicuna"),
            )
            self.log(f"New merged DataFrame:\n{merged}")
            self.log("Checking for conflicts in overlapping cells.", indent="exit")
            # Identify common columns (excluding 'filename') to check for conflicts
            common_cols = set(self.conc_df.columns) & set(
                vicuna_class.concentrations.columns
            ) - {"filename"}

            # Check for conflicts in overlapping cells for each common column
            for col in common_cols:
                conflict = merged.loc[
                    (merged[col + "_self"].notna())
                    & (merged[col + "_vicuna"].notna())
                    & (merged[col + "_self"] != merged[col + "_vicuna"])
                ]
                if not conflict.empty:
                    self.log(
                        f"Conflict detected in column [{col}] for rows:\n{conflict[['filename', col + '_self', col + '_vicuna']]}",
                        level="error",
                        indent="exit",
                    )
                    raise ValueError(
                        f"Conflict detected in column '{col}' for rows:\n{conflict[['filename', col + '_self', col + '_vicuna']]}"
                    )
            self.log("No conflicts detected.", level="ok")
            self.log("Updating _self columns with _vicuna values when _self is NaN.", indent = "enter")
            for col in common_cols:
                self.log(f"working on column {col}")
                merged[col+"_self"] = merged[col + "_self"].fillna(merged[col + "_vicuna"])
            
            self.log("Successfully updated _self columns with _vicuna values.", level="ok", indent="exit")
            # Drop the duplicate columns created by the merge ('*_self' and '*_vicuna' suffixes)
            self.log("Dropping duplicate columns.")
            merged.drop(columns=[col + "_vicuna" for col in common_cols], inplace=True)
            merged.rename(
                columns={col + "_self": col for col in common_cols}, inplace=True
            )

            # Assign the merged DataFrame back to self.conc_df
            self.conc_df = merged
            self.filename_col = self.conc_df.columns[0]

        # Fill any NaN values with zeros
        self.conc_df.fillna(0, inplace=True)
        self.log("Concentration data loaded successfully.", level="ok", indent="exit")

    def make_subdataframes_concentration(self) -> None:
        """
        Creates a dictionary of sub-dataframes for each compound based on concentration data.

        For each concentration column in `self.conc_df` (columns starting with "Conc"),
        this method extracts the compound name and creates a sub-dataframe containing only
        the rows where the concentration of that compound is non-zero. The sub-dataframes
        are stored in `self.sub_df_dict` with the compound name as the key.

        Returns:
            None
        """
        self.log("Creating sub-dataframes for each compound.", indent="enter")
        self.sub_df_dict = {}

        conc_columns = [col for col in self.conc_df.columns if col.startswith("Conc")]

        for col in conc_columns:
            compound = col.split("_")[1]
            self.log(f"Processing compound: {compound}")
            self.sub_df_dict[compound] = self.conc_df[self.conc_df[col] != 0]

        self.log("Sub-dataframes created successfully.", level="ok", indent="exit")

    def peak_matching(self, delta_mu: float = 0.1, **kwargs: Any) -> None:
        """
        Identifies common peaks across spectra for each compound in `self.sub_df_dict`, organizing them as Clusters.

        Args:
            delta_mu (float): The maximum difference in `mu` values for peaks to be considered matching.
            **kwargs: Additional arguments passed to `peaks_same` for peak similarity checks.

        Raises:
            ValueError: If models or sub-dataframes are not properly loaded.

        Returns:
            None: Results are stored directly in `self.common_peaks`, organized by compound.
        """
        self.log(
            f"Starting peak matching process with delta_mu = {delta_mu:.4f}.",
            indent="enter",
        )
        # Ensure required data is loaded
        self._check_models_loaded()
        self.make_subdataframes_concentration()

        # Dictionary to store common peaks per compound as lists of matched peaks
        common_peaks: Dict[str, List[List[VoigtPeakIdentification]]] = {}

        # Process each compound's sub-dataframe
        for compound, df in self.sub_df_dict.items():
            self.log(f"Processing compound: {compound}", indent="enter")
            # Collect peaks per spectrum
            spectra_peaks, spectra_files = self._collect_peaks_per_spectrum(df)

            if not spectra_files:
                self.log(
                    f"No spectra files found for compound {compound}. Skipping.",
                    level="warning",
                    indent="exit",
                )
                continue

            # Flatten all peaks into a single list
            all_peaks = [peak for peaks in spectra_peaks.values() for peak in peaks]

            # Cluster peaks based on delta_mu
            clusters = self._cluster_peaks(all_peaks, delta_mu)

            # For each cluster, select the best combination of peaks
            matched_peaks: List[List[VoigtPeakIdentification]] = []
            for cluster in clusters:
                best_combination = self._select_best_peak_combination(
                    cluster, spectra_files, **kwargs
                )
                if best_combination:
                    matched_peaks.append(best_combination)
                    self.log(
                        f"Found common peak cluster at mu ~ {cluster.centroid_mu:.4f}"
                    )
                else:
                    self.log(
                        f"No valid peak combination found for cluster at mu ~ {cluster.centroid_mu:.4f}",
                        level="warning",
                    )

            # Secondary loop: Filter clusters present in spectra where the compound is not present
            matched_peaks = self._filter_clusters_against_other_spectra(
                matched_peaks, cluster.centroid_mu, compound, delta_mu, **kwargs
            )

            # Store the matched peaks for the compound
            if matched_peaks:
                common_peaks[compound] = matched_peaks
                self.log(
                    f"Found {len(matched_peaks)} common peak clusters for compound {compound}.",
                    level="ok",
                )
            else:
                self.log(
                    f"No common peaks found for compound {compound}. The compound may have no characteristic peaks.",
                    level="warning",
                )

            self.log(f"Finished processing compound: {compound}", indent="exit")

        # Save common peaks to the instance for access
        self.common_peaks = common_peaks
        self.log("Peak matching process completed.", level="ok", indent="exit")

    def _cluster_peaks(
        self, all_peaks: List[VoigtPeakIdentification], delta_mu: float
    ) -> List[PeakCluster]:
        """
        Clusters peaks based on their `mu` values within a specified tolerance.

        Args:
            all_peaks (List[VoigtPeakIdentification]): List of all peaks from all spectra.
            delta_mu (float): The maximum difference in `mu` values for peaks to be considered matching.

        Returns:
            List[PeakCluster]: A list of clusters, each being a PeakCluster object.
        """
        self.log("Clustering peaks based on mu values.", indent="enter")
        # Sort all peaks by their mu values
        all_peaks.sort(key=lambda peak: peak.mu)
        clusters: List[PeakCluster] = []

        for peak in all_peaks:
            found_cluster = False
            for cluster in clusters:
                if abs(peak.mu - cluster.centroid_mu) <= delta_mu:
                    cluster.add_peak(peak)
                    found_cluster = True
                    break
            if not found_cluster:
                # Create a new cluster with this peak
                clusters.append(PeakCluster(peaks=[peak]))

        self.log(f"Formed {len(clusters)} clusters.", indent="exit")
        return clusters

    def _filter_clusters_against_other_spectra(
        self,
        matched_peaks: List[List[VoigtPeakIdentification]],
        cluster_mu: float,
        compound: str,
        delta_mu: float,
        **kwargs: Any,
    ) -> List[List[VoigtPeakIdentification]]:
        """
        Filters clusters by checking if they are present in spectra where the compound is not present.

        Args:
            matched_peaks (List[List[VoigtPeakIdentification]]): Matched peaks for the compound.
            cluster_mu (float): The centroid mu of the cluster.
            compound (str): The compound being processed.
            delta_mu (float): The delta mu used for matching.
            **kwargs: Additional arguments for peak similarity checks.

        Returns:
            List[List[VoigtPeakIdentification]]: Filtered list of matched peaks.
        """
        self.log(
            "Filtering clusters against spectra where the compound is not present.",
            indent="enter",
        )
        delta_mu = kwargs.get("delta_mu_filter", delta_mu)
        # Get the filenames of spectra where the compound is not present
        not_compound_df = self.conc_df[
            ~self.conc_df["filename"].isin(self.sub_df_dict[compound]["filename"])
        ]
        not_compound_files = not_compound_df["filename"].tolist()

        # Collect peaks from spectra where the compound is not present
        not_compound_spectra_peaks = []
        for filename in not_compound_files:
            if filename in self.models:
                peaks = self.models[filename]
                # Annotate each peak with its filename
                for peak in peaks:
                    peak.filename = filename
                not_compound_spectra_peaks.extend(peaks)

        # Filter out clusters that appear in spectra without the compound
        filtered_matched_peaks: List[List[VoigtPeakIdentification]] = []
        for combination in matched_peaks:
            cluster_mu = sum(p.mu for p in combination) / len(combination)
            # Check if this cluster appears in not_compound_spectra_peaks
            cluster_present_in_not_compound = False
            for peak in not_compound_spectra_peaks:
                if abs(peak.mu - cluster_mu) <= delta_mu:
                    # Use peaks_same for precise comparison
                    mode = kwargs.get("similarity_mode", "mu_only")
                    similarity_score = self.peak_same(
                        peak, combination[0], mode=mode, **kwargs
                    )
                    if similarity_score >= 0.8:
                        cluster_present_in_not_compound = True
                        break
                if cluster_present_in_not_compound:
                    break
            if not cluster_present_in_not_compound:
                filtered_matched_peaks.append(combination)
            else:
                self.log(
                    f"Cluster at mu ~ {cluster_mu:.4f} appears in spectra without {compound}. Excluding it.",
                )
        self.log("Filtering complete.", level="ok", indent="exit")
        return filtered_matched_peaks

    def _collect_peaks_per_spectrum(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, List[VoigtPeakIdentification]], List[str]]:
        """
        Collects peaks per spectrum for a given compound's sub-dataframe.

        Args:
            df (pd.DataFrame): The sub-dataframe for a compound.

        Returns:
            Tuple[Dict[str, List[VoigtPeakIdentification]], List[str]]: A dictionary mapping filenames to lists of peaks,
            and a list of filenames (spectra_files).
        """
        spectra_peaks: Dict[str, List[VoigtPeakIdentification]] = {}
        spectra_files: List[str] = []

        for _, row in df.iterrows():
            filename = row["filename"]
            if filename in self.models:
                peaks = self.models[filename]
                # Annotate each peak with its filename
                for peak in peaks:
                    peak.filename = filename
                spectra_peaks[filename] = peaks
                spectra_files.append(filename)
                self.log(f"Collected {len(peaks)} peaks from file: {filename}")
            else:
                self.log(f"No model found for file: {filename}", level="warning")

        return spectra_peaks, spectra_files

    def _select_best_peak_combination(
        self,
        cluster: PeakCluster,
        spectra_files: List[str],
        **kwargs: Any,
    ) -> Optional[List[VoigtPeakIdentification]]:
        """
        For a given cluster, selects the best combination of one peak per spectrum that maximizes the total similarity score.

        Args:
            cluster PeakCluster: The cluster of peaks.
            spectra_files (List[str]): List of spectra filenames.
            **kwargs: Additional arguments passed to `peaks_same` for peak similarity checks.

        Returns:
            Optional[List[VoigtPeakIdentification]]: The best combination of peaks (one per spectrum), or None if no valid combination found.
        """
        # Build a mapping from spectrum to peaks in this cluster
        spectrum_to_peaks: Dict[str, List[VoigtPeakIdentification]] = {}
        for peak in cluster.peaks:
            spectrum_to_peaks.setdefault(peak.filename, []).append(peak)

        # Check if we have at least one peak per spectrum
        if not set(spectra_files).issubset(spectrum_to_peaks.keys()):
            cluster_mu = sum(p.mu for p in cluster) / len(cluster)
            self.log(
                f"Cluster at mu ~ {cluster_mu:.4f} does not have peaks from all spectra.",
                level="warning",
            )
            return None

        # Generate all possible combinations of one peak per spectrum
        peaks_per_spectrum = [spectrum_to_peaks[filename] for filename in spectra_files]
        num_combinations = 1
        for peaks_list in peaks_per_spectrum:
            num_combinations *= len(peaks_list)
        cluster_mu = cluster.centroid_mu
        self.log(
            f"Evaluating {num_combinations} combinations for cluster at mu ~ {cluster_mu:.4f}"
        )

        # Limit combinations if too many (optional, based on your performance needs)
        max_combinations = 10000  # Adjust as needed
        if num_combinations > max_combinations:
            self.log(
                f"Too many combinations ({num_combinations}). Skipping this cluster.",
                level="warning",
            )
            return None

        all_combinations = itertools.product(*peaks_per_spectrum)

        # For each combination, compute the total similarity score
        best_combination = None
        best_total_similarity = -1
        for combination in all_combinations:
            combination_list = list(combination)
            # Compute total similarity score as sum of pairwise similarities
            total_similarity = self._compute_total_similarity(
                combination_list, **kwargs
            )
            if total_similarity > best_total_similarity:
                best_total_similarity = total_similarity
                best_combination = combination_list

        if best_combination is not None:
            self.log(
                f"Best combination found with average similarity {best_total_similarity:.2f}"
            )
            return best_combination
        else:
            self.log("No valid peak combination found.", level="warning")
            return None

    def _compute_total_similarity(
        self, peaks: List[VoigtPeakIdentification], **kwargs: Any
    ) -> float:
        """
        Computes the average similarity score for a list of peaks.

        Args:
            peaks (List[VoigtPeakIdentification]): The list of peaks (one per spectrum).
            **kwargs: Additional arguments passed to `peaks_same` for peak similarity checks.

        Returns:
            float: The average similarity score between all pairs of peaks.
        """
        total_similarity = 0.0
        num_pairs = 0
        for i in range(len(peaks)):
            for j in range(i + 1, len(peaks)):
                mode = kwargs.get("similarity_mode", "mu_only")
                similarity = self.peaks_same(peaks[i], peaks[j], mode=mode, **kwargs)
                total_similarity += similarity
                num_pairs += 1
        if num_pairs > 0:
            average_similarity = total_similarity / num_pairs
            return average_similarity
        else:
            return 0.0

    def fit_and_filter_peaks(
        self, r2_threshold: float = 0.8, p_value_threshold: float = 0.05, **kwargs: Any
    ) -> None:
        """
        Performs linear fitting of amplitude and area vs. concentration for each matched peak set.
        Only keeps sets of peaks with good fit quality based on R^2 or p-value criteria.

        Args:
            r2_threshold (float): Minimum R^2 value to consider the fit acceptable.
            p_value_threshold (float): Maximum p-value to consider the fit significant.

        Returns:
            None: Results are stored directly in `self.filtered_peak_fits`, organized by compound as lists of `VoigtLinearFitResult` objects.
        """
        self.log("Starting peak fitting and filtering.", indent="enter")
        # Ensure common peaks have been identified
        if not hasattr(self, "common_peaks"):
            self.log(
                "Common peaks not found. Run peak_matching first.",
                level="error",
                indent="exit",
            )
            raise ValueError("Common peaks not found. Run peak_matching first.")

        # Initialize a dictionary to store filtered fit results for each compound
        filtered_peak_fits = {}

        # Process each compound's common peaks
        for compound, matched_peaks_list in self.common_peaks.items():
            self.log(f"Processing compound: {compound}", indent="enter")
            compound_fit_results = []

            # Retrieve concentration values from the sub-dataframe
            compound_df = self.sub_df_dict[compound]
            concentrations = compound_df[f"Conc_{compound}"].values

            # Iterate over each matched peak set
            for matched_peaks in matched_peaks_list:
                amplitudes = [peak.A for peak in matched_peaks]
                areas = [peak.area for peak in matched_peaks]
                mu_value = matched_peaks[
                    0
                ].mu  # Assume mu is consistent across matched peaks

                # Ensure we have data for each file in the sub-dataframe
                if len(concentrations) == len(amplitudes) == len(areas):
                    # Perform linear regression for amplitude vs. concentration
                    if len(concentrations) == 1:
                        # Constrain the linear fit to pass through (0, 0)
                        slope_amp = amplitudes[0] / concentrations[0]
                        intercept_amp = 0.0
                        r_value_amp = 1.0  # Perfect correlation for a single point constrained through origin
                        p_value_amp = (
                            0.0  # Meaningless but conventionally 0 for perfect fit
                        )

                        slope_area = areas[0] / concentrations[0]
                        intercept_area = 0.0
                        r_value_area = 1.0
                        p_value_area = 0.0
                    else:
                        # Perform linear regression for amplitude vs. concentration
                        (
                            slope_amp,
                            intercept_amp,
                            r_value_amp,
                            p_value_amp,
                            _,
                        ) = linregress(concentrations, amplitudes)

                        # Perform linear regression for area vs. concentration
                        (
                            slope_area,
                            intercept_area,
                            r_value_area,
                            p_value_area,
                            _,
                        ) = linregress(concentrations, areas)

                    # Check fit quality for both amplitude and area
                    if (
                        r_value_amp**2 >= r2_threshold
                        and p_value_amp <= p_value_threshold
                        and r_value_area**2 >= r2_threshold
                        and p_value_area <= p_value_threshold
                    ):
                        # Create a VoigtLinearFitResult instance for peaks that pass the threshold criteria
                        fit_result = VoigtLinearFitResult(
                            peak_set=matched_peaks,
                            slope_amplitude=slope_amp,
                            intercept_amplitude=intercept_amp,
                            r_value_amplitude=r_value_amp,
                            p_value_amplitude=p_value_amp,
                            slope_area=slope_area,
                            intercept_area=intercept_area,
                            r_value_area=r_value_area,
                            p_value_area=p_value_area,
                        )
                        compound_fit_results.append(fit_result)
                        self.log(
                            f"Peak at mu: {mu_value} passed filtering criteria.",
                            indent="enter",
                        )
                        self.log(
                            f"Fitting results: {fit_result}", level="ok", indent="exit"
                        )
                    else:
                        self.log(
                            f"Peak at mu: {mu_value} did not meet filtering criteria.",
                            level="warning",
                        )
                else:
                    self.log(
                        f"Inconsistent data lengths for peak at mu: {mu_value}. Skipping.",
                        level="warning",
                    )

            # Store the compound's fit results if any passed the filtering criteria
            if compound_fit_results:
                filtered_peak_fits[compound] = compound_fit_results
                self.log(
                    f"Compound {compound} has {len(compound_fit_results)} peaks after filtering."
                )
            else:
                self.log(
                    f"No peaks passed filtering for compound {compound}.",
                    level="warning",
                )

            self.log(f"Finished processing compound: {compound}", indent="exit")

        # Save filtered peak fits for access
        self.filtered_peak_fits = filtered_peak_fits
        self.log("Peak fitting and filtering completed.", level="ok", indent="exit")

    def save_data(self, base_directory: str = "filtered_peak_fits"):
        """
        Saves the filtered peak fits data in a structured format:
        - Main DataFrame with each compound's fitting summary and paths to sub-DataFrames.
        - Sub-DataFrames for each compound with peak fitting details.
        - Separate files for each peak set in each compound's directory.

        Parameters:
            base_directory (str): Directory to save the data files. Default is "filtered_peak_fits".

        Returns:
            None
        """

        # Create main directory for saving results
        os.makedirs(base_directory, exist_ok=True)
        self.log(
            f"Saving filtered peak fits data in {base_directory}...", indent="enter"
        )

        # Initialize a list to collect compound summaries for the main DataFrame
        compound_summary_data = []

        # Process each compound in filtered_peak_fits
        for compound, fit_results in self.filtered_peak_fits.items():
            self.log(f"Saving data for compound: {compound}")
            # Create a subdirectory for each compound
            compound_dir = os.path.join(base_directory, "subdfs", compound)
            os.makedirs(compound_dir, exist_ok=True)

            # Create a list to store each row for the compound's sub-DataFrame
            compound_fit_data = []

            # Iterate over each fit result for the compound
            for idx, fit_result in enumerate(fit_results):
                # Define the path for each peak set DataFrame
                peak_set_file = os.path.join(
                    compound_dir, "peak_sets", f"peak_set_{idx}.csv"
                )
                os.makedirs(os.path.dirname(peak_set_file), exist_ok=True)

                # Save each peak set as a DataFrame file
                peak_set_df = pd.DataFrame(
                    [peak.peak_to_df_row for peak in fit_result.peak_set]
                )
                peak_set_df.to_csv(peak_set_file, index=False)

                # Collect fit result data for the compound sub-DataFrame
                fit_data_row = {
                    "peak_mu": fit_result.mu,
                    "slope_amplitude": fit_result.slope_amplitude,
                    "intercept_amplitude": fit_result.intercept_amplitude,
                    "r_value_amplitude": fit_result.r_value_amplitude,
                    "p_value_amplitude": fit_result.p_value_amplitude,
                    "slope_area": fit_result.slope_area,
                    "intercept_area": fit_result.intercept_area,
                    "r_value_area": fit_result.r_value_area,
                    "p_value_area": fit_result.p_value_area,
                    "peak_set_path": os.path.relpath(
                        peak_set_file, start=base_directory
                    ),  # Relative path for reference
                }
                compound_fit_data.append(fit_data_row)

            # Save the compound's fit data as a DataFrame in its subdirectory
            compound_fit_df = pd.DataFrame(compound_fit_data)
            compound_fit_file = os.path.join(
                compound_dir, f"{compound}_fit_results.csv"
            )
            compound_fit_df.to_csv(compound_fit_file, index=False)

            # Collect compound summary data for the main DataFrame
            compound_summary_data.append(
                {
                    "compound": compound,
                    "fit_results_path": os.path.relpath(
                        compound_fit_file, start=base_directory
                    ),
                }
            )

        # Create the main DataFrame with compound summaries and save it
        main_summary_df = pd.DataFrame(compound_summary_data)
        main_summary_file = os.path.join(
            base_directory, "filtered_peak_fits_summary.csv"
        )
        main_summary_df.to_csv(main_summary_file, index=False)

        # Log completion
        self.log(
            f"Filtered peak fits data saved successfully in {base_directory}.",
            level="ok",
            indent="reset",
        )

    def load_fits(self, base_directory: str = "filtered_peak_fits"):
        """
        Loads the filtered peak fits data saved in a structured format:
        - Loads the main summary DataFrame to access each compound's fitting summary and paths to sub-DataFrames.
        - Loads each compound's sub-DataFrame with fit results.
        - Reconstructs each peak set from separate files in each compound's directory.

        Parameters:
            base_directory (str): Directory where the data files are saved. Default is "filtered_peak_fits".

        Returns:
            None
        """

        # Define paths
        main_summary_file = os.path.join(
            base_directory, "filtered_peak_fits_summary.csv"
        )

        # Check that the main summary file exists
        if not os.path.exists(main_summary_file):
            raise FileNotFoundError(
                f"Main summary file not found at {main_summary_file}"
            )

        # Load the main summary DataFrame
        main_summary_df = pd.read_csv(main_summary_file)

        # Initialize the dictionary to store the loaded fit results
        self.filtered_peak_fits = {}

        # Process each compound in the main summary DataFrame
        for _, row in main_summary_df.iterrows():
            compound = row["compound"]
            fit_results_path = os.path.join(base_directory, row["fit_results_path"])

            # Check that the compound fit results file exists
            if not os.path.exists(fit_results_path):
                raise FileNotFoundError(
                    f"Fit results file for compound {compound} not found at {fit_results_path}"
                )

            # Load the compound's fit results DataFrame
            compound_fit_df = pd.read_csv(fit_results_path)

            # Initialize a list to store VoigtLinearFitResult objects for the compound
            compound_fit_results = []

            # Process each fit result row for the compound
            for _, fit_row in compound_fit_df.iterrows():
                # Load each peak set DataFrame
                peak_set_path = os.path.join(base_directory, fit_row["peak_set_path"])
                if not os.path.exists(peak_set_path):
                    raise FileNotFoundError(
                        f"Peak set file not found at {peak_set_path}"
                    )

                peak_set_df = pd.read_csv(peak_set_path)

                # Reconstruct VoigtPeakIdentification objects from the peak set DataFrame
                peak_set = [
                    VoigtPeakIdentification(df_row=row)
                    for _, row in peak_set_df.iterrows()
                ]

                # Recreate the VoigtLinearFitResult object with the loaded data
                fit_result = VoigtLinearFitResult(
                    peak_set=peak_set,
                    slope_amplitude=fit_row["slope_amplitude"],
                    intercept_amplitude=fit_row["intercept_amplitude"],
                    r_value_amplitude=fit_row["r_value_amplitude"],
                    p_value_amplitude=fit_row["p_value_amplitude"],
                    slope_area=fit_row["slope_area"],
                    intercept_area=fit_row["intercept_area"],
                    r_value_area=fit_row["r_value_area"],
                    p_value_area=fit_row["p_value_area"],
                )

                # Add the fit result to the list for this compound
                compound_fit_results.append(fit_result)

            # Store the list of fit results for this compound in the main dictionary
            self.filtered_peak_fits[compound] = compound_fit_results

        # Log completion
        self.log(
            "Filtered peak fits data loaded successfully.", indent="reset", level="ok"
        )
