"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import os
import copy
from typing import List, Any, Dict, Optional, Type, Tuple, Union

import numpy as np
import pandas as pd
import torch

from robrains.base_classes import BaseLoggedClass

from robrains.utils import FloatWithError


class baseMLBackend(BaseLoggedClass):
    _ML_primed: bool = False
    _com_primed: bool = False
    input_default: dict = {}
    parameters: dict = {}
    run_index: int = None

    @property
    def res_df(self):
        if hasattr(self, "results_df") and isinstance(self.results_df, pd.DataFrame):
            return self.results_df.copy()
        else:
            return None

    @property
    def primed(self):
        return self._ML_primed and self._com_primed

    def ensure_writable(self, tensor):
        if tensor.is_leaf:
            try:
                tensor[0] = tensor[0]  # Try writing to the tensor
            except RuntimeError:
                return tensor.clone().detach()  # Create a writable copy of the tensor
        return tensor

    def __init__(self):
        self.parameters = self.input_defaults.copy()

    def ML_prime(
        self,
        ML_parameters: List["MLParameter"],
        targets: List[Any],
        save_path: str,
    ) -> None:
        """
        Initialize and prime the machine-learning optimisation process.

        :param ML_parameters: List of MLParameter instances built by the frontend.
        :param targets: List of target identifiers for optimisation.
        :param save_path: File system path where results and checkpoints will be saved.
        """
        # Log initial priming call
        self.log_mssg(f"Priming ML with:")
        self.log_mssg(f" - {len(ML_parameters)} ML parameters", indent=1)
        for param in ML_parameters:
            self.log_mssg(f" - {param.name}", indent=2)
        self.log_mssg(f" - {len(targets)} targets", indent=1)
        for target in targets:
            self.log_mssg(f" - {target}", indent=2)
        self.log_mssg(f" - Save path: {save_path}", indent=1)

        self.path = save_path

        # If already primed, regenerate checks and exit
        if self._ML_primed:
            self.log_mssg(
                "ML already primed; regenerating check dictionary and exiting.",
                indent=1,
            )
            self._generate_check_dict()
            return

        # Assign parameters and determine embedding type
        self.ML_parameters = tuple(ML_parameters)
        self.targets = tuple(targets)

        high_categorical = any(
            "HighCategoricalEmbeddings" in base.__qualname__
            for base in self.__class__.__mro__
        )
        self.log_mssg(
            f"High categorical embeddings detected: {high_categorical}",
            indent=1,
        )

        # Generate translators for each parameter
        for param in self.ML_parameters:
            param.high_categorical = high_categorical
            param.generate_translators()
        self.log_mssg("Generated translators for ML parameters.", level="ok", indent=1)

        # Create or refresh the check dictionary
        self._generate_check_dict()
        self.log_mssg("Generated check dictionary.", level="ok", indent=1)

        # Restore run index if previous results exist
        if hasattr(self, "results_df"):
            self.run_index = self.results_df["run_index"].max()
            self.log_mssg(
                f"Found existing results_df, set run_index to {self.run_index}.",
                level="ok",
                indent=1,
            )

        # Initialize training placeholders and mark as primed
        self.train_y = None
        self.train_x = None
        self._ML_primed = True
        self.log_mssg("ML priming completed successfully.", level="ok")

    def ML_prime_from_json(
        self,
        json_data: Dict[str, Any],
        ml_parameters: Optional[List["MLParameter"]],
        exp_path: str,
    ) -> None:
        """
        Initialize optimizer state from a JSON configuration.

        :param json_data: Dictionary with keys 'added_keys', 'parameters', 'results_df', 'utilities_df', 'targets', and 'path'.
        :param ml_parameters: Optional list of MLParameter instances for priming, passed to ML_prime if provided.
        :param exp_path: Base path where experiment files (dataframes) are stored.
        """
        self.log_mssg("Initializing optimizer from JSON data.", level="ok")

        # Restore any added keys
        if "added_keys" in json_data:
            self.added_keys = json_data["added_keys"]  # type: ignore[attr-defined]
            self.log_mssg(f"Restored added_keys: {self.added_keys}", indent=1)

        # Validate and apply saved parameters
        if "parameters" in json_data:
            for key, value in json_data["parameters"].items():
                self.validate_and_update(key, value)
                self.log_mssg(f"Updated parameter '{key}' to {value}.", indent=1)

        # Load previous results DataFrame
        if "results_df" in json_data:
            self.log_mssg(
                f"Loading results_df from path: {json_data['results_df']}",
                indent=1,
            )
            rel_path = json_data["results_df"].split("]", 1)[1]
            full_path = os.path.join(exp_path, rel_path)
            self.results_df = pd.read_csv(full_path)
            self.log_mssg(
                f"Loaded results_df with {len(self.results_df)} rows.",
                level="ok",
                indent=2,
            )

        # Load previous utilities DataFrame
        if "utilities_df" in json_data:
            self.log_mssg(
                f"Loading utilities_df from path: {json_data['utilities_df']}",
                indent=1,
            )
            rel_util_path = json_data["utilities_df"].split("]", 1)[1]
            full_util_path = os.path.join(exp_path, rel_util_path)
            self.utilities_df = pd.read_csv(full_util_path)
            self.log_mssg(
                f"Loaded utilities_df with {len(self.utilities_df)} rows.",
                level="ok",
                indent=2,
            )

        # Prime ML if parameters and targets are provided
        has_ml_args = (
            ml_parameters is not None and "targets" in json_data and "path" in json_data
        )
        if has_ml_args:
            self.log_mssg("Priming ML from JSON configuration.", level="ok")
            self.ML_prime(ml_parameters, json_data["targets"], json_data["path"])

    def get_data_to_save(self) -> Dict[str, Any]:
        """
        Gather and return all relevant attributes for saving to a JSON configuration.

        :return: Dictionary containing any of the following keys if present:
            - 'parameters': Saved parameter configurations.
            - 'added_keys': Any additional keys tracked.
            - 'results_df': DataFrame rows with status != 'submitted'.
            - 'utilities_df': Saved utilities DataFrame.
            - 'ML_parameters': List of ML parameter names with '_ml' suffix.
            - 'targets': Tuple of target identifiers.
            - 'path': Experiment save path.
        """
        # Log start of data collection
        self.log_mssg("Collecting data for JSON save.", level="ok")

        data: Dict[str, Any] = {}

        # Save parameters if present
        if hasattr(self, "parameters"):
            data["parameters"] = self.parameters
            self.log_mssg("Included 'parameters'.", indent=1)
            self.log_mssg(self.parameters, level="none", indent=2)

        # Save added keys
        if hasattr(self, "added_keys"):
            data["added_keys"] = self.added_keys
            self.log_mssg("Included 'added_keys'.", indent=1)
            self.log_mssg(self.added_keys, level="none", indent=2)

        # Save filtered results DataFrame
        if hasattr(self, "results_df"):
            filtered = self.results_df[self.results_df["status"] != "submitted"]
            data["results_df"] = filtered
            self.log_mssg(
                f"Included 'results_df' with {len(filtered)} rows (status != 'submitted').",
                indent=1,
            )

        # Save utilities DataFrame
        if hasattr(self, "utilities_df"):
            data["utilities_df"] = self.utilities_df
            self.log_mssg("Included 'utilities_df'.", indent=1)
            self.log_mssg(self.utilities_df, level="none", indent=2)

        # Save ML parameter names
        if hasattr(self, "ML_parameters"):
            ml_names = [f"{param.name}_ml" for param in self.ML_parameters]
            data["ML_parameters"] = ml_names
            self.log_mssg(f"Included 'ML_parameters': {ml_names}.", indent=1)

        # Save targets and path
        if hasattr(self, "targets"):
            data["targets"] = self.targets
            self.log_mssg("Included 'targets'.", indent=1)
            self.log_mssg(self.targets, level="none", indent=2)

        if hasattr(self, "path"):
            data["path"] = self.path
            self.log_mssg(f"Included 'path': {self.path}.", indent=1)
            self.log_mssg(self.path, level="none", indent=2)

        # Log completion
        self.log_mssg("Data collection for JSON save completed.", level="ok")
        return data

    @classmethod
    def from_json(
        cls,
        json_data: Dict[str, Any],
        ml_parameters: Optional[List["MLParameter"]] = None,
        **kwargs: Any,
    ):
        """
        Instantiate and initialize the class from a JSON configuration.

        :param json_data: Dictionary of data read from JSON, with keys such as 'parameters', 'exp_path', etc.
        :param ml_parameters: Optional list of MLParameter instances to use for ML priming.
        :param kwargs: Additional keyword arguments passed to `com_prime` when ML parameters are provided.
        :return: An instance of the initialized class.

        :raises ValueError: If initialization from JSON fails.
        """
        cls.log_mssg(cls, "Creating instance from JSON data.", level="ok")
        try:
            instance = cls()
            # If ML parameters are given, perform both common and ML priming
            if ml_parameters is not None:
                instance.com_prime(**kwargs)
                cls.log_mssg(cls, "Primed common settings via com_prime.", indent=1)
                instance.ML_prime_from_json(
                    json_data=json_data,
                    ml_parameters=ml_parameters,
                    exp_path=json_data.get("exp_path", ""),
                )
                cls.log_mssg(cls, "Completed ML priming from JSON.", indent=1)
            # Otherwise, only update parameters from JSON
            else:
                for key, value in json_data.get("parameters", {}).items():
                    instance.validate_and_update(key, value)
                    cls.log_mssg(cls, f"Set parameter '{key}' to {value}.", indent=1)
            cls.log_mssg(cls, "Instance creation from JSON successful.", level="ok")
            return instance
        except Exception as e:
            cls.log_mssg(cls, f"Error initializing from JSON: {e}", level="error")
            raise ValueError(f"Failed to initialize {cls.__name__} from JSON: {e}")

    def set_status(
        self,
        run_index: int,
        status: str,
    ) -> None:
        """
        Update the status of a given run in the results DataFrame.

        :param run_index: The index of the run to update.
        :param status: The new status string to assign.
        :raises KeyError: If 'results_df' does not exist or the run_index is not found.
        """
        try:
            # Log intent to update status
            self.log_mssg(f"Updating status for run_index={run_index} to '{status}'.")

            # Perform the status update
            if not hasattr(self, "results_df"):
                raise KeyError("'results_df' attribute not found on instance.")

            mask = self.results_df["run_index"] == run_index
            if not mask.any():
                raise KeyError(f"No entries found for run_index {run_index}.")

            self.results_df.loc[mask, "status"] = status

            # Confirm successful update
            self.log_mssg(
                f"Status for run_index {run_index} set to '{status}' successfully.",
                level="ok",
                indent=1,
            )
        except Exception as e:
            # Log and propagate error
            self.log_mssg(
                f"Error updating status for run_index {run_index}: {e}", level="error"
            )
            raise

    def validate_in_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows based on 'run_index', keeping the last occurrence.

        :param df: DataFrame containing a 'run_index' column.
        :return: DataFrame with unique 'run_index' values, duplicates removed.
        """
        # Log original and cleaned sizes
        self.log_mssg(
            f"Validating DataFrame: original size {df.shape}, checking for duplicates.",
        )
        self.log_mssg(f"{df.columns.tolist()}", indent=1)
        original_count = len(df)
        df_clean = df.drop_duplicates(subset=["run_index"], keep="last")
        dropped = original_count - len(df_clean)
        self.log_mssg(
            f"Removed {dropped} duplicate row(s) from DataFrame based on 'run_index'.",
            level="ok",
            indent=1,
        )
        return df_clean

    def set_vial(
        self,
        run_index: int,
        vial: Any,
    ) -> None:
        """
        Update the vial identifier for a given run in the results DataFrame.

        :param run_index: The index of the run to update.
        :param vial: The vial identifier to assign (string or other).
        :raises KeyError: If 'results_df' is missing or the run_index is not found.
        """
        try:
            # Log the intended update
            self.log_mssg(f"Updating vial for run_index={run_index} to '{vial}'.")

            # Ensure results_df exists
            if not hasattr(self, "results_df"):
                raise KeyError("'results_df' attribute not found on instance.")

            mask = self.results_df["run_index"] == run_index
            if not mask.any():
                raise KeyError(f"No entries found for run_index {run_index}.")

            # Perform update
            self.results_df.loc[mask, "vial_idx"] = vial

            # Confirm successful update
            self.log_mssg(
                f"Vial for run_index {run_index} set to '{vial}' successfully.",
                level="ok",
                indent=1,
            )
        except Exception as e:
            # Log and re-raise the error
            self.log_mssg(
                f"Error updating vial for run_index {run_index}: {e}", level="error"
            )
            raise

    def _add_savename_to_df(
        self,
        savename: str,
        run_index: int,
    ) -> None:
        """
        Add or update the save filename for a specific run in the results DataFrame.

        :param savename: The filename to associate with the run.
        :param run_index: The index of the run to update.
        :raises KeyError: If 'results_df' is missing or the run_index is not found.
        """
        try:
            # Log intent to add savename
            self.log_mssg(f"Adding save name '{savename}' for run_index={run_index}.")

            # Ensure results_df exists
            if not hasattr(self, "results_df"):
                raise KeyError("'results_df' attribute not found on instance.")

            # Initialize column if it does not exist
            if "file_save_name" not in self.results_df.columns:
                self.results_df["file_save_name"] = ""
                self.log_mssg("Created 'file_save_name' column.", indent=1)

            # Check run_index exists
            mask = self.results_df["run_index"] == run_index
            if not mask.any():
                raise KeyError(f"No entries found for run_index {run_index}.")

            # Perform update
            self.results_df.loc[mask, "file_save_name"] = savename
            self.log_mssg(
                f"File save name set for run_index {run_index}.", level="ok", indent=1
            )
        except Exception as e:
            # Log and propagate error
            self.log_mssg(
                f"Error adding save name for run_index {run_index}: {e}", level="error"
            )
            raise

    def update_attempt(
        self,
        run_index: int,
    ) -> None:
        """
        Increment the attempt count for a specific run in the results DataFrame.

        :param run_index: The index of the run whose attempts counter will be incremented.
        :raises KeyError: If 'results_df' is missing, the 'attempts' column is absent, or the run_index is not found.
        """
        try:
            # Log intent to update attempts
            self.log_mssg(f"Incrementing attempts for run_index={run_index}.")

            # Ensure results_df exists
            if not hasattr(self, "results_df"):
                raise KeyError("'results_df' attribute not found on instance.")

            # Initialize 'attempts' column if missing
            if "attempts" not in self.results_df.columns:
                self.results_df["attempts"] = 0
                self.log_mssg(
                    "Created 'attempts' column with default 0 values.",
                    level="ok",
                    indent=1,
                )

            # Ensure run_index exists
            mask = self.results_df["run_index"] == run_index
            if not mask.any():
                raise KeyError(f"No entries found for run_index {run_index}.")

            # Perform increment
            self.results_df.loc[mask, "attempts"] += 1

            # Confirm successful update
            new_val = self.results_df.loc[mask, "attempts"].iloc[0]
            self.log_mssg(
                f"Attempts for run_index {run_index} incremented to {new_val}.",
                level="ok",
                indent=1,
            )
        except Exception as e:
            # Log and re-raise error
            self.log_mssg(
                f"Error incrementing attempts for run_index {run_index}: {e}",
                level="error",
            )
            raise

    def _from_df(
        self,
        df: pd.DataFrame,
        check_nan: bool = True,
        keywords: List[str] = ["finished"],
        update_df: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert run results DataFrame into feature and target tensors, preserving original DataFrame if requested.

        :param df: DataFrame including parameter, target, status, vial_idx, run_index, and file_save_name columns.
        :param check_nan: Raise if any NaN in feature matrix.
        :param keywords: Status values to keep (e.g. ['finished']).
        :param update_df: If True, restore self.results_df to original df on exit.
        :return: (tensor_x, tensor_y, tensor_y_var)
        :raises ValueError: On NaNs in feature data.
        """
        # Log start
        self.log_mssg(
            f"Starting DataFrame to tensor conversion: {df.shape[0]} rows, {df.shape[1]} cols",
        )
        # 1) Deep copy original for potential restoration
        df_original = df.copy(deep=True)

        # 2) Filter by status
        if "status" in df_original.columns:
            before = len(df_original)
            df_filtered = df_original[df_original["status"].isin(keywords)].copy()
            kept = len(df_filtered)
            self.log_mssg(
                f"Filtered status: kept {kept}/{before} rows with keywords {keywords}",
                indent=1,
            )
        else:
            df_filtered = df_original.copy()
            self.log_mssg("No status column to filter", indent=1)

        # 3) Extract and drop metadata columns from features
        df_features = df_filtered.copy()
        for meta in ("vial_idx", "file_save_name"):
            if meta in df_features.columns:
                n_nan = df_features[meta].isnull().sum()
                if n_nan > 0:
                    df_features.drop(columns=[meta], inplace=True)
                    self.log_mssg(
                        f"Dropped '{meta}' with {n_nan} NaN entries",
                        indent=1,
                    )
                else:
                    _ = df_features.pop(meta)
                    self.log_mssg(f"Extracted '{meta}' before drop", indent=1)

        # 4) Sort and drop run_index
        df_features.sort_values("run_index", inplace=True)
        df_features.drop(columns=["run_index"], inplace=True)
        self.log_mssg("Sorted by 'run_index' and dropped the column", indent=1)

        # 5) Drop status, predicted and variance columns from features
        if "status" in df_features.columns:
            df_features.drop(columns=["status"], inplace=True)
            self.log_mssg("Dropped 'status' column", indent=1)
        for t in self.targets:
            for col in (f"{t}_predicted", f"{t}_variance"):  # predicted and variance
                if col in df_features.columns:
                    df_features.drop(columns=[col], inplace=True)
                    self.log_mssg(f"Dropped '{col}' column", indent=1)

        # 6) NaN check
        has_nan = df_features.isnull().values.any()
        self.log_mssg(f"Feature DataFrame NaN present: {has_nan}", indent=1)
        if check_nan and has_nan:
            self.log_mssg("NaN values found in feature DataFrame", level="error")
            raise ValueError("NaN values found in feature DataFrame")

        # 7) Build feature tensor
        tensor_x_list: List[float] = []
        self.log_mssg("Building feature tensor", indent=1)
        for idx, row in df_features.iterrows():
            self.log_mssg(f"Processing row {idx}", indent=1)
            for param in self.ML_parameters:
                base = param.name.replace("_ml", "")
                # continuous
                if hasattr(param, "translation_continuous"):
                    if f"{base}_conc" in row:
                        raw_c = row[f"{base}_conc"]
                    else:
                        raw_c = row[base]
                    val_c = param.translation_continuous(raw_c)
                    tensor_x_list.append(val_c)
                    self.log_mssg(f"Param {base}: continuous -> {val_c}", indent=2)
                # discrete
                if hasattr(param, "translation_discrete") and base in row:
                    raw_d = row[base]
                    #put the raw_d in the param.discrete_type type:
                    if hasattr(param, "discrete_type"):
                        # Map the string name to the actual Python class
                        type_map = {
                            "str": str,
                            "int": int,
                            "float": float
                        }

                        # Look up which class we need
                        dtype = type_map.get(param.discrete_type)

                        if dtype is None:
                            # If someone set discrete_type to something else, you can choose to error or skip.
                            raise ValueError(
                                f"Unknown discrete_type '{param.discrete_type}' for parameter '{param.name}'")

                        # Only coerce if it's not already an instance of that class
                        if not isinstance(raw_d, dtype):
                            try:
                                raw_d = dtype(raw_d)
                            except (ValueError, TypeError) as e:
                                # Handle conversion failure however you like.
                                # For example, you could log and re-raise:
                                raise ValueError(
                                    f"Failed to convert raw value {raw_d!r} to {param.discrete_type}"
                                ) from e

                    disc = param.translation_discrete(raw_d)
                    if isinstance(disc, (list, tuple, torch.Tensor)):
                        tensor_x_list.extend(list(disc))
                    else:
                        tensor_x_list.append(disc)
                    self.log_mssg(f"Param {base}: discrete -> {disc}", indent=2)
                # fidelity
                if hasattr(param, "translation_fidelity") and base in row:
                    fid = param.translation_fidelity(row[base])
                    tensor_x_list.append(fid)
                    self.log_mssg(f"Param {base}: fidelity -> {fid}", indent=2)
                # task
                if hasattr(param, "translation_task") and base in row:
                    task = param.translation_task(row[base])
                    tensor_x_list.append(task)
                    self.log_mssg(f"Param {base}: task -> {task}", indent=2)

        tensor_x = torch.tensor(tensor_x_list, dtype=torch.float64)
        self.log_mssg(
            f"Built tensor_x of size {tensor_x.numel()}", level="ok", indent=1
        )

        # 8) Build target tensors
        tensor_y = torch.tensor(
            df_filtered[list(self.targets)].values,
            dtype=torch.float64,
        )
        try:
            tensor_y_var = torch.tensor(
                df_filtered[[f"{t}_variance" for t in self.targets]].values,
                dtype=torch.float64,
            )
            self.log_mssg("Built tensor_y_var", level="ok", indent=1)
        except KeyError:
            tensor_y_var = None
            self.log_mssg(
                "No variance columns found; tensor_y_var set to None",
                indent=1,
            )

        # 9) Restore original results_df if requested
        if update_df:
            self.results_df = df_original
            self.log_mssg("Restored original results_df", level="ok", indent=1)

        self.log_mssg("Completed DataFrame to tensor conversion", level="ok")
        return tensor_x, tensor_y, tensor_y_var

    def _extract_tensor_from_result_df(
        self,
        run_index: Union[int, str],
    ) -> torch.Tensor:
        """
        Extract input features as a tensor from a single run in results_df.

        :param run_index: Identifier of the run to extract.
        :return: torch.Tensor of input feature values for the run.
        :raises ValueError: If results_df is missing or no matching run is found.
        """
        # Log start of extraction
        self.log_mssg(f"Extracting tensor for run_index={run_index} from results_df.")

        # Validate existence of results_df
        if not hasattr(self, "results_df"):
            msg = "Attribute 'results_df' not found."
            self.log_mssg(msg, level="error")
            raise ValueError(msg)

        # Filter for the specified run
        df = self.results_df
        row_df = df[df["run_index"] == run_index]
        if row_df.empty:
            msg = f"No row found with run_index {run_index}."
            self.log_mssg(msg, level="error")
            raise ValueError(msg)

        row = row_df.iloc[0]

        # Remove and drop 'status' and 'run_index' if present
        for col in ["status", "run_index"]:
            if col in row_df.columns:
                self.log_mssg(f"Dropping column '{col}' from extraction.", indent=1)

        tensor_vals = []
        self.log_mssg("Extracting tensor values from row.", indent=1)
        for parameter in self.ML_parameters:
            self.log_mssg(f"Processing parameter '{parameter.name}'", indent=2)
            pname = parameter.name
            base = pname.replace("_ml", "")
            phy = parameter.phy_chem
            val_cont = None
            val_disc = None
            val_fid = None
            val_task = None

            # Chemical features
            if phy == "Chemical" and base in row.index:
                value = row[base]
                conc = row.get(f"{base}_conc")
                if conc is None:
                    msg = f"Concentration missing for '{base}'."
                    self.log_mssg(msg, level="error", indent=1)
                    raise ValueError(msg)
                if not getattr(parameter, "constant_valued", False):
                    val_cont = parameter.translation_continuous(conc)
                if not getattr(parameter, "discrete_single_value", False):
                    val_disc = parameter.translation_discrete(value)
                if getattr(parameter, "fidelity_feature", False):
                    val_fid = parameter.translation_fidelity(value)
                if getattr(parameter, "task_feature", False):
                    val_task = parameter.translation_task(value)

            # Physical features
            elif phy == "Physical" and base in row.index:
                value = row[base]
                if hasattr(parameter, "translation_continuous"):
                    val_cont = parameter.translation_continuous(value)
                elif hasattr(parameter, "translation_discrete"):
                    val_disc = parameter.translation_discrete(value)

            # Collect values into tensor list
            for v in (val_cont, val_disc, val_fid, val_task):
                if v is None:
                    continue
                if isinstance(v, list):
                    tensor_vals.extend(v)
                elif isinstance(v, torch.Tensor):
                    tensor_vals.extend(list(v))
                else:
                    tensor_vals.append(v)
            self.log_mssg(
                f"Updated tensor vals (val_cont={val_cont}, val_disc={val_disc}, val_fid={val_fid}, val_task={val_task})",
                indent=2,
            )

        # Convert to torch tensor
        tensor_x = torch.tensor(tensor_vals, dtype=torch.float64)
        self.log_mssg(
            f"Extracted tensor_x with {tensor_x.numel()} elements.", level="ok"
        )
        return tensor_x

    def _to_df(
        self,
        tensor: torch.Tensor,
        run_idx: Optional[int] = None,
        predicted_y: Optional[Union[torch.Tensor, List[torch.Tensor], str]] = None,
    ) -> pd.DataFrame:
        """
        Convert a feature tensor into a new row in `results_df` and return the updated DataFrame.

        :param tensor: 1D torch.Tensor containing feature values in the order defined by `self.check_dict`.
        :param run_idx: Optional run index to update; if None, a new run_index is assigned.
        :param predicted_y: Optional tensor or list of tensors for predictions; if 'recover', existing predictions are reused.
        :return: A copy of the updated `results_df` DataFrame.
        :raises ValueError: If no value can be back-translated for a parameter, or if `results_df` is missing when required.
        """
        try:
            self.log_mssg(
                f"Converting tensor of size {tensor.numel()} to DataFrame row (run_idx={run_idx}).",
            )
            row_data: Dict[str, Any] = {}
            self.log_mssg("Initializing row data dictionary.", indent=1)
            for param in self.ML_parameters:
                self.log_mssg(f"Processing parameter '{param.name}'", indent=2)
                name_ml = param.name
                base = name_ml.replace("_ml", "")
                phy = "Chemical" if param.phy_chem == "Chemical" else "Physical"

                # extract raw values
                val_cont = None
                val_disc = None
                val_fid = None
                val_task = None

                # continuous
                idx_cont = self.check_dict.get(f"{name_ml}_continuous")
                if idx_cont is not None:
                    raw = tensor[int(idx_cont)].item()
                    val_cont = param.back_translation_continuous(raw)

                # discrete
                idx_disc = self.check_dict.get(f"{name_ml}_discrete")
                if idx_disc is not None:
                    if isinstance(idx_disc, int):
                        raw = tensor[int(idx_disc)].item()
                    else:
                        raw = tensor[idx_disc[0] : idx_disc[1]]
                    val_disc = param.back_translation_discrete(raw)

                elif param.discrete_single_value:
                    val_disc = param.discrete[0]

                # fidelity
                idx_fid = self.check_dict.get(f"{name_ml}_fidelity")
                if idx_fid is not None:
                    raw = tensor[int(idx_fid)].item()
                    val_fid = param.back_translation_fidelity(raw)

                # task
                idx_task = self.check_dict.get(f"{name_ml}_task")
                if idx_task is not None:
                    raw = tensor[int(idx_task)].item()
                    val_task = param.back_translation_task(raw)

                # chemical-specific logic
                if phy == "Chemical":
                    # discrete or default
                    if val_disc is not None:
                        row_data[base] = (
                            val_disc
                            if not param.discrete_single_value
                            else param.discrete[0]
                        )
                    # fidelity override
                    if val_fid is not None and val_fid == 0:
                        if (
                            hasattr(self, "results_df")
                            and run_idx is not None
                            and run_idx in self.results_df["run_index"].values
                        ):
                            prev = self.results_df.loc[
                                self.results_df["run_index"] == run_idx, base
                            ].values[0]
                            row_data[base] = prev
                        else:
                            row_data[base] = "LowFidelityOption"
                    # task override
                    if val_task is not None:
                        row_data[base] = val_task
                    # constant valued
                    if (
                        param.constant_valued
                        and param.const_value is not None
                        and val_cont is None
                    ):
                        val_cont = param.const_value
                    # continuous concentration column
                    row_data[f"{base}_conc"] = val_cont

                else:
                    # physical parameters: prefer discrete then continuous
                    if val_disc is not None:
                        row_data[base] = val_disc
                        #task override, as task is "higher order" than discrete
                        if val_task is not None:
                            row_data[base] = val_task

                    elif val_cont is not None:
                        row_data[base] = val_cont
                    else:
                        raise ValueError(
                            f"No back-translated value for parameter '{base}'"
                        )
                self.log_mssg(
                    f"Set '{base}' to {row_data[base]} (cont={val_cont}, disc={val_disc}, fid={val_fid}, task={val_task})",
                    indent=2,
                )
            # Initialize targets columns
            self.log_mssg("Initializing target columns.", indent=1)
            for target in self.targets:
                self.log_mssg(f"Processing target '{target}'", indent=2)
                row_data[target] = np.nan
                if self.parameters.get("Model") == "NoisySingleTaskGP":
                    row_data[f"{target}_variance"] = np.nan
                # Handle predicted_y
                if predicted_y is None:
                    mean, var = None, None
                elif predicted_y == "recover":
                    existing = self.results_df.loc[
                        self.results_df["run_index"] == run_idx, f"{target}_predicted"
                    ].values
                    mean, var = (
                        (existing[0].value, existing[0].error)
                        if existing.size
                        else (None, None)
                    )
                else:
                    # assume list: [mean_tensor, var_tensor]
                    mean = predicted_y[0].flatten()[self.targets.index(target)].item()
                    var = predicted_y[-1].flatten()[self.targets.index(target)].item()

                flo_error = FloatWithError(value=mean, error=var)
                row_data[f"{target}_predicted"] = flo_error
                self.log_mssg(f"Set predicted '{target}': {flo_error}", indent=1)

            # Vial and status
            row_data["vial_idx"] = ""
            row_data["status"] = "written"

            # Determine run_index and attempts
            if hasattr(self, "results_df"):
                if run_idx is not None:
                    row_data["run_index"] = run_idx
                    attempts = self.results_df.loc[
                        self.results_df["run_index"] == run_idx, "attempts"
                    ].values
                    row_data["attempts"] = int(attempts[0]) if attempts.size else 0
                else:
                    row_data["run_index"] = len(self.results_df) + 1
                    row_data["attempts"] = 0
                # Preserve file_save_name if exists
                if "file_save_name" in self.results_df.columns:
                    fs = self.results_df.loc[
                        self.results_df["run_index"] == row_data["run_index"],
                        "file_save_name",
                    ].values
                    row_data["file_save_name"] = fs[0] if fs.size else ""
                # Insert or replace row
                self.results_df.loc[row_data["run_index"] - 1] = row_data
            else:
                # First entry
                row_data["run_index"] = 1
                row_data["attempts"] = 0
                self.results_df = pd.DataFrame([row_data])

            self.log_mssg(
                f"Updated results_df with run_index={row_data['run_index']}", level="ok"
            )
            return self.results_df.copy()

        except Exception as e:
            self.log_mssg(f"Error converting tensor to DataFrame: {e}", level="error")
            raise

    def _add_y_to_df(
        self,
        run_index: int,
        y: torch.Tensor,
        y_var: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add target values and optional variances to the results DataFrame for a given run.

        :param run_index: The index of the run to update.
        :param y: 1D tensor of target predictions, length must match number of targets.
        :param y_var: Optional 1D tensor of target variances; if provided, length must match number of targets.
        :raises ValueError: If results_df is not initialized or tensor lengths mismatch targets.
        """
        try:
            # Log start of y addition
            self.log_mssg(
                f"Adding target values for run_index={run_index}.", level="ok"
            )

            # Validate results_df
            if not hasattr(self, "results_df"):
                raise ValueError("results_df not initialized on instance.")

            # Validate lengths
            n_targets = len(self.targets)
            if y.numel() != n_targets:
                raise ValueError(
                    f"Length of y ({y.numel()}) does not match number of targets ({n_targets})."
                )
            if y_var is not None and y_var.numel() != n_targets:
                raise ValueError(
                    f"Length of y_var ({y_var.numel()}) does not match number of targets ({n_targets})."
                )

            # Assign values
            for i, target in enumerate(self.targets):
                self.results_df.loc[
                    self.results_df["run_index"] == run_index, target
                ] = y[i].item()
                self.log_mssg(f"Set '{target}' to {y[i].item()}.", indent=1)
                if (
                    y_var is not None
                    and f"{target}_variance" in self.results_df.columns
                ):
                    self.results_df.loc[
                        self.results_df["run_index"] == run_index, f"{target}_variance"
                    ] = y_var[i].item()
                    self.log_mssg(
                        f"Set '{target}_variance' to {y_var[i].item()}.",
                        indent=1,
                    )
        except Exception as e:
            # Log error before propagating
            self.log_mssg(
                f"Error adding target values for run_index {run_index}: {e}",
                level="error",
            )
            raise

    def _generate_check_dict(self) -> None:
        """
        Generate dictionaries and metadata for translating ML parameters to tensor indices and bounds.

        Initializes:
            - self.check_dict: mapping parameter names to tensor indices or slices.
            - self.bounds: torch.Tensor of shape (2, num_features) with min/max bounds per feature.
            - self.categoricals_indexes, continuous_indexes, categorical_levels.
            - self.fidelity_feature, task_feature, task_linked_physical_indexes as needed.
            - self.tensor_shape: expected shape for input tensors.

        :raises ValueError: If more than one fidelity or task parameter is found.
        """
        # Start generation
        self.log_mssg(
            "Generating check and bounds dictionaries for ML parameters.", level="ok"
        )
        self.fidelity_found = False
        self.task_found = False
        self.check_dict: Dict[str, Any] = {}
        minbounds: List[float] = []
        maxbounds: List[float] = []
        self.categoricals_indexes: List[int] = []
        self.categorical_levels: List[int] = []
        self.continuous_indexes: List[int] = []
        start_index = 0
        task_parameter_index: Optional[int] = None

        # Loop over parameters
        self.log_mssg("Mapping ML parameters to tensor indices.", indent=1)
        for idx, param in enumerate(self.ML_parameters):
            self.log_mssg(f"Processing parameter '{param.name}'", indent=2)
            name = param.name
            # Continuous mapping
            if hasattr(param, "translation_continuous"):
                # check for special case fidelity:
                if param.fidelity_feature:
                    self.fidelity_found = True
                    self.fidelity_feature = [start_index]
                self.check_dict[f"{name}_continuous"] = start_index
                self.continuous_indexes.append(start_index)
                minbounds.append(0.0)
                maxbounds.append(1.0)
                self.log_mssg(
                    f"Mapped continuous '{name}' to index {start_index}.",
                    indent=1,
                )
                start_index += 1

            # Discrete / categorical mapping
            if hasattr(param, "translation_discrete"):
                dim = param.discrete_embed_dim
                # Case: single-dimension discrete
                if dim == 1:
                    self.check_dict[f"{name}_discrete"] = start_index
                    self.categorical_levels.append(dim)
                    self.categoricals_indexes.append(start_index)
                    minbounds.append(0)
                    maxbounds.append(1)
                    self.log_mssg(
                        f"Mapped single-value discrete '{name}' to index {start_index}.",
                        indent=1,
                    )
                    start_index += 1
                # Case: low-cardinal one-hot encoding
                elif not param.high_categorical:
                    indices = (start_index, start_index + dim)
                    self.check_dict[f"{name}_discrete"] = indices
                    self.categorical_levels.append(dim)
                    self.categoricals_indexes.extend(
                        range(start_index, start_index + dim)
                    )
                    minbounds.extend([0] * dim)
                    maxbounds.extend([1] * dim)
                    self.log_mssg(
                        f"Mapped discrete '{name}' to slice {indices}.",
                        indent=1,
                    )
                    start_index += dim
                # Case: high-cardinality embedding
                else:
                    self.check_dict[f"{name}_discrete"] = start_index
                    self.categorical_levels.append(dim)
                    self.categoricals_indexes.append(start_index)
                    minbounds.append(0)
                    maxbounds.append(dim - 1)
                    self.log_mssg(
                        f"Mapped high-categorical '{name}' to index {start_index}.",
                        indent=1,
                    )
                    start_index += 1

            # Fidelity feature
            if hasattr(param, "translation_fidelity"):
                if self.fidelity_found or self.task_found:
                    msg = "Only one fidelity or task parameter is allowed"
                    self.log_mssg(msg, level="error")
                    raise ValueError(msg)
                self.check_dict[f"{name}_fidelity"] = start_index
                self.fidelity_feature = [start_index]
                self.fidelity_found = True
                self.fidelity_value = param.translation_fidelity(param.fidelity_value)
                minbounds.append(0.0)
                maxbounds.append(1.0)
                self.log_mssg(
                    f"Mapped fidelity '{name}' to index {start_index}.",
                    indent=1,
                )
                start_index += 1

            # Task feature
            if hasattr(param, "translation_task"):
                if self.fidelity_found or self.task_found:
                    msg = "Only one fidelity or task parameter is allowed"
                    self.log_mssg(msg, level="error")
                    raise ValueError(msg)
                self.check_dict[f"{name}_task"] = start_index
                self.task_feature = [start_index]
                self.task_found = True
                self.task_value = param.translation_task(param.task_value)
                task_parameter_index = idx
                minbounds.append(0)
                # maxbounds set by linked features logic later or default
                maxbounds.append(len(param.discrete) - 1)
                self.log_mssg(
                    f"Mapped task '{name}' to index {start_index}.",
                    indent=1,
                )
                start_index += 1

        # Handle linked physical features after loop
        if task_parameter_index is not None and getattr(
            self.ML_parameters[task_parameter_index], "linked_physical_features", None
        ):
            param = self.ML_parameters[task_parameter_index]
            linked_idxs: List[int] = []
            for phys in param.linked_physical_features:
                cont = self.check_dict.get(f"{phys}_continuous")
                disc = self.check_dict.get(f"{phys}_discrete")
                if isinstance(disc, tuple):
                    linked_idxs.extend(list(range(disc[0], disc[1])))
                elif disc is not None:
                    linked_idxs.append(disc)
                if cont is not None:
                    linked_idxs.append(cont)
            # Remove bounds for linked physicals
            remove_set = set(linked_idxs)
            minbounds = [v for i, v in enumerate(minbounds) if i not in remove_set]
            maxbounds = [v for i, v in enumerate(maxbounds) if i not in remove_set]
            self.task_linked_physical_indexes = linked_idxs
            self.log_mssg(
                f"Linked physical indexes removed from bounds: {linked_idxs}.",
                indent=1,
            )

        # Finalize bounds and tensor shape
        self._adjust_bounds(minbounds, maxbounds)
        self.tensor_shape = (1, start_index)
        self.log_mssg(
            f"Generated bounds tensor of shape {self.bounds.shape} and tensor_shape={self.tensor_shape}.",
        )

    def _adjust_bounds(self, minbounds: List[float], maxbounds: List[float]) -> None:
        """
        Adjust the bounds for each parameter based on the provided min and max values.

        :param minbounds: List of minimum bounds for each parameter.
        :param maxbounds: List of maximum bounds for each parameter.
        """
        self.log_mssg(f"Adjusting bounds with min={minbounds} and max={maxbounds}.")

        # check which ML are we doing:
        match self._bound_adjustment_method:
            case "None":
                pass
            case "RemoveTask":
                # remove task feature
                if hasattr(self, "task_feature"):
                    minbounds.pop(self.task_feature[0])
                    maxbounds.pop(self.task_feature[0])
                    self.log_mssg(
                        f"Removed task feature from bounds: {self.task_feature}.",
                        indent=1,
                    )
            case "RemoveFidelity":
                # remove fidelity feature
                if hasattr(self, "fidelity_feature"):
                    minbounds.pop(self.fidelity_feature[0])
                    maxbounds.pop(self.fidelity_feature[0])
                    self.log_mssg(
                        f"Removed fidelity feature from bounds: {self.fidelity_feature}.",
                        indent=1,
                    )
            case _:
                self.log_mssg("Unknown adjustment method", level="error")
                raise ValueError(
                    f"Unknown adjustment method: {self._bound_adjustement_method}"
                )
        # check if we have a task feature

        self.bounds = torch.tensor([minbounds, maxbounds], dtype=torch.float64)
        self.log_mssg(
            f"Adjusted bounds tensor to shape {self.bounds.shape}.", level="ok"
        )

    @property
    def stop_event(self):
        return self._stop_event
