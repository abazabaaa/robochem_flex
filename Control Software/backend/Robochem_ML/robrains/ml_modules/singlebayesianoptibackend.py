"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import copy
import time
from typing import Any, Optional, Type, Tuple, Dict

import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    qLogExpectedImprovement,
    ProbabilityOfImprovement,
    qProbabilityOfImprovement,
    UpperConfidenceBound,
    qUpperConfidenceBound,
    qMaxValueEntropy,
    qKnowledgeGradient,
    qNegIntegratedPosteriorVariance,
    ScalarizedPosteriorTransform,
    LinearMCObjective,
)
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective
from botorch.models import SingleTaskGP, MixedSingleTaskGP, ModelListGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.utils import draw_sobol_samples
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch import ExactMarginalLogLikelihood
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from robrains.communication_module import baseMLBackend
from robrains.custom_acquisition import MaxVariance
from robrains.custom_models import (
    RandomForestSurrogate,
    NeuralNetworkSurrogate,
    BayesianNeuralNetworkSurrogate,
    SVRSurrogate,
    MultiOutputSurrogate,
)
from robrains.utils import initialise_parameter, ColumnNormalizer


class SingleBayesianOptiBackend(baseMLBackend):
    """
    class members:
    input_parameters: dict, dictionary of the input parameters for the class
    input_defaults: dict, dictionary of the default values for the input parameters
    constructors: dict, dictionary of the constructors for the models and acquisition functions
    ml_parameters: dict, dictionary of the parameters for the ML side
    tags: list, list of tags for the class
    _primed: bool, flag to check if the class has been primed
    results_df: pd.DataFrame, dataframe to store the results of the optimisation process
    train_x: torch.tensor, tensor of the input values
    train_y: torch.tensor, tensor of the output values
    model: botorch model, model for the optimisation process
    acquisition_function: botorch acquisition function, acquisition function for the optimisation process
    check_dict: dict, dictionary to check the position of the values in the tensor


    class methods:
    prime: method to prime the class
    validate_and_update: method to validate and update the parameters
    initialise_model: method to initialise the model
    save_state: method to save the state of the class
    load_state_from_file: method to load the state of the class from a file
    run: method to run the optimisation process
    push_results: method to push the results of the optimisation process
    validate_data: method to validate the data
    reset: method to reset the class
    initialise_model: method to initialise the model
    _to_machine: method to translate the values to the hardware environment
    _from_machine: method to translate the values from the hardware environment
    _from_df: method to translate the values from a DataFrame
    _to_df: method to translate the values to a DataFrame
    res_df: property to return the results_df
    primed: property to return the primed flag

    """

    input_parameters = {
        "Model": [
            "SingleTaskGP",
            "MixedSingleTaskGP",
            "NoisySingleTaskGP",
            "RandomForest",
            "NeuralNetworkEnsemble",
            "BayesianNeuralNetwork",
            "SVR",
        ],
        "Acquisition Function": ["EI", "PI", "UCB", "qEHVI", "qLogNEHVI"],
        "Initialisation Method": ["LHS", "Random"],
        "Number of initial points": "int",
        "Number of total points": "int",
        "Number of Experiments per batch": "int",
        "Explorative Factor": "float",
        "eta": "float",
        "alpha": "float",
        "force_categorical": "bool",
        "adaptive": "bool",
        "adaptive_threshold_exploration": "float",
        "adaptive_threshold_exploitation": "float",
        "adaptivity_counter": "int",
        "weighted_multi_objective": "bool",
        "weigths": "list",
        "Termination criterion": ["max_iter", "performance"],
        "Resubmission of Failed N": "int",
    }
    input_defaults = {
        "Model": "SingleTaskGP",
        "Acquisition Function": "UCB",
        "Initialisation Method": "LHS",
        "Number of initial points": 10,
        "Number of total points": 100,
        "Number of Experiments per batch": 1,
        "Explorative Factor": 0.1,
        "eta": 0.001,
        "alpha": 0.0,
        "force_categorical": False,
        "adaptive": False,
        "adaptive_threshold_exploration": 0.05,
        "adaptive_threshold_exploitation": 0.8,
        "adaptivity_counter": 3,
        "weighted_multi_objective": False,
        "weigths": [1, 1],
        "Termination criterion": "max_iter",
        "Resubmission of Failed N": 0,
    }
    constructors = {
        "SingleTaskGP": SingleTaskGP,
        "MixedSingleTaskGP": MixedSingleTaskGP,
        "NoisySingleTaskGP": SingleTaskGP,
        "RandomForest": RandomForestSurrogate,
        "NeuralNetworkEnsemble": NeuralNetworkSurrogate,
        "BayesianNeuralNetwork": BayesianNeuralNetworkSurrogate,
        "SVR": SVRSurrogate,
    }
    botorch_models = [
        "SingleTaskGP",
        "MixedSingleTaskGP",
        "NoisySingleTaskGP",
        "SingleTaskMultiFidelityGP",
    ]
    custom_models = [
        "RandomForest",
        "NeuralNetworkEnsemble",
        "BayesianNeuralNetwork",
        "SVR",
    ]
    ml_parameters = {}
    tags = ["Bayes", "SingleTask", "Machine", "ML"]
    _bound_adjustment_method = "None"

    def __init__(self):
        super().__init__()

    def validate_and_update(self, key: str, value: Any) -> None:
        """
        Validate and update the specified parameter.

        This method checks whether the provided key exists in the parameter set and
        updates its value if valid. Additional validation may be performed depending
        on the parameter type and predefined constraints.

        Parameters:
        -----------
        key : str
            The name of the parameter to update.
        value : Any
            The new value to assign to the parameter. The function may enforce
            specific type constraints depending on the parameter.

        Returns:
        --------
        None
            This function updates the parameter in place and does not return a value.

        Raises:
        -------
        KeyError
            If the specified key does not exist in the parameter set.
        ValueError
            If the provided value does not meet validation criteria for the parameter.
        TypeError
            If the value's type is incompatible with the expected parameter type.
        """
        if key not in self.input_parameters and key not in self.added_keys:
            self.log_mssg(f"Unknown parameter passed: {key}", level="warning")
            return

        match key:
            case "Model":
                if value not in self.input_parameters[key]:
                    self.log_mssg(f"Unknown model passed: {value}", level="warning")
                    return
                else:
                    self._update_parameter_list(value)
            case "Acquisition Function":
                if value not in self.input_parameters[key]:
                    self.log_mssg(
                        f"Unknown acquisition function passed: {value}",
                        level="warning",
                    )
                    return
            case "Initialisation Method":
                if value not in self.input_parameters[key]:
                    self.log_mssg(
                        f"Unknown initialisation method passed: {value}",
                        level="warning",
                    )
                    return
            case "Termination criterion":
                if value not in self.input_parameters[key]:
                    self.log_mssg(
                        f"Unknown termination criterion passed: {value}",
                        level="warning",
                    )
                    return
            case "Number of initial points":
                if not isinstance(value, int) or value < 1:
                    self.log_mssg(
                        f"Number of initial points must be a positive integer >1 got {value}",
                        level="warning",
                    )
                    return
            case "Number of total points":
                if not isinstance(value, int) or value < 1:
                    self.log_mssg(
                        f"Number of total points must be a positive integer >1 got {value}",
                        level="warning",
                    )
                    return
            case "Number of Experiments per batch":
                if not isinstance(value, int) or value < 1:
                    self.log_mssg(
                        f"Number of experiments per batch must be a positive integer >1 got {value}",
                        level="warning",
                    )
                    return
            case "Explorative Factor":
                if not isinstance(value, float) or value < 0:
                    self.log_mssg(
                        f"Explorative factor must be a positive float got {value}",
                        level="warning",
                    )
                    return
            case "force_categorical":
                if not isinstance(value, bool):
                    self.log_mssg(
                        f"force_categorical must be a boolean got {value}",
                        level="warning",
                    )
                    return
            case "Resubmission of Failed N":
                if not isinstance(value, int) or value < 0:
                    self.log_mssg(
                        f"Resubmission of Failed N must be a positive integer >0 got {value}",
                        level="warning",
                    )
                    return
            case "adaptive":
                if not isinstance(value, bool):
                    self.log_mssg(
                        f"adaptive must be a boolean got {value}", level="warning"
                    )
                    return
            case "adaptive_threshold_exploration":
                if not isinstance(value, float) or value < 0 or value > 1:
                    self.log_mssg(
                        f"adaptive_threshold_exploration must be a positive float got {value}",
                        level="warning",
                    )
                    return
            case "adaptive_threshold_exploitation":
                if not isinstance(value, float) or value < 0 or value > 1:
                    self.log_mssg(
                        f"adaptive_threshold_exploitation must be a positive float got {value}",
                        level="warning",
                    )
                    return
            case "adaptivity_counter":
                if not isinstance(value, int) or value < 0:
                    self.log_mssg(
                        f"adaptivity_counter must be a positive integer >0 got {value}",
                        level="warning",
                    )
                    return
            case "weighted_multi_objective":
                if not isinstance(value, bool):
                    self.log_mssg(
                        f"weighted_multi_objective must be a boolean got {value}",
                        level="warning",
                    )
                    return
            case "weigths":
                # parsing the str to a list
                if isinstance(value, str):
                    try:
                        val = value.strip("{[( )]}")
                        value = [float(i) for i in val.split(",")]
                    except Exception as e:
                        self.log_mssg(
                            f"Failed to parse the weights: {value}, error: {e}",
                            level="warning",
                        )
                        return
                if not isinstance(value, list):
                    self.log_mssg(
                        f"weigths must be a list of floats of the size of the targets got {value}",
                        level="warning",
                    )
                    return
            case "Cost Function":
                if value not in self.input_parameters[key]:
                    self.log_mssg(
                        f"Unknown cost function passed: {value}",
                        level="warning",
                    )
                    return

            case _:
                self._validate_extra_parameters(key, value)

        self.parameters[key] = value
        self.log_mssg(f"Parameter {key} updated to {value}", level="ok")

    def _validate_extra_parameters(self, key: str, value: Any) -> None:
        """
        Validate extra parameters based on their expected type.

        This method ensures that the provided parameter value matches its expected type.
        The expected type is determined either from `self.input_defaults` or from
        `self.constructors[model].param_dict`. If the value does not match the expected
        type, a warning is logged, but no exception is raised.

        Parameters:
        -----------
        key : str
            The name of the parameter to validate.
        value : Any
            The value to validate against the expected type.

        Returns:
        --------
        None
            This function performs validation and logs warnings but does not return a value.

        Raises:
        -------
        KeyError
            If the key is not found in `self.input_defaults` or `self.added_keys`.

        Notes:
        ------
        - The expected type is determined dynamically.
        - Logs warnings if the value does not match the expected type.
        - Supports validation for `int`, `float`, `bool`, and `list` types.
        - If an unknown type is encountered, a generic warning is logged.
        """
        model = self.parameters["Model"]
        # get the map type from input_defaults
        if key in self.input_defaults:
            type_ = type(self.input_defaults[key])
        elif key in self.added_keys:
            type_ = self.constructors[model].param_dict[key]

        match type_:
            case "int":
                if not isinstance(value, int):
                    self.log_mssg(
                        f"Parameter {key} must be an integer got {value}",
                        level="warning",
                    )
                    return
            case "float":
                if not isinstance(value, float):
                    self.log_mssg(
                        f"Parameter {key} must be a float got {value}",
                        level="warning",
                    )
                    return
            case "bool":
                if not isinstance(value, bool):
                    self.log_mssg(
                        f"Parameter {key} must be a boolean got {value}",
                        level="warning",
                    )
                    return
            case "list":
                if not isinstance(value, list):
                    self.log_mssg(
                        f"Parameter {key} must be a list got {value}",
                        level="warning",
                    )
                    return
            case _:
                self.log_mssg("Unknown type", level="warning")

    def _update_parameter_list(self, model: str) -> None:
        """
        Update the list of parameters available to the model.

        This method ensures that the list of parameters dynamically updates when the
        model is changed. If the model is reset to a BoTorch-built model, all added
        parameters are removed, reverting to the default parameter set. If the model
        is switched to a custom model, additional parameters defined in the model's
        `param_dict` and `param_defaults` are included in the parameter list.

        Parameters:
        -----------
        model : str
            The name of the model whose parameters should be updated.

        Returns:
        --------
        None
            This function updates the internal parameter lists in place.

        Notes:
        ------
        - If the model belongs to `self.botorch_models`, all extra parameters are removed.
        - If the model belongs to `self.custom_models`, the function ensures that only
          the parameters required for the new model are included.
        - Parameters that are no longer needed are removed.
        - New parameters from `param_dict` are added with their default values from `param_defaults`.
        - The `self.added_keys` list keeps track of parameters that were dynamically added.
        """
        if not hasattr(self, "added_keys"):
            self.added_keys = []
        if model in self.botorch_models:
            for key in self.added_keys:
                self.input_parameters.pop(key)
                self.input_defaults.pop(key)
            self.added_keys = []
        elif model in self.custom_models:
            # get what needs to be added from the model:
            param_dict = self.constructors[model].param_dict
            param_defaults = self.constructors[model].param_defaults

            param_keys = list(param_dict.keys())

            # check between the keys and the added keys if there's stuff to remove/add
            to_remove = [key for key in self.added_keys if key not in param_keys]
            to_add = [key for key in param_keys if key not in self.added_keys]

            for key in to_remove:
                self.input_parameters.pop(key)
                self.input_defaults.pop(key)
                self.added_keys.remove(key)
            for key in to_add:
                self.input_parameters[key] = param_dict[key]
                self.input_defaults[key] = param_defaults[key]
                self.added_keys.append(key)

    def first_run(self) -> pd.DataFrame:
        """
        Perform the first run of the optimization process before entering the main loop.

        This method initializes the first set of parameter points for the optimization
        process. If a results DataFrame (`self.results_df`) already exists and contains
        completed runs, the number of initial points to generate is adjusted accordingly.

        The function follows these steps:
        1. Retrieves the "Explorative Factor" parameter (defaulting to 1 if not set).
        2. Sets `self.run_index` to 0.
        3. Determines the number of initial points to generate (`extra_points`).
        4. If `self.results_df` exists and is non-empty:
           - Adjusts `extra_points` based on completed runs.
           - Updates `self.run_index` to the highest existing `run_index` + 1.
           - If no more initial points are needed, returns a copy of `self.results_df`.
        5. Generates the required number of initial points (`self.initial_x`).
        6. Returns the processed initial points using `self.out_data()`.

        Returns:
        --------
        pd.DataFrame
            - If no additional initial points need to be generated, returns a copy of `self.results_df`.
            - Otherwise, returns the initialized data processed by `self.out_data()`.

        Notes:
        ------
        - The number of initial points is determined by `self.parameters["Number of initial points"]`.
        - If the initialization method includes categorical variables, it respects
          `self.parameters["force_categorical"]`.
        - Logs messages throughout the process to track execution steps.
        """
        self.explorative_factor_backup = self.parameters.get("Explorative Factor", 1)
        self.alpha_backup = self.parameters.get("alpha", 0.0)
        self.eta_backup = self.parameters.get("eta", 0.001)
        self.run_index = 0
        existing_points = None
        extra_points = self.parameters["Number of initial points"]

        if (
            hasattr(self, "results_df")
            and isinstance(self.results_df, pd.DataFrame)
            and not self.results_df.empty
        ):
            # diminish the amount of initial points to generate
            self.log_mssg(
                "Existing results detected; adjusting initial points.",
                indent=1,
            )

            self.run_index = max(self.results_df["run_index"].values) + 1
            self.log_mssg(f"Run index set to {self.run_index}")

            extra_points = self.parameters["Number of initial points"] - len(
                self.results_df[self.results_df["status"] == "finished"]
            )
            existing_points = self.results_df[self.results_df["status"] == "finished"]
            self.log_mssg(
                f"New run_index={self.run_index}, extra_points={extra_points}",
                indent=2,
            )
            if extra_points <= 0:
                self.log_mssg(
                    "No initial points needed; returning existing results_df.",
                    level="ok",
                    indent=1,
                )
                return self.results_df.copy()

        self.log_mssg(f"generating {extra_points} initial points")
        self.initial_x = initialise_parameter(
            self.ML_parameters,
            number_of_points=extra_points,
            method=self.parameters["Initialisation Method"],
            force_categorical=self.parameters["force_categorical"],
            existing_points=existing_points,
        )

        self.log_mssg("first_run() completed; initial points appended.", level="ok")

        return self.out_data(self.initial_x)

    def _process_data(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract training tensors from the current results DataFrame.

        :return: A tuple (x, y, y_var) where:
            - x: Input feature tensor.
            - y: Target value tensor.
            - y_var: Optional tensor of target variances.
        """
        self.log_mssg("Processing results_df into training tensors.")
        x, y, y_var = self._from_df(self.results_df)
        self.log_mssg(
            f"Processed data shapes — x: {x.shape}, y: {y.shape}, "
            f"y_var: {None if y_var is None else y_var.shape}",
            level="ok",
            indent=1,
        )
        return x, y, y_var

    def _check_adaptive(self, y: torch.Tensor) -> None:
        """
        Adjust exploration parameters dynamically based on recent output variability.

        If adaptive mode is off, does nothing. Otherwise:
          - Looks at the last 3 rows of y (shape: n_runs x n_targets).
          - If any column’s std < exploration_threshold * mean ➔ set 'explorative'.
          - If any column’s std > exploitation_threshold * mean ➔ set 'exploitative'.
          - Else, remain in 'user' mode.
        Adaptivity can only switch once per buffer period controlled by 'adaptivity_counter'.

        :param y: 2D tensor of past target values.
        """
        if not self.parameters.get("adaptive", False):
            self.log_mssg("Adaptive mode disabled; skipping check.")
            return

        counter = getattr(self, "_adaptivity_counter", 0)
        if counter > 0:
            self._adaptivity_counter = counter - 1
            self.log_mssg(
                f"Adaptivity buffer active: {self._adaptivity_counter} calls remaining.",
                level="ok",
                indent=1,
            )
            return

        self.log_mssg(
            "Performing adaptivity analysis on last 3 data points.", level="ok"
        )
        window = y[-3:, :]
        means = window.mean(dim=0)
        stds = window.std(dim=0)
        self.log_mssg(
            f"Computed means={means.tolist()}, stds={stds.tolist()}",
            indent=1,
        )

        expl_thresh = self.parameters.get("adaptive_threshold_exploration", 0.05)
        explo_thresh = self.parameters.get("adaptive_threshold_exploitation", 0.8)
        mode = "user"
        for m, s in zip(means, stds):
            if s < expl_thresh * m:
                mode = "explorative"
                break
            if s > explo_thresh * m:
                mode = "exploitative"
                break

        self.log_mssg(f"Adaptivity mode set to '{mode}'.", level="ok")
        self._set_adaptivity(mode)
        # Reset buffer only if we switched out of 'user'
        self._adaptivity_counter = (
            self.parameters.get("adaptivity_counter", 3) if mode != "user" else 0
        )

    def _set_adaptivity(self, level: str = "explorative") -> None:
        """
        Set exploration parameters based on adaptivity level.

        :param level: One of 'explorative', 'exploitative', or 'user'.
        :raises ValueError: If level is unrecognized.
        """
        self.log_mssg(f"Setting adaptivity level to '{level}'")

        match level:
            case "explorative":
                self.parameters["Explorative Factor"] = 10.0
                self.parameters["alpha"] = 0.2
                self.parameters["eta"] = 0.02
            case "exploitative":
                self.parameters["Explorative Factor"] = 0.5
                self.parameters["alpha"] = 0.0
                self.parameters["eta"] = 0.0001
            case "user":
                # Restore backed-up settings
                self.parameters["Explorative Factor"] = getattr(
                    self,
                    "explorative_factor_backup",
                    self.parameters.get("Explorative Factor", 1.0),
                )
                self.parameters["alpha"] = getattr(
                    self, "alpha_backup", self.parameters.get("alpha", 0.0)
                )
                self.parameters["eta"] = getattr(
                    self, "eta_backup", self.parameters.get("eta", 0.001)
                )
            case _:
                self.log_mssg(f"Unknown adaptivity level: '{level}'", level="error")
                raise ValueError(f"Unknown adaptivity level: {level}")

        # Confirm new settings
        self.log_mssg(
            f"Adaptivity parameters updated: Explorative Factor={self.parameters['Explorative Factor']}, "
            f"alpha={self.parameters['alpha']}, eta={self.parameters['eta']}",
            level="ok",
            indent=1,
        )

    def update(self) -> None:
        """
        Perform one Bayesian optimization loop:
          1. Process incoming data into training tensors.
          2. Handle failure cases by signalling retries.
          3. Apply adaptive parameter adjustments if enabled.
          4. Normalize and reshape training data.
          5. Check termination criteria (max iterations).
          6. Execute BO step to propose next point.
        """
        self.log_mssg("Starting BO loop")
        start_time = time.time()

        # 1. Process data
        x, y, y_var = self._process_data()
        self.log_mssg(
            f"Data shapes — x: {x.shape}, y: {y.shape}, y_var: {None if y_var is None else y_var.shape}",
            indent=1,
        )

        # 2. Failure or retry check
        if (
            x.numel() == 0
            or self.results_df["status"].str.contains("retry", na=False).any()
        ):
            self.log_mssg(
                "Detected empty data or retries pending; signalling failures.",
                level="warning",
            )
            self.out_data("failures")
            return

        # 3. Adaptive adjustments
        self._check_adaptive(y=y)

        # 4. Normalize training data
        finished_mask = self.results_df["status"] == "finished"
        n_finished = finished_mask.sum()
        train_y = y.reshape(n_finished, -1)
        self.log_mssg(f"Normalizing {n_finished} finished runs.", indent=1)
        self.normalizer_object = ColumnNormalizer(train_y)
        self.train_y = self.normalizer_object.normalize(train_y)
        self.train_x = x.reshape(n_finished, -1)

        # Handle noise variance if applicable
        if self.parameters.get("Model") == "NoisySingleTaskGP" and y_var is not None:
            train_y_var = y_var.reshape(n_finished, -1)
            self.train_y_var = self.normalizer_object.normalize_variance(train_y_var)
            self.log_mssg("Normalized variance for noisy GP model.", indent=1)

        # 5. Termination check
        if self.parameters.get("Termination criterion") == "max_iter":
            max_pts = int(self.parameters.get("Number of total points", 0))
            if self.run_index >= max_pts:
                self.log_mssg(f"Reached max iterations ({self.run_index}).", level="ok")
                self.out_data("stop")
                elapsed = time.time() - start_time
                self.log_mssg(f"Optimization complete in {elapsed:.1f}s", level="ok")
                return

        # 6. Bayesian optimization step
        self._BO_step()
        elapsed = time.time() - start_time
        self.log_mssg(f"BO step finished in {elapsed:.1f}s", level="ok")

    def _BO_step(self) -> None:
        """
        Execute one Bayesian optimization step:
          1. Prepare training data and optional variance.
          2. Initialize the surrogate model.
          3. Build and optimize the acquisition function to propose next point.
          4. Predict on the new point and enqueue via out_data().
        """
        # 1. Prepare data
        self.log_mssg("Starting BO step.")
        x = self.ensure_writable(self.train_x)
        y = self.ensure_writable(self.train_y)
        y_var = (
            self.ensure_writable(self.train_y_var)
            if hasattr(self, "train_y_var")
            else None
        )
        self.log_mssg(
            f"Training data shapes — x: {x.shape}, y: {y.shape}, "
            f"y_var: {None if y_var is None else y_var.shape}",
            indent=1,
        )

        # 2. Train model
        self._init_model(x, y, y_var=y_var)
        self.log_mssg("Surrogate model initialized and trained.", level="ok", indent=1)

        # 3. Acquisition setup
        acq_class = self._select_acq_constructor(
            fn_name=self.parameters["Acquisition Function"],
            is_single=self.parameters["Number of Experiments per batch"] == 1,
            weighted=self.parameters.get("weighted_multi_objective", False),
        )
        acq_args = self._get_acquisition_args()
        device = torch.device("cpu")
        acqf = acq_class(**acq_args).to(device)
        self.log_mssg(
            f"Acquisition function {acq_class.__name__} instantiated.",
            level="ok",
            indent=1,
        )

        # Optimize acquisition
        next_point, acq_val = self._optimize_acquisition_function(acqf)
        self.log_mssg(
            f"Acquisition optimized; next_point shape: {next_point.shape}, value: {acq_val}",
            level="ok",
            indent=1,
        )

        # 4. Prediction and enqueue
        with torch.no_grad():
            post = self.model.posterior(next_point)
            mean, var = post.mean, post.variance
            pred_y = self.normalizer_object.denormalize_mean_variance((mean, var))

        self.log_mssg(
            "Predicted mean and variance for next point.", level="ok", indent=1
        )
        self.log_mssg(f"Predicted value:{pred_y}", indent=2)
        self.out_data(next_point, predicted_y=pred_y)
        self.log_mssg("BO step complete; next point enqueued.", level="ok")

    def _select_acq_constructor(
        self, fn_name: str, is_single: bool, weighted: Optional[bool] = False
    ) -> Type:
        """
        Helper method that returns the appropriate acquisition function *class*
        given a function name (e.g. "EI", "UCB", "qEHVI", etc.) and whether
        we are single-point or batched.

        Args:
            fn_name: Name of the acquisition function, e.g. "EI", "UCB", "qLogNEHVI", ...
            is_single: Whether the user requested single-experiment (True) or batched (False).
            weighted: Whether the multi-objective acquisition function is weighted.

        Returns:
            The acquisition function class (not yet instantiated).
        """
        match fn_name:
            case "EI":
                return qLogExpectedImprovement
            case "PI":
                return (
                    ProbabilityOfImprovement if is_single else qProbabilityOfImprovement
                )
            case "UCB":
                return UpperConfidenceBound if is_single else qUpperConfidenceBound
            case "qEHVI":
                # multi-objective hypervolume
                # if weighted:
                #     pass
                #     # self.log_mssg(
                #     #     "You selected qEHVI but this is done for unweighted multi-objective optimisation \n"
                #     #     "Switching to qUpperConfidenceBound"
                #     # )
                #     # self.parameters["Acquisition Function"] = "UCB"
                #     # return UpperConfidenceBound if is_single else qUpperConfidenceBound
                return qLogExpectedHypervolumeImprovement

            case "qLogNEHVI":
                # MC-based noisy hypervolume improvement
                return qLogNoisyExpectedHypervolumeImprovement
            case "qMVE":
                return qMaxValueEntropy
            case "qKG":
                return qKnowledgeGradient
            case "MaxVariance":
                return MaxVariance
            case "qNegIPV":
                return qNegIntegratedPosteriorVariance
            case "qLogNParEGO":
                return qLogNParEGO
            case "qPPES":
                return qMultiObjectivePredictiveEntropySearch

            # If you have custom classes like qLogExpectedImprovement, handle them here
            # case "qLogEI":
            #     return qLogExpectedImprovement if not is_single else SomeOtherClass
            case _:
                raise ValueError(f"Unsupported acquisition function: {fn_name}")

    def _get_acquisition_func(self):
        """gets the correct acquisition function constructor"""
        num_experiments = self.parameters["Number of Experiments per batch"]
        is_single = num_experiments == 1
        weighted = self.parameters.get("weighted_multi_objective", False)

        return self._select_acq_constructor(
            self.parameters["Acquisition Function"], is_single, weighted
        )

    def _init_model(
        self, x: torch.Tensor, y: torch.Tensor, y_var: Optional[torch.Tensor] = None
    ) -> None:
        """
        Initialize and fit the surrogate model on provided training data.

        :param x: Input features tensor of shape (n_samples, n_features).
        :param y: Target values tensor of shape (n_samples, n_targets).
        :param y_var: Optional variance tensor of shape (n_samples, n_targets) for noisy GPs.
        :raises ValueError: If the specified model is unsupported.
        """
        self.log_mssg("Initializing surrogate model")

        model_name = self.parameters["Model"]
        constructor = self.constructors[model_name]
        model_kwargs = self._get_model_kwargs()

        device = torch.device("cpu")
        x = x.to(device)
        y = y.to(device)

        if y_var is not None:
            y_var = y_var.to(device)
            model_kwargs["train_Yvar"] = y_var
            self.log_mssg(
                f"Including train_Yvar of shape {y_var.shape} in model kwargs",
                indent=1,
            )

        if model_name in self.botorch_models:
            self.log_mssg(f"Using BoTorch model '{model_name}'", indent=1)
            self._init_botorch_models(x, y, constructor, device, model_kwargs)
        elif model_name in self.custom_models:
            self.log_mssg(f"Using custom model '{model_name}'", indent=1)
            self._init_custom_models(x, y, constructor, device, model_kwargs)
        else:
            self.log_mssg(f"Unsupported model '{model_name}'", level="error")
            raise ValueError(f"Unsupported model: {model_name}")

        self.log_mssg("Model initialization complete", level="ok")

    def _init_botorch_models(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model_constructor: Type,
        device: torch.device,
        model_kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize and fit BoTorch GP models.

        :param x: Input tensor of shape (n_samples, n_features).
        :param y: Output tensor of shape (n_samples, n_targets).
        :param model_constructor: BoTorch model class to instantiate.
        :param device: torch.device to move models and data onto.
        :param model_kwargs: Dict of keyword arguments for model construction,
                             may include 'train_Yvar' for noise variances.
        """
        import copy
        from botorch.models import ModelListGP
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll

        self.log_mssg("Initializing BoTorch models")

        # Extract and remove variance if present
        y_var = model_kwargs.pop("train_Yvar", None)

        # Multi-output case: separate one GP per target
        if y.shape[1] > 1:
            models = []
            for i in range(y.shape[1]):
                self.log_mssg(f"  Building GP for target index {i}", indent=1)
                kwargs_copy = copy.deepcopy(model_kwargs)
                if y_var is not None:
                    kwargs_copy["train_Yvar"] = y_var[:, i : i + 1]
                gp = model_constructor(
                    train_X=x, train_Y=y[:, i : i + 1], **kwargs_copy
                )
                gp.to(device)
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                models.append(gp)
            self.model = ModelListGP(*models).to(device)
            self.log_mssg(
                "Combined multi-output GP model created", level="ok", indent=1
            )

        # Single-output case
        else:
            self.log_mssg("Building single-output GP model", indent=1)
            gp = model_constructor(train_X=x, train_Y=y, **model_kwargs)
            gp.to(device)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            self.model = gp
            self.log_mssg("Single-output GP model created", level="ok", indent=1)

    def _init_custom_models(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model_constructor: Type,
        device: torch.device,
        model_kwargs: Dict[str, Any],
    ) -> None:
        """
        Initialize and fit custom surrogate models.

        :param x: Input tensor of shape (n_samples, n_features).
        :param y: Output tensor of shape (n_samples, n_targets).
        :param model_constructor: Custom surrogate class to instantiate.
        :param device: torch.device to move models onto.
        :param model_kwargs: Dict of keyword arguments for model construction.
        """
        self.log_mssg("Initializing custom surrogate models")

        # Multi-output: wrap individual models
        if y.shape[1] > 1:
            models = []
            for i in range(y.shape[1]):
                self.log_mssg(
                    f"  Building custom model for target index {i}",
                    indent=1,
                )
                mdl = model_constructor(
                    train_X=x, train_Y=y[:, i : i + 1], **model_kwargs
                )
                mdl.to(device)
                models.append(mdl)
            self.model = MultiOutputSurrogate(models)
            self.log_mssg("Multi-output custom surrogate created", level="ok", indent=1)

        # Single-output
        else:
            self.log_mssg("Building single-output custom surrogate", indent=1)
            mdl = model_constructor(train_X=x, train_Y=y, **model_kwargs)
            mdl.to(device)
            self.model = mdl
            self.log_mssg(
                "Single-output custom surrogate created", level="ok", indent=1
            )

    def _get_model_kwargs(self):
        """Get the keyword arguments for the model based on the parameters"""
        match self.parameters["Model"]:
            case "SingleTaskGP":
                return {
                    "outcome_transform": Standardize(m=1),
                    # "input_transform": Normalize(d=self.train_x.shape[1]),
                    # "covar_module": self.kernel
                }
            case "MixedSingleTaskGP":
                return {
                    "cat_dims": self.categoricals_indexes,
                    "outcome_transform": Standardize(m=1),
                    "covar_module": self.kernel,
                }
            case "FixedNoiseGP":
                return {"noise": torch.tensor(0.1), "noise_lower_bound": 1e-5}
            case "NoisySingleTaskGP":
                return {"outcome_transform": Standardize(m=1)}
            case "RandomForest":
                mod_kwargs = {
                    "outcome_transform": Standardize(m=1),
                }
                mod_kwargs.update(self._get_custom_kwargs(RandomForestSurrogate))
                return mod_kwargs
            case "NeuralNetworkEnsemble":
                mod_kwargs = {
                    "outcome_transform": Standardize(m=1),
                }
                mod_kwargs.update(self._get_custom_kwargs(NeuralNetworkSurrogate))
                return mod_kwargs
            case "BayesianNeuralNetwork":
                mod_kwargs = {
                    "outcome_transform": Standardize(m=1),
                }
                mod_kwargs.update(
                    self._get_custom_kwargs(BayesianNeuralNetworkSurrogate)
                )
                return mod_kwargs
            case "SVR":
                mod_kwargs = {
                    "outcome_transform": Standardize(m=1),
                }
                mod_kwargs.update(self._get_custom_kwargs(SVRSurrogate))
                return mod_kwargs

            case _:
                raise ValueError("Unsupported model")

    def _get_custom_kwargs(self, model_constructor):
        """
        Gets the keyword arguments for the custom model from the current parameters
        :param model_constructor: Custom model class
        :return: dict, dictionary of the model kwargs
        """
        ret = {}
        for key in model_constructor.param_dict:
            if key in self.parameters:
                ret[key] = self.parameters[key]

        return ret

    def _select_acq_args(
        self,
        fn_name: str,
        weighted: Optional[bool] = False,
        is_single: Optional[str] = "check",
    ) -> dict:
        """
        A helper that returns the appropriate argument dictionary for the given
        acquisition function name. This includes handling the 'weighted_multi_objective'
        logic, best_f for EI/PI, ref_point for HV-based methods, etc.

        Args:
            fn_name: Name of the acquisition function (e.g. "EI", "UCB", "qEHVI", "qLogNEHVI").

        Returns:
            A dictionary of keyword args to pass to the constructor of that acquisition.
        """
        # Common parameter dictionary (start with the model).
        # If you have more than one model, handle that here.
        param = {"model": self.model}

        # number of experiments is used for, e.g., best_f
        # or deciding if single or batched, but we mostly need it for the function itself.
        num_experiments = self.parameters["Number of Experiments per batch"]
        if is_single == "no_check":
            is_single = num_experiments == 1
        elif is_single == "true":
            is_single = True
        else:
            is_single = False

        match fn_name:
            # --------------------------------------------------
            # EI: need best_f
            case "EI":
                param["best_f"] = self.train_y.max().clone()
            # PI: also need best_f
            case "PI":
                param["best_f"] = self.train_y.max().clone()

            # --------------------------------------------------
            # UCB: need beta, plus handle weighting if multi-objective
            case "UCB":
                param["beta"] = self.parameters.get("Explorative Factor", 1)

                if weighted:
                    # We have weights for a multi-output model
                    weights = torch.tensor(
                        self.parameters.get("weigths", [1] * self.train_y.shape[-1]),
                        dtype=torch.float64,
                    )

                    # if is_single:
                    # Analytical UCB -> use PosteriorTransform
                    param["posterior_transform"] = ScalarizedPosteriorTransform(
                        weights=weights
                    )
                    # else:
                    #     # MC-based qUCB -> use MCObjective
                    #     param["objective"] = LinearMCObjective(weights=weights)

            # --------------------------------------------------
            # qEHVI: multi-objective hypervolume improvement
            case "qEHVI":
                # We set the partitioning and ref_point
                ref_point = self.train_y.min(dim=0)[0] - 0.1
                partition = NondominatedPartitioning(
                    ref_point=ref_point, Y=self.train_y
                )
                param["partitioning"] = partition
                param["ref_point"] = ref_point
                param["eta"] = self.parameters.get("eta", 0.001)

                if weighted:
                    # We have weights for a multi-output model
                    weights = torch.tensor(
                        self.parameters.get("weigths", [1] * self.train_y.shape[-1]),
                        dtype=torch.float64,
                    )

                    param["objective"] = WeightedMCMultiOutputObjective(weights=weights)

            # --------------------------------------------------
            # qLogNEHVI: noisy EI for hypervolume
            case "qLogNEHVI":
                ref_point = self.train_y.min(dim=0)[0] - 0.1
                X_baseline = copy.deepcopy(self.train_x)
                param["ref_point"] = ref_point
                param["X_baseline"] = X_baseline
                param["prune_baseline"] = True
                param["eta"] = self.parameters.get("eta", 0.001)
                param["alpha"] = self.parameters.get("alpha", 0.1)
                param["incremental_nehvi"] = False

            # If you have custom logic for "qLogExpectedImprovement" or others, add them here
            # case "qLogEI":
            #     param["best_f"] = self.train_y.max().clone()
            #     # etc.
            # qMaxValueEntropy: single-outcome Max-Value Entropy Search
            case "qMVE":
                # BoTorch docs note that model must be single-outcome or
                # we need a PosteriorTransform that reduces to 1 dimension.
                param["candidate_set"] = draw_sobol_samples(
                    bounds=self.bounds, n=1000, q=1, seed=666
                ).squeeze(
                    dim=-2
                )  # e.g. a (N x d) tensor
                param["num_fantasies"] = self.parameters.get("mves_num_fantasies", 16)
                param["num_mv_samples"] = self.parameters.get("mves_num_mv_samples", 10)
                param["num_y_samples"] = self.parameters.get("mves_num_y_samples", 128)
                param["posterior_transform"] = ScalarizedPosteriorTransform(
                    weights=torch.ones(self.train_y.shape[-1], dtype=torch.float64)
                )  # Or a transform if multi-output
                param["use_gumbel"] = True
                param["maximize"] = True
                if self.train_y.shape[-1] > 1:
                    param["train_inputs"] = self.train_x

            # --------------------------------------------------
            # MaxVariance: custom single- or multi-objective variance-based exploration
            case "MaxVariance":
                # Our custom class might expect 'num_samples' for MC estimation
                param["num_samples"] = self.parameters.get("MV_num_samples", 1000)
                if self.train_y.shape[-1] > 1:
                    # Multi output model, need to specify a posterior transform
                    param["posterior_transform"] = ScalarizedPosteriorTransform(
                        weights=torch.ones(self.train_y.shape[-1], dtype=torch.float64)
                    )

                # If you had a custom 'sampler', you'd pass that here too
                # param["sampler"] = my_custom_sampler

            # --------------------------------------------------
            # qKnowledgeGradient
            case "qKG":
                # # of fantasy points
                param["num_fantasies"] = self.parameters.get("kg_num_fantasies", 32)
                # We could specify a sampler for the outer fantasies
                param["sampler"] = None  # or an actual MCSampler instance
                # MC objective if needed
                param["objective"] = None
                param["posterior_transform"] = ScalarizedPosteriorTransform(
                    weights=torch.ones(self.train_y.shape[-1], dtype=torch.float64)
                )  # Or a transform if multi-output
            case "qNegIPV":
                param["mc_points"] = draw_sobol_samples(
                    bounds=self.bounds, n=512, q=1, seed=666
                ).squeeze(dim=-2)
                param["posterior_transform"] = ScalarizedPosteriorTransform(
                    weights=torch.ones(self.train_y.shape[-1], dtype=torch.float64)
                )
            case "qLogNParEGO":
                # baseline set for Chebyshev scalarization
                param["X_baseline"] = copy.deepcopy(self.train_x)
                # reference points are drawn via weights if not provided
                # you can optionally supply `scalarization_weights`
                # Monte Carlo sampler (defaults if omitted)
                # objective: a multi‐output objective, e.g. identity or custom
                # smoothing & caching options:
                param["eta"] = self.parameters.get("eta", 1e-3)
                param["fat"] = self.parameters.get("fat", True)
                param["cache_root"] = self.parameters.get("cache_root", True)
                param["tau_relu"] = self.parameters.get("tau_relu", 1e-2)
                param["tau_max"] = self.parameters.get("tau_max", 10.0)

                # --------------------------------------------------
                # qPPES: predictive Pareto‐set entropy search
            case "qPPES":
                S = self.parameters.get("ppe_num_samples", 64)
                P = self.parameters.get("ppe_max_pareto_size", None)
                # Generate Sobol‐based Pareto‐set samples
                param["pareto_sets"] = self._compute_pareto_sets_sobol(
                    num_representer_points=self.parameters.get("ppe_n_rep", 512),
                    num_samples=S,
                    P_target=P,
                    seed=self.parameters.get("ppe_sobol_seed", 0),
                )
                param["maximize"] = self.parameters.get("pareto_maximize", True)
                param["X_pending"] = self.parameters.get("pareto_X_pending", None)
                param["max_ep_iterations"] = self.parameters.get(
                    "ppe_max_ep_iterations", 250
                )
                param["ep_jitter"] = self.parameters.get("ppe_ep_jitter", 1e-4)
                param["test_jitter"] = self.parameters.get("ppe_test_jitter", 1e-4)
                param["threshold"] = self.parameters.get("ppe_threshold", 1e-2)
            case _:
                pass  # If it's none of the above, we just return the default param {"model": ...}

        return param

    def _get_acquisition_args(self):
        """Get the arguments for the acquisition function based on the parameters"""

        weighted = self.parameters.get("weighted_multi_objective", False)
        acquisition_function = self.parameters["Acquisition Function"]

        return self._select_acq_args(acquisition_function, weighted)

    def _optimize_acquisition_function(
        self, acquisition_function: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize the acquisition function to propose the next evaluation point(s).

        Steps:
          1. Prepare bounds and gradient settings.
          2. Call `optimize_acqf` with configured arguments.
          3. Log the chosen point and acquisition value.

        :param acquisition_function: Instantiated acquisition function module.
        :return: Tuple of
            - next_point: Tensor of shape (q, d) with proposed point(s).
            - acq_value: Tensor of acquisition value(s) at next_point.
        """

        self.log_mssg("Starting acquisition optimization", level="ok")

        # 1. Prepare bounds and options
        device = torch.device("cpu")
        bounds = self.ensure_writable(self.bounds).to(device)
        model_name = self.parameters.get("Model", "")
        with_grad = model_name not in ["RandomForest", "SVR"]
        options = {"with_grad": with_grad}

        q = int(self.parameters.get("Number of Experiments per batch", 1))
        self.log_mssg("Optimizing acquisition function with parameters:")
        self.log_mssg(f"  - q: {q}", indent=1)
        self.log_mssg(f"  - bounds: {bounds}", indent=1)
        self.log_mssg(f"  - options: {options}", indent=1)
        args = {
            "acq_function": acquisition_function,
            "bounds": bounds,
            "q": q,
            "num_restarts": 10,
            "raw_samples": 512,
            "options": options,
        }

        # 2. Run optimization
        next_point, acq_value = optimize_acqf(**args)
        self.log_mssg(
            f"Optimized acquisition; next_point shape={next_point.shape}, "
            f"acq_value={acq_value}",
            level="ok",
            indent=1,
        )

        # 3. Return result
        return next_point, acq_value

    def _compute_pareto_sets_sobol(
        self,
        num_representer_points: int = 512,
        num_samples: int = 64,
        P_target: int | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """
        Generate S Pareto‐set samples by:
          1) Sobol sampling the unit cube in d dims.
          2) Drawing posterior MC samples at those points.
          3) Extracting & padding/truncating non‐dominated fronts.

        Returns a tensor of shape (S × P_target × d).
        """
        d = self.train_x.size(-1)
        # 1) Sobol-representer inputs in [0,1]^d
        X_rep = draw_sobol_samples(
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]),
            n=num_representer_points,
            q=1,
            seed=seed,
        ).squeeze(
            -2
        )  # (N × d)

        pareto_sets = []
        max_size = 0

        for _ in range(num_samples):
            # 2) sample one joint function realization
            with torch.no_grad():
                post = self.model.posterior(X_rep)
                f_samp = post.sample(sample_shape=torch.Size([1])).squeeze(0)  # (N × m)
            # 3) extract current front
            mask = is_non_dominated(f_samp)
            X_front = X_rep[mask]  # (P_i × d)
            pareto_sets.append(X_front)
            max_size = max(max_size, X_front.size(0))

        # determine P_target
        P = P_target or max_size

        # 4) pad/truncate each front to P
        padded = []
        for X_front in pareto_sets:
            n = X_front.size(0)
            if n >= P:
                trimmed = X_front[:P]
            else:
                pad = X_front[-1:].expand(P - n, -1)
                trimmed = torch.cat([X_front, pad], dim=0)
            padded.append(trimmed.unsqueeze(0))  # (1 × P × d)

        # stack into (S × P × d)
        return torch.cat(padded, dim=0)
