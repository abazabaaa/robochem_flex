"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import List

import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from botorch.models import MultiTaskGP

from robrains.ml_modules.scopeacceleratortaskbackend import ScopeAcceleratorTaskBackend


class AllScopeTaskBackend(ScopeAcceleratorTaskBackend):
    """
    The scope accelerator is used to feed data for old optimisations into the system:

    Imagine the following scenario:
    Dima is working with the reaction: A+B -> P plus a bunch of parameters
    he has optimised it with RobERTA and has now an optimised reaction system. Now he has to
    start doing his substrate scope, where he will keep the parameters constant except B, which will be switched
    out for other substrates.

    Difference with previous method is that this uses a multi task approach instead of a multi fidelity approach:
    difference being multifidelity assumes one underlying function with different level of precision (fidelity) of each
    observation. Multi task assumes one underlaying function each task (in our case substrate) which are related to each other
    but not the same. Multi task leverages transfer learning in a different way.

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
            "MultiTaskGP",
        ],
        "Acquisition Function": ["EI", "UCB", "qEHVI"],
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
        "Model": "MultiTaskGP",
        "Acquisition Function": "EI",
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
        "MultiTaskGP": MultiTaskGP,
    }
    ml_parameters = {}
    tags = ["Bayes", "MultiTask", "Machine", "ML"]

    def __init__(self):
        super().__init__()

    def _BO_step(self) -> None:
        """
        Perform the BO step with fixed-feature acquisition across task values:
          1. Train the surrogate model on full data.
          2. For each task value in self.task_range:
             a. Wrap base acquisition in FixedFeatureAcquisitionFunction.
             b. Optimize to propose a candidate for that task.
             c. Reconstruct full feature vector with fixed task.
          3. Concatenate all task-specific candidates and enqueue via out_data().
        """
        self.log_mssg("Starting BO step with fixed-feature acquisition")

        # 1. Train surrogate on all tasks
        x = self.ensure_writable(self.train_x)
        y = self.ensure_writable(self.train_y)
        self._init_model(x, y)
        self.log_mssg(
            f"Model trained on data shape x={x.shape}, y={y.shape}",
            indent=1,
        )

        # 2. Base acquisition instantiation
        device = torch.device("cpu")
        acq_class = self._get_acquisition_func()
        acq_args = self._get_acquisition_args()
        base_acq = acq_class(**acq_args).to(device)
        self.log_mssg("Base acquisition function instantiated", indent=1)
        self.log_mssg(
            f"- Acquisition function: {base_acq.__class__.__name__}", indent=2
        )
        self.log_mssg(f"- Acquisition args: {acq_args}", indent=2)
        candidates: List[torch.Tensor] = []
        for task_val in self.task_range:
            self.log_mssg(f"Optimizing for task value {task_val}", indent=1)

            # 2a. Create fixed-feature acquisition
            fixed_acq = FixedFeatureAcquisitionFunction(
                acq_function=base_acq,
                d=x.shape[1],
                columns=self.task_feature,
                values=[task_val],
            )
            self.log_mssg("Fixed-feature acquisition wrapped", level="ok", indent=2)

            # 2b. Optimize under fixed-feature
            next_pt, acq_val = self._optimize_acquisition_function(fixed_acq)
            self.log_mssg(
                f"Candidate for task {task_val}: shape={next_pt.shape}, acq_val={acq_val}",
                level="ok",
                indent=2,
            )

            # 2c. Reconstruct full candidate vector
            full_pt = self._reconstruct_candidates(
                next_pt,
                task_feature_index=self.task_feature,
                task_feature_values=[task_val],
            )
            candidates.append(full_pt)

        # 3. Concatenate and enqueue
        next_batch = torch.cat(candidates, dim=0)
        self.log_mssg(f"Enqueuing {next_batch.shape[0]} total candidates", level="ok")
        self.out_data(next_batch)
