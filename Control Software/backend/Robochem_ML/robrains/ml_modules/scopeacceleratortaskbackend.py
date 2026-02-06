"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import List

import numpy as np
import torch
from botorch.acquisition import (
    FixedFeatureAcquisitionFunction,
    LinearMCObjective,
    ScalarizedPosteriorTransform,
)
from botorch.models import MultiTaskGP
from botorch.models.transforms import Standardize
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning

from robrains.ml_modules.scopeacceleratorfidelitybackend import (
    ScopeAcceleratorFidelityBackend,
)


class ScopeAcceleratorTaskBackend(ScopeAcceleratorFidelityBackend):
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
        "force_categorical": "bool",
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
        "force_categorical": False,
        "Termination criterion": "max_iter",
        "Resubmission of Failed N": 0,
    }
    constructors = {
        "MultiTaskGP": MultiTaskGP,
    }

    botorch_models = [
        "MultiTaskGP",
    ]
    ml_parameters = {}
    tags = ["Bayes", "MultiTask", "Machine", "ML"]
    _bound_adjustment_method = "RemoveTask"

    def __init__(self):
        super().__init__()

    def _BO_step(self) -> None:
        """
        Main BO step with task‐feature constraints:
          1. Train the surrogate on current data.
          2. Instantiate and constrain the acquisition function to fixed task features.
          3. Optimize the constrained acquisition to get next candidate.
          4. Reconstruct full candidate vectors including fixed features.
          5. Predict via model posterior and enqueue via out_data().
        """
        # 1. Train model
        self.log_mssg("Starting BO step with fixed task features", level="ok")
        x = self.ensure_writable(self.train_x)
        y = self.ensure_writable(self.train_y)
        self._init_model(x, y)
        self.log_mssg(
            f"Model trained on x.shape={x.shape}, y.shape={y.shape}",
            level="ok",
            indent=1,
        )

        # 2. Prepare acquisition
        acq_cls = self._get_acquisition_func()
        acq_args = self._get_acquisition_args()
        base_acq = acq_cls(**acq_args).to(torch.device("cpu"))
        self.log_mssg("Instantiated base acquisition function", level="ok", indent=1)

        # 3. Fix task features
        # 3. Fix task features
        # Safely grab linked indexes (empty list if not present)
        linked = getattr(self, "task_linked_physical_indexes", [])

        # Combine the always-present base features with any linked ones
        task_features = self.task_feature + linked

        # Start values from the task_value (scalar → single-element list; iterable → list)
        if isinstance(self.task_value, (int, float)):
            values = [self.task_value]
        else:
            values = list(self.task_value)

        # Pad with zeros for each linked index (if any)
        values += [0.0] * len(linked)

        self.log_mssg(
            f"Fixing features {task_features} to values {values}", level="ok", indent=1
        )

        fixed_acq = FixedFeatureAcquisitionFunction(
            acq_function=base_acq,
            d=x.shape[1],
            columns=task_features,
            values=values,
        )
        self.log_mssg(
            "Wrapped acquisition with fixed feature constraints", level="ok", indent=1
        )

        # 4. Optimize constrained acquisition
        next_pt, acq_val = self._optimize_acquisition_function(fixed_acq)
        self.log_mssg(
            f"Optimized acquisition: point shape={next_pt.shape}", level="ok", indent=1
        )

        # 5. Reconstruct full candidates
        full_pt = self._reconstruct_candidates(next_pt, task_features, values)
        self.log_mssg(
            f"Reconstructed full candidate: shape={full_pt.shape}", level="ok", indent=1
        )

        # 6. Predict and enqueue
        with torch.no_grad():
            post = self.model.posterior(full_pt)
            predict_y = self.normalizer_object.denormalize_mean_variance(
                (post.mean, post.variance)
            )
        self.out_data(full_pt, predicted_y=predict_y)
        self.log_mssg("BO step complete; next point enqueued", level="ok")

    def _reconstruct_candidates(
        self,
        candidate_unfixed: torch.Tensor,
        task_feature_index: List[int],
        task_feature_values: List[float],
    ) -> torch.Tensor:
        """
        Reinsert fixed task features into candidate vectors.

        :param candidate_unfixed: Tensor of shape (n, m) without fixed columns.
        :param task_feature_index: List of indices where features are fixed.
        :param task_feature_values: Corresponding values for each fixed feature.
        :return: Tensor of shape (n, m + len(task_feature_index)).
        """
        n, m = candidate_unfixed.shape
        total_dim = m + len(task_feature_index)
        full = torch.empty(
            n, total_dim, dtype=candidate_unfixed.dtype, device=candidate_unfixed.device
        )

        # Identify and fill unfixed columns
        fixed_set = set(task_feature_index)
        unfixed = [i for i in range(total_dim) if i not in fixed_set]
        full[:, unfixed] = candidate_unfixed

        # Insert fixed values
        for idx, val in zip(task_feature_index, task_feature_values):
            full[:, idx] = val

        self.log_mssg(
            f"Reconstructed candidates with fixed indices {task_feature_index}",
            level="ok",
            indent=1,
        )
        return full

    def _reconstruct_candidates_legacy(
        self,
        candidate_unfixed: torch.tensor,
        task_feature_index: int = -1,
        task_value=1.0,
    ):
        """Reconstructs the candidates in the correct shape as fixed feature removes
        the fixed feature from the candidates"""
        """
        Reconstruct the full candidate tensor by inserting the fixed fidelity feature at the specified index.

        Args:
            candidate_unfixed (torch.Tensor): Tensor of shape [n, m], where n is the number of candidates and m is the number of unfixed features.
            task_feature_index (int): The index where the fixed task feature should be inserted.
            task_value (float): The value of the fixed task feature to insert.

        Returns:
            torch.Tensor: Tensor of shape [n, m + 1], with the fixed fidelity feature inserted at the specified index.
        """
        (
            n,
            m,
        ) = (
            candidate_unfixed.shape
        )  # n: number of candidates, m: number of unfixed features
        candidate_full = torch.zeros(
            n, m + 1, dtype=candidate_unfixed.dtype, device=candidate_unfixed.device
        )

        if task_feature_index == 0:
            # Insert fidelity feature at the beginning
            candidate_full[:, 0] = task_value
            candidate_full[:, 1:] = candidate_unfixed
        elif task_feature_index == m:
            # Insert fidelity feature at the end
            candidate_full[:, :m] = candidate_unfixed
            candidate_full[:, m] = task_value
        else:
            # Insert fidelity feature in the middle
            candidate_full[:, :task_feature_index] = candidate_unfixed[
                :, :task_feature_index
            ]
            candidate_full[:, task_feature_index] = task_value
            candidate_full[:, task_feature_index + 1 :] = candidate_unfixed[
                :, task_feature_index:
            ]

        return candidate_full

    def _get_model_kwargs(self):
        """Get the keyword arguments for the model based on the parameters"""
        match self.parameters["Model"]:
            case "MultiTaskGP":
                outcome_transform = Standardize(m=1)
                return {
                    "task_feature": self.task_feature[0],
                    "outcome_transform": outcome_transform,
                    # "covar_module": self.kernel,
                }
            case _:
                raise ValueError("Unsupported model")

    def _get_acquisition_args(self):
        """Get the arguments for the acquisition function based on the parameters"""
        match self.parameters["Acquisition Function"]:
            case "EI":
                mt_obj = LinearMCObjective(weights=torch.tensor([1.0]))
                param = {
                    "model": self.model,
                    "best_f": self.train_y.max().clone(),
                    "objective": mt_obj,
                }
            case "UCB":
                mt_obj = ScalarizedPosteriorTransform(weights=torch.tensor([1.0]))
                param = {
                    "model": self.model,
                    "beta": self.parameters["Explorative Factor"],
                    "posterior_transform": mt_obj,
                    # "objective": mt_obj,
                }
            case "qEHVI":
                ref_point = self.train_y.min(dim=0)[0] - 0.1
                partition = NondominatedPartitioning(
                    ref_point=ref_point, Y=self.train_y
                )
                param = {
                    "model": self.model,
                    "ref_point": ref_point,
                    "partitioning": partition,
                }
            case _:
                raise ValueError("Unsupported acquisition function")

        return param
