"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import torch
from botorch.acquisition import FixedFeatureAcquisitionFunction
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.transforms import Standardize

from robrains.ml_modules.singlebayesianoptibackend import SingleBayesianOptiBackend


class ScopeAcceleratorFidelityBackend(SingleBayesianOptiBackend):
    """
    The scope accelerator is used to feed data for old optimisations into the system:

    Imagine the following scenario:
    Dima is working with the reaction: A+B -> P plus a bunch of parameters
    he has optimised it with RobERTA and has now an optimised reaction system. Now he has to
    start doing his substrate scope, where he will keep the parameters constant except B, which will be switched
    out for other substrates.

    This module allows to load all the data from the previous scopes and then have the system optimise the new one


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
            "SingleTaskMultiFidelityGP",
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
        "Model": "SingleTaskMultiFidelityGP",
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
        "SingleTaskMultiFidelityGP": SingleTaskMultiFidelityGP,
    }
    ml_parameters = {}
    tags = ["Bayes", "Multifidelity", "Machine", "ML"]
    _bound_adjustment_method = "RemoveFidelity"

    def __init__(self):
        super().__init__()

    def _BO_step(self) -> None:
        """
        Perform a single BO step with fidelity fixed at high value:
          1. Train surrogate model on current data.
          2. Instantiate acquisition function and wrap in FixedFeatureAcquisitionFunction
             fixing the fidelity feature to value 1.
          3. Optimize the fixed-feature acquisition to get the next candidate.
          4. Reconstruct full candidate tensor including the fidelity feature.
          5. Enqueue via out_data().
        """
        self.log_mssg("Starting fidelity‐fixed BO step", level="ok")

        # 1. Train model
        x = self.ensure_writable(self.train_x)
        y = self.ensure_writable(self.train_y)
        self._init_model(x, y)
        self.log_mssg(
            f"Model trained on x.shape={x.shape}, y.shape={y.shape}",
            level="ok",
            indent=1,
        )

        # 2. Build acquisition
        acq_cls = self._get_acquisition_func()
        acq_args = self._get_acquisition_args()
        acq = acq_cls(**acq_args).to(torch.device("cpu"))
        self.log_mssg("Base acquisition instantiated", level="ok", indent=1)

        fixed_acq = FixedFeatureAcquisitionFunction(
            acq_function=acq,
            d=x.shape[1],
            columns=self.fidelity_feature,
            values=[1],
        )
        self.log_mssg("Wrapped acquisition with fixed fidelity=1", level="ok", indent=1)

        # 3. Optimize
        next_pt, acq_val = self._optimize_acquisition_function(fixed_acq)
        self.log_mssg(
            f"Optimized fixed‐fidelity acquisition; point shape={next_pt.shape}",
            level="ok",
            indent=1,
        )

        # 4. Reconstruct full candidate
        full_pt = self._reconstruct_candidates(
            next_pt, fidelity_feature_index=self.fidelity_feature[0]
        )
        self.log_mssg(
            f"Reconstructed full candidate shape={full_pt.shape}", level="ok", indent=1
        )

        # 5. Enqueue
        self.out_data(full_pt)
        self.log_mssg("Fidelity‐fixed BO step complete; candidate enqueued", level="ok")

    def _reconstruct_candidates(
        self,
        candidate_unfixed: torch.Tensor,
        fidelity_feature_index: int = -1,
        fidelity_value: float = 1.0,
    ) -> torch.Tensor:
        """
        Insert a fixed fidelity feature into candidate vectors.

        :param candidate_unfixed: Tensor of shape (n, m) without fidelity column.
        :param fidelity_feature_index: Position to insert fidelity_value (0..m).
        :param fidelity_value: Value to insert at the fidelity index.
        :return: Tensor of shape (n, m+1) with fidelity column restored.
        """
        n, m = candidate_unfixed.shape
        self.log_mssg(
            f"Reconstructing {n} candidates by inserting fidelity at index {fidelity_feature_index}",
            level="ok",
        )

        full = torch.zeros(
            n, m + 1, dtype=candidate_unfixed.dtype, device=candidate_unfixed.device
        )
        if fidelity_feature_index == 0:
            full[:, 0] = fidelity_value
            full[:, 1:] = candidate_unfixed
        elif fidelity_feature_index == m:
            full[:, :m] = candidate_unfixed
            full[:, m] = fidelity_value
        else:
            full[:, :fidelity_feature_index] = candidate_unfixed[
                :, :fidelity_feature_index
            ]
            full[:, fidelity_feature_index] = fidelity_value
            full[:, fidelity_feature_index + 1 :] = candidate_unfixed[
                :, fidelity_feature_index:
            ]

        self.log_mssg(
            f"Candidate reconstruction complete; full shape={full.shape}",
            level="ok",
            indent=1,
        )
        return full

    def _get_model_kwargs(self):
        """Get the keyword arguments for the model based on the parameters"""
        match self.parameters["Model"]:
            case "SingleTaskMultiFidelityGP":
                return {
                    "outcome_transform": Standardize(m=1),
                    "data_fidelities": self.fidelity_feature,
                    # "covar_module": self.kernel,
                }
            case _:
                raise ValueError("Unsupported model")
