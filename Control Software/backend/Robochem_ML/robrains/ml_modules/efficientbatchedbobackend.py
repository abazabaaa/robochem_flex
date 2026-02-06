"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Tuple, Type

import torch
from botorch.models import SingleTaskGP, MixedSingleTaskGP

from robrains.custom_models import (
    RandomForestSurrogate,
    NeuralNetworkSurrogate,
    BayesianNeuralNetworkSurrogate,
    SVRSurrogate,
)
from robrains.ml_modules.singlebayesianoptibackend import SingleBayesianOptiBackend


class EfficientBatchedBOBackend(SingleBayesianOptiBackend):
    """
    This class is made to handle a specific batched bayesian optimisation methodology.
    here the model generates q candidates that are geared towards optimising and q candidates
    that are forced to explore where the model has less confidence. This is done by using two
    acqisition function.

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
        "Exploitative Function": ["EI", "PI", "UCB", "qEHVI", "qLogNEHVI"],
        "Explorative Function": ["qMVE", "qKG", "MaxVariance", "qNegIPV"],
        "Initialisation Method": ["LHS", "Random"],
        "Number of initial points": "int",
        "Number of total points": "int",
        "Number of Experiments per batch": "int",
        "Explorative Factor": "float",
        "mves_num_fantasies": "int",
        "mves_num_mv_samples": "int",
        "mves_num_y_samples": "int",
        "MV_num_samples": "int",
        "kg_num_fantasies": "int",
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
        "Exploitative Function": "UCB",
        "Explorative Function": "qMVE",
        "Initialisation Method": "LHS",
        "Number of initial points": 10,
        "Number of total points": 100,
        "Number of Experiments per batch": 1,
        "Explorative Factor": 0.1,
        "mves_num_fantasies": 16,
        "mves_num_mv_samples": 10,
        "mves_num_y_samples": 128,
        "MV_num_samples": 1000,
        "kg_num_fantasies": 64,
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
    ]
    custom_models = [
        "RandomForest",
        "NeuralNetworkEnsemble",
        "BayesianNeuralNetwork",
        "SVR",
    ]
    ml_parameters = {}
    tags = ["Bayes", "SingleTask", "Machine", "ML"]

    def _get_acquisition_func(
        self,
    ) -> Tuple[Type, Type]:
        """
        Determine the acquisition function *classes* (constructors) for both
        exploitative and explorative acquisitions, based on user-specified parameters.

        Returns:
            (constructor_exploitative, constructor_explorative)
            Each is the class of the acquisition function you will instantiate later.
        """
        # Decide if we're single or batched
        num_experiments = self.parameters["Number of Experiments per batch"]
        is_single = num_experiments == 1

        exploit_fn_name = self.parameters["Exploitative Function"]
        explore_fn_name = self.parameters["Explorative Function"]

        constructor_exploitative = self._select_acq_constructor(
            exploit_fn_name, is_single
        )
        constructor_explorative = self._select_acq_constructor(
            explore_fn_name, is_single
        )

        return constructor_exploitative, constructor_explorative

    def _get_acquisition_args(
        self,
    ) -> Tuple[dict, dict]:
        """
        Get the keyword-argument dicts for both the exploitative and explorative
        acquisition functions, based on user parameters.

        Returns:
            (params_exploitative, params_explorative)
        """
        exploit_fn_name = self.parameters["Exploitative Function"]
        explore_fn_name = self.parameters["Explorative Function"]

        params_exploitative = self._select_acq_args(fn_name=exploit_fn_name)
        params_explorative = self._select_acq_args(fn_name=explore_fn_name)

        return params_exploitative, params_explorative

    def _BO_step(self) -> None:
        """
        Perform dual-phase BO step:
          1. Train the surrogate on current data.
          2. Optimize exploitative acquisition to get first point.
          3. Mark exploit point as pending for explorative acquisition.
          4. Optimize explorative acquisition to get second point.
          5. Concatenate both and enqueue via out_data().
        """

        self.log_mssg("Starting dual acquisition BO step")

        # 1. Prepare data and train model
        x = self.ensure_writable(self.train_x)
        y = self.ensure_writable(self.train_y)
        y_var = getattr(self, "train_y_var", None)
        if y_var is not None:
            y_var = self.ensure_writable(y_var)
        self._init_model(x, y, y_var=y_var)
        self.log_mssg(
            f"Model trained on x.shape={x.shape}, y.shape={y.shape}",
            indent=1,
        )

        # 2. Retrieve exploitative & explorative acquisitions and their args
        acq_exp_cls, acq_expl_cls = self._get_acquisition_func()
        args_exp, args_expl = self._get_acquisition_args()
        device = torch.device("cpu")

        # 3. Instantiate exploitative acquisition and optimize
        acq_exp = acq_exp_cls(**args_exp).to(device)
        self.log_mssg("Optimizing exploitative acquisition", indent=1)
        self.log_mssg(f"Acquisition class: {acq_exp_cls.__name__}", indent=2)
        self.log_mssg(f"Acquisition args: {args_exp}", indent=2)
        pt_exp, val_exp = self._optimize_acquisition_function(acq_exp)
        self.log_mssg(f"Exploit point shape={pt_exp.shape}, value={val_exp}", indent=2)
        self.log_mssg(f"Pending exploit point: {pt_exp}", indent=2)

        # 4. Instantiate explorative acquisition, set pending, and optimize
        acq_expl = acq_expl_cls(**args_expl).to(device)
        acq_expl.set_X_pending(pt_exp)
        self.log_mssg(
            "Optimizing explorative acquisition with pending exploit",
            indent=1,
        )
        self.log_mssg(
            f"Acquisition class: {acq_expl_cls.__name__}", level="ok", indent=2
        )
        self.log_mssg(f"Acquisition args: {args_expl}", level="ok", indent=2)
        pt_expl, val_expl = self._optimize_acquisition_function(acq_expl)
        self.log_mssg(
            f"Explore point shape={pt_expl.shape}, value={val_expl}",
            level="ok",
            indent=2,
        )
        self.log_mssg(f"Pending explore point: {pt_expl}", level="ok", indent=2)

        # 5. Combine, predict, and enqueue
        next_pts = torch.cat([pt_exp, pt_expl], dim=0)
        with torch.no_grad():
            post = self.model.posterior(next_pts)
            mean, var = post.mean, post.variance
            pred = self.normalizer_object.denormalize_mean_variance((mean, var))

        self.log_mssg(f"Predicted y: {pred}", level="ok", indent=1)
        self.out_data(next_pts, predicted_y=pred)
        self.log_mssg("Dual acquisition BO step complete; points enqueued", level="ok")
