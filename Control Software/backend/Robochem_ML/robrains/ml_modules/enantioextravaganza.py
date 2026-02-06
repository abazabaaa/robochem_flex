"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import traceback

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import LinearMCObjective, ScalarizedPosteriorTransform

from .singlebayesianoptibackend import SingleBayesianOptiBackend


class EnantioExtravaganzaBackend(SingleBayesianOptiBackend):
    """
    Hi Simo, This is the class you need to use for the enantiomeric thing,
    the idea is: 1 point highly exploitatlive per target, n points that map the pareto front

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
        "Acquisition Function Single": ["EI", "PI", "UCB"],
        "Acquisition Function Multiple": ["qLogNParEGO", "qEHVI", "qPPES"],
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
        "Acquisition Function Single": "UCB",
        "Acquisition Function Multiple": "qLogNParEGO",
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

    def _BO_step(self) -> None:
        """
        Execute one Bayesian optimization step:
          1. Prepare training data and optional variance.
          2. Initialize the surrogate model.
          3. Build and optimize acquisition functions to propose next point(s).
          4. Predict on the new point(s) and enqueue via out_data().
        """
        # 1. Prepare data
        self.log_mssg("Starting BO step.", level="ok")
        x = self.ensure_writable(self.train_x)
        y = self.ensure_writable(self.train_y)
        y_var = (
            self.ensure_writable(self.train_y_var)
            if hasattr(self, "train_y_var")
            else None
        )
        n_targets = y.shape[1]
        self.log_mssg(
            f"Prepared training data | X: {x.shape}, Y: {y.shape}, Y_var: {None if y_var is None else y_var.shape}",
            level="ok",
            indent=1,
        )

        # 2. Train surrogate model
        self._init_model(x, y, y_var=y_var)
        self.log_mssg("Surrogate model trained.", level="ok", indent=1)

        # 3a. Single-objective acquisitions per target
        self.log_mssg(
            "Building single-objective acquisitions for each target.",
            level="ok",
            indent=1,
        )
        acq_single_ctor = self._select_acq_constructor(
            fn_name=self.parameters["Acquisition Function Single"], is_single=False
        )
        acq_single_args = self._select_acq_args(
            fn_name=self.parameters["Acquisition Function Single"], is_single="False"
        )
        self.log_mssg(
            f"Single-objective constructor: {acq_single_ctor.__name__}", indent=2
        )
        self.log_mssg(f"Single-objective args: {acq_single_args}", indent=2)

        pending_stack = []  # collect pending points
        for i in range(n_targets):
            # one-hot weights for target i
            weights = torch.zeros(n_targets, device=x.device, dtype=torch.float64)
            weights[i] = 1.0
            acq_single_args["posterior_transform"] = ScalarizedPosteriorTransform(
                weights=weights
            )

            acqf = acq_single_ctor(**acq_single_args).to(torch.device("cpu"))
            # register previous pending points
            if pending_stack:
                X_pending = torch.cat(pending_stack, dim=0)
                acqf.set_X_pending(X_pending)
            self.log_mssg(
                f"Acqfn for target {i} instantiated; set {len(pending_stack)} pending.",
                level="ok",
                indent=2,
            )

            # optimize
            next_pt, acq_val = self._optimize_acquisition_function(acqf)
            self.log_mssg(
                f"Target {i}: optimized candidate shape {next_pt.shape}, acqval={acq_val:.4f}",
                level="ok",
                indent=2,
            )

            pending_stack.append(next_pt)

        # convert list of tensors to a single tensor
        X_pending_all = torch.cat(pending_stack, dim=0)
        self.log_mssg(
            f"Collected {len(pending_stack)} single-target candidates; stacked shape {X_pending_all.shape}",
            level="ok",
            indent=1,
        )

        batch_size_backup = self.parameters["Number of Experiments per batch"]
        batch_size = y.shape[1]
        self.parameters["Number of Experiments per batch"] = batch_size
        # 3b. Multi-objective acquisition over pareto front
        self.log_mssg("Building multi-objective acquisition.", level="ok", indent=1)
        acq_multi_ctor = self._select_acq_constructor(
            fn_name=self.parameters["Acquisition Function Multiple"], is_single=True
        )
        acq_multi_args = self._select_acq_args(
            fn_name=self.parameters["Acquisition Function Multiple"]
        )
        acqf_multi = acq_multi_ctor(**acq_multi_args).to(torch.device("cpu"))
        acqf_multi.set_X_pending(X_pending_all)
        self.log_mssg(
            f"Multi-objective acqfn instantiated; pending shape {X_pending_all.shape}",
            level="ok",
            indent=2,
        )

        next_pt_batch, acq_val_batch = self._optimize_acquisition_function(acqf_multi)
        self.log_mssg(
            f"Multi-objective optimized batch shape {next_pt_batch.shape}, acqval={acq_val_batch:.4f}",
            level="ok",
            indent=1,
        )
        # append final candidate
        all_next_pts = torch.cat([X_pending_all, next_pt_batch], dim=0)
        self.log_mssg(
            f"Total proposed points: {all_next_pts.shape[0]}", level="ok", indent=1
        )
        # restore batch size
        self.parameters["Number of Experiments per batch"] = batch_size_backup

        # remove any points that are very similar to each other
        all_next_pts = self.filter_close_points(all_next_pts, threshold=1e-2)

        # 4. Predict & enqueue
        with torch.no_grad():
            post = self.model.posterior(all_next_pts)
            mean, var = post.mean, post.variance
            pred_y = self.normalizer_object.denormalize_mean_variance((mean, var))

        self.log_mssg("Predictions for final batch obtained.", level="ok", indent=1)
        self.log_mssg(f"Predicted y: {pred_y}", level="none", indent=2)
        self.out_data(all_next_pts, predicted_y=pred_y)
        self.log_mssg("BO step complete; points enqueued.", level="ok")

    def filter_close_points(self, X: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Greedily keep points in X that are at least `threshold` apart.

        Args:
            X: Tensor of shape (N, d)
            threshold: minimal allowed pairwise distance

        Returns:
            Tensor of filtered points (M, d), M <= N
        """
        if X.numel() == 0:
            return X
        keep = []
        for x in X:
            if not keep:
                keep.append(x)
                continue
            # stack existing kept points
            K = torch.stack(keep)  # (k, d)
            # compute distances to all kept
            dists = torch.norm(K - x.unsqueeze(0), dim=1)  # (k,)
            if torch.all(dists >= threshold):
                keep.append(x)
        return torch.stack(keep)
