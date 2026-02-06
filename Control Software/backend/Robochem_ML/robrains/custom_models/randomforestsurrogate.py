"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Optional, List, Union

import numpy as np
import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior
from sklearn.ensemble import RandomForestRegressor
from torch import Tensor

from robrains.custom_models.custom_models import (
    OutcomeTransformMixin,
    PosteriorMakingMixin,
)


class RandomForestSurrogate(Model, OutcomeTransformMixin, PosteriorMakingMixin):
    """
    Random Forest surrogate model for Bayesian Optimization using BoTorch.

    This model uses scikit-learn's `RandomForestRegressor` to fit the training data.
    It aggregates predictions from all trees in the forest to compute the predictive
    mean and variance, which are then used to construct the posterior distribution
    required for Bayesian Optimization.

    **Attributes:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        model (RandomForestRegressor): The trained Random Forest model.

    **Args:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        **kwargs: Additional keyword arguments passed to the `RandomForestRegressor` constructor.

    **Methods:**
        __init__(self, train_X, train_Y, **kwargs):
            Initializes and trains the Random Forest model.

        posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
            Computes the posterior over model outputs at the provided points.

    """

    param_dict = {
        "n_estimators": "int",
    }
    param_defaults = {
        "n_estimators": 100,
    }

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        """
        Initializes and trains the Random Forest surrogate model.

        **Args:**
            train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
            train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
            **kwargs: Additional keyword arguments passed to the `RandomForestRegressor` constructor.

        **Notes:**
            - The model is trained using scikit-learn's `RandomForestRegressor`.
            - Ensures that the training data are tensors and have matching sample sizes.
            - Stores the trained model for making predictions in the `posterior` method.

        """
        super().__init__()

        OutcomeTransformMixin.__init__(
            self, train_Y, train_X, outcome_transform=outcome_transform
        )
        PosteriorMakingMixin.__init__(self, dtype=train_Y.dtype, device=train_Y.device)

        self._validate_tensor_args(train_X, self.train_Y)

        self.train_X = train_X.numpy()
        if self.train_Y.shape[1] == 1:
            self.train_Y = self.train_Y.ravel()

        sample_weights = kwargs.pop("sample_weights", None)
        n_estimators = kwargs.pop("n_estimators", 100)
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
        self.model.fit(self.train_X, self.train_Y, sample_weight=sample_weights)

    @property
    def num_outputs(self) -> int:
        return OutcomeTransformMixin.num_outputs.fget(self)

    @property
    def batch_shape(self) -> torch.Size:
        return OutcomeTransformMixin.batch_shape.fget(self)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> Posterior:
        """
        Computes the posterior over model outputs at the provided points.

        **Args:**
            X (Tensor): A `b x q x d`-dimensional tensor of input data where `b` is the batch size,
                `q` is the number of query points, and `d` is the input dimensionality.
            output_indices (List[int], optional): Indices of the outputs to consider. If `None`, all outputs are used.
            observation_noise (bool or Tensor, optional): If `True`, includes observation noise in the posterior.
                If a tensor, specifies the observation noise levels. Default is `False`.
            posterior_transform (PosteriorTransform, optional): An optional transform applied to the posterior. Default is `None`.
            **kwargs: Additional keyword arguments.

        **Returns:**
            Posterior: A `Posterior` object representing the distribution over outputs at the query points.

        **Notes:**
            - Flattens batched input `X` for compatibility with scikit-learn's prediction methods.
            - Aggregates predictions from all trees to compute the predictive mean and variance.
            - Assumes independence between outputs when constructing the covariance matrix.
            - Adds observation noise to the covariance matrix if specified.
            - Constructs a Multivariate Normal distribution with the predictive mean and covariance.

        """
        assert isinstance(
            X, Tensor
        ), f"Tensor X must be a tensor, actual type {type(X)}"

        # Move X to CPU for scikit-learn compatibility
        X_cpu = X.cpu()
        original_shape = X_cpu.shape  # batch_shape x q x d

        # Flatten X for prediction
        X_flat = (
            X_cpu.view(-1, X_cpu.size(-1)).detach().numpy()
        )  # Shape: (batch_size * q) x d

        # Get predictions from all trees
        predictions = np.array(
            [est.predict(X_flat) for est in self.model.estimators_]
        )  # Shape: n_trees x n_samples x n_outputs

        # Compute mean and variance across trees
        mean = predictions.mean(axis=0)  # Shape: n_samples x n_outputs
        variance = predictions.var(axis=0)  # Shape: n_samples x n_outputs

        posterior = self._build_posterior(
            mean=torch.from_numpy(mean),
            variance=torch.from_numpy(variance),
            X_shape=original_shape,
            observation_noise=observation_noise,
            output_indices=output_indices,
        )

        return posterior

    def __repr__(self):
        """Representation of model parameters"""
        return f"RandomForestSurrogate(n_estimators={self.n_estimators})"
