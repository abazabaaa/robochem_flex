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
from sklearn.svm import SVR
from torch import Tensor

from robrains.custom_models.custom_models import (
    OutcomeTransformMixin,
    PosteriorMakingMixin,
)


class SVRSurrogate(Model, OutcomeTransformMixin, PosteriorMakingMixin):
    """
    Support Vector Regression surrogate model using bootstrapping for uncertainty estimation.

    This model trains an ensemble of SVR models on bootstrapped samples of the training data.
    The predictive mean and variance are estimated from the ensemble predictions, which are then
    used to construct the posterior distribution required for Bayesian Optimization.

    **Attributes:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples,)`.
        n_bootstrap (int): Number of bootstrap samples/models in the ensemble.
        models (List[SVR]): List of trained SVR models.

    **Args:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples,)`.
        n_bootstrap (int, optional): Number of bootstrap samples/models. Default is `10`.
        **kwargs: Additional keyword arguments passed to the SVR constructor.

    **Methods:**
        __init__(self, train_X, train_Y, n_bootstrap=10, **kwargs):
            Initializes and trains the ensemble of SVR models.

        posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
            Computes the posterior over model outputs at the provided points.

    """

    param_dict = {
        "n_bootstrap": "int",
    }
    param_defaults = {
        "n_bootstrap": 10,
    }

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        n_bootstrap: int = 10,
        **kwargs,
    ):
        """
        Initializes and trains the ensemble of SVR models using bootstrapping.

        **Args:**
            train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
            train_Y (Tensor): Training target data of shape `(n_samples,)`.
            n_bootstrap (int, optional): Number of bootstrap samples/models. Default is `10`.
            **kwargs: Additional keyword arguments passed to the `SVR` constructor.

        """
        super().__init__()
        OutcomeTransformMixin.__init__(
            self, train_Y, train_X, outcome_transform=outcome_transform
        )
        PosteriorMakingMixin.__init__(self, dtype=train_Y.dtype, device=train_Y.device)
        self._validate_tensor_args(train_X, self.train_Y)

        self.train_X = train_X
        self.n_bootstrap = n_bootstrap
        self.models = []

        # Bootstrapping to estimate uncertainty
        n_samples = train_X.shape[0]
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = train_X[indices]
            Y_boot = train_Y[indices]
            svr = SVR(**kwargs)
            svr.fit(X_boot.numpy(), Y_boot.numpy().ravel())
            self.models.append(svr)

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

        """
        assert isinstance(
            X, Tensor
        ), f"Tensor X must be a tensor, actual type {type(X)}"

        # Flatten X for prediction
        X_flat = X.view(-1, X.size(-1)).numpy()

        # Collect predictions from all models
        predictions = np.array(
            [model.predict(X_flat) for model in self.models]
        )  # Shape: n_models x n_samples

        # Compute mean and variance
        mean = predictions.mean(axis=0)  # Shape: n_samples
        variance = predictions.var(axis=0)  # Shape: n_samples

        posterior = self._build_posterior(
            mean=torch.from_numpy(mean).squeeze(-1),
            variance=torch.from_numpy(variance).squeeze(-1),
            X_shape=X.shape,
            observation_noise=observation_noise,
            output_indices=output_indices,
        )
        return posterior
