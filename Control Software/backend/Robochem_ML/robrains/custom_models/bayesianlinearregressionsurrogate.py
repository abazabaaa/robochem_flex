"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Optional, List, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior
from torch import Tensor

from robrains.custom_models.custom_models import (
    OutcomeTransformMixin,
    PosteriorMakingMixin,
)


class BayesianLinearRegressionSurrogate(
    Model, OutcomeTransformMixin, PosteriorMakingMixin
):
    """
    Bayesian Linear Regression surrogate model with analytical posterior computation.

    This model assumes a linear relationship between inputs and outputs and computes the posterior
    distribution of the weights analytically based on Bayesian inference. The predictive mean and variance
    at new inputs are derived from the posterior parameters of the weights.

    **Attributes:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        alpha (float): Prior precision (inverse variance) of the weights.
        beta (float): Noise precision (inverse variance) of the observations.
        m_N (Tensor): Posterior mean of the weights, of shape `(n_features + 1, n_outputs)`.
        S_N (Tensor): Posterior covariance matrix of the weights, of shape `(n_features + 1, n_features + 1)`.

    **Args:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        alpha (float, optional): Prior precision of the weights. Default is `1.0`.
        beta (float, optional): Noise precision of the observations. Default is `1.0`.
        **kwargs: Additional keyword arguments.

    **Methods:**
        __init__(self, train_X, train_Y, alpha=1.0, beta=1.0, **kwargs):
            Initializes the Bayesian Linear Regression model and computes the posterior parameters.

        posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
            Computes the posterior over model outputs at the provided points.

    """

    param_dict = {
        "alpha": "float",
        "beta": "float",
    }
    param_defaults = {"alpha": 1.0, "beta": 1.0}

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        **kwargs,
    ):
        """
        Initializes the Bayesian Linear Regression model and computes the posterior parameters.

        **Args:**
            train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
            train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
            alpha (float, optional): Prior precision (inverse variance) of the weights. Default is `1.0`.
            beta (float, optional): Noise precision (inverse variance) of the observations. Default is `1.0`.
            **kwargs: Additional keyword arguments.

        **Notes:**
            - Adds a bias term to the input data.
            - Computes the posterior mean (`m_N`) and covariance (`S_N`) of the weights using analytical expressions.

        """
        super().__init__()
        OutcomeTransformMixin.__init__(
            self, train_Y, train_X, outcome_transform=outcome_transform
        )
        PosteriorMakingMixin.__init__(self, dtype=train_Y.dtype, device=train_Y.device)
        self._validate_tensor_args(train_X, self.train_Y)
        self.train_X = train_X
        self.alpha = alpha  # Prior precision (inverse variance)
        self.beta = beta  # Noise precision

        # Add bias term
        X = torch.cat(
            [train_X, torch.ones(train_X.size(0), 1)], dim=1
        )  # Shape: n_samples x (d+1)
        Y = train_Y

        # Compute posterior parameters
        XtX = X.t() @ X
        S_N_inv = self.alpha * torch.eye(X.size(1)) + self.beta * XtX
        self.S_N = torch.inverse(S_N_inv)
        self.m_N = self.beta * self.S_N @ X.t() @ Y  # Shape: (d+1) x n_outputs

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
            - Adds a bias term to the input data.
            - Computes the predictive mean and variance using the posterior parameters of the weights.
            - Constructs a Multivariate Normal distribution with the predictive mean and covariance.

        """

        assert isinstance(
            X, Tensor
        ), f"Tensor X must be a tensor, actual type {type(X)}"

        # Add bias term
        bias = torch.ones(*X.shape[:-1], 1, device=X.device, dtype=X.dtype)
        X_new = torch.cat([X, bias], dim=-1)  # Shape: n_test x (d+1)

        # Predictive mean and variance
        mean = X_new @ self.m_N  # Shape: n_test x n_outputs
        tmp = X_new @ self.S_N
        variance = (1 / self.beta) + torch.sum(
            tmp * X_new, dim=-1, keepdim=True
        )  # Shape: n_test x n_outputs

        posterior = self._build_posterior(
            mean=mean.squeeze(-1).view(-1),
            variance=variance.squeeze(-1).view(-1),
            X_shape=X.shape,
            observation_noise=observation_noise,
            output_indices=output_indices,
        )

        return posterior
