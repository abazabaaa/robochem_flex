"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: here we can build all the custom models we want. Every tom dick harry jerry and their grandmas have used GPs why not do smth spicy?

"""

# Standard Library Imports
from typing import List, Optional, Tuple, Union

# Third-Party Imports
import torch
from torch import Tensor

from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior


class OutcomeTransformMixin:
    """
    Mixin to handle the outcome transforms and tensor validation, used to mimick the GPYtorchModel class
    """

    def __init__(
        self,
        train_Y: Tensor,
        train_X: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        """
        :param train_Y: the y tensor to apply the outcome transform to
        :param outcome_transform: the outcome transform to apply
        :param kwargs:
        """
        self.outcome_transform = outcome_transform
        self._num_outputs = train_Y.shape[-1] if train_Y.ndim > 1 else 1
        self._set_dimensions(train_Y=train_Y, train_X=train_X)
        self.train_Y = self._apply_outcome_transform(train_Y)

    def _set_dimensions(self, train_X: Tensor, train_Y: Tensor) -> None:
        r"""Store the number of outputs and the batch shape.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.
        """
        self._num_outputs = train_Y.shape[-1]
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )

    @staticmethod
    def get_batch_dimensions(
        train_X: Tensor, train_Y: Tensor
    ) -> Tuple[torch.Size, torch.Size]:
        r"""Get the raw batch shape and output-augmented batch shape of the inputs.

        Args:
            train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
                features.
            train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
                training observations.

        Returns:
            2-element tuple containing

            - The `input_batch_shape`
            - The output-augmented batch shape: `input_batch_shape x (m)`
        """
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = input_batch_shape
        num_outputs = train_Y.shape[-1]
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        return input_batch_shape, aug_batch_shape

    @property
    def num_outputs(self) -> int:
        """returns the number of outputs"""
        return self._num_outputs

    @property
    def batch_shape(self) -> torch.Size:
        """returns the batch shape"""
        return self._input_batch_shape

    def _validate_tensor_args(
        self, X: Tensor, Y: Tensor, Yvar: Optional[Tensor] = None
    ):
        """
        Validates the shape consistency of the input and outputs
        :param X: The X tensor
        :param Y: The Y tensor
        :param Yvar: Not sure what this is
        :return: None, raises errors in case
        """
        assert isinstance(X, Tensor), f"Expected X to be a tensor, got {type(X)}"
        assert isinstance(Y, Tensor), f"Expected Y to be a tensor, got {type(Y)}"
        assert (
            X.shape[0] == Y.shape[0]
        ), f"X and Y must have the same number of rows, got {X.shape[0]} and {Y.shape[0]}"
        if Yvar is not None:
            assert isinstance(
                Yvar, Tensor
            ), f"Expected Yvar to be a tensor, got {type(Yvar)}"
            assert (
                Yvar.shape == Y.shape
            ), f"Expected Yvar to have the same shape as Y, got {Yvar.shape} and {Y.shape}"

    def _apply_outcome_transform(
        self, Y: Tensor, YVar: Optional[Tensor] = None
    ) -> Tensor:
        """
        Applies the outcome transform to the Y tensor
        :param Y: The Y tensor
        :param YVar: The YVar tensor
        :return: The transformed Y tensor
        """
        if self.outcome_transform is not None:
            Y, YVar = self.outcome_transform.forward(Y, YVar)
        return Y

    def _untransform_posterior(self, posterior: Posterior) -> Posterior:
        """
        Applies the outcome transform to the posterior
        :param posterior: The posterior to apply the transform to
        :return: The transformed posterior
        """
        if self.outcome_transform is not None:
            posterior = self.outcome_transform.untransform_posterior(posterior)
        return posterior


class PosteriorMakingMixin:
    def __init__(
        self,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device("cpu"),
    ):
        self._dtype = dtype
        self._device = device

    def _build_posterior(
        self,
        mean: Tensor,  # Shape: [n*q, t]
        variance: Tensor,  # Shape: [n*q, t]
        X_shape: torch.Size,  # Shape: [n, q, v] or [n, q]
        observation_noise: Union[bool, Tensor] = False,
        output_indices: Optional[List[int]] = None,
    ) -> GPyTorchPosterior:
        """
        Constructs a GPyTorchPosterior given the mean and variance predictions.

        Args:
            mean (Tensor): Predicted mean tensor of shape [n*q, t].
            variance (Tensor): Predicted variance tensor of shape [n*q, t].
            X_shape (torch.Size): Original shape of X excluding feature dimension.
            observation_noise (bool or Tensor, optional): Observation noise to add.
            output_indices (List[int], optional): Indices of outputs to consider.

        Returns:
            GPyTorchPosterior: The constructed posterior object.
        """
        # Ensure mean and variance are on the correct device and dtype
        mean = mean.to(dtype=self._dtype, device=self._device)
        variance = variance.to(dtype=self._dtype, device=self._device)

        # Select specific output indices if provided
        if output_indices is not None:
            mean = mean[:, output_indices]  # Shape: [n*q, selected_t]
            variance = variance[:, output_indices]  # Shape: [n*q, selected_t]
            t = len(output_indices)
        else:
            t = self._num_outputs  # Total number of outputs

        # Extract n and q from X_shape
        if len(X_shape) == 2:
            n, v = X_shape
            q = 1
        elif len(X_shape) == 3:
            n, q, v = X_shape
        else:
            raise ValueError(f"X_shape must have 2 or 3 dimensions, got {len(X_shape)}")

        if t == 1:
            # Single-output case
            # Reshape mean and variance to [n, q]
            mean_t = mean.view(n, q)  # Shape: [n, q]
            variance_t = variance.view(n, q)  # Shape: [n, q]

            # Add observation noise if specified
            if observation_noise is True:
                variance_t = variance_t + 1e-6  # Small noise for numerical stability
            elif isinstance(observation_noise, Tensor):
                if observation_noise.numel() != n * q:
                    raise ValueError("Observation noise tensor must have n*q elements.")
                obs_noise = observation_noise.view(n, q)
                variance_t = variance_t + obs_noise

            # Create diagonal covariance matrices for each [n, q]
            # Shape: [n, q, q]
            covar = torch.diag_embed(variance_t)  # Each [q, q] is diagonal

            # Create MultivariateNormal distribution
            # Mean shape: [n, q]
            mvn = MultivariateNormal(
                mean_t, covariance_matrix=covar
            )  # Mean: [n, q], Covar: [n, q, q]

            # Construct the GPyTorchPosterior
            posterior = GPyTorchPosterior(mvn)

            # Apply any necessary transformations
            posterior = self._untransform_posterior(posterior)

            # Convert to standard Posterior
            return posterior

        else:
            # Multi-output case
            # Reshape mean and variance to [n, q, t]
            mean_t = mean.view(n, q, t)  # Shape: [n, q, t]
            variance_t = variance.view(n, q, t)  # Shape: [n, q, t]

            # Add observation noise if specified
            if observation_noise is True:
                variance_t = variance_t + 1e-6  # Small noise for numerical stability
            elif isinstance(observation_noise, Tensor):
                if observation_noise.numel() != n * q * t:
                    raise ValueError(
                        "Observation noise tensor must have n*q*t elements."
                    )
                obs_noise = observation_noise.view(n, q, t)
                variance_t = variance_t + obs_noise

            # Create diagonal covariance matrices for each [n, q, t]
            # Shape: [n, q*t, q*t]
            covar = torch.diag_embed(
                variance_t.view(n, q * t)
            )  # Each [q*t, q*t] is diagonal

            # Reshape mean to [n, q*t]
            # mean_flat = mean_t.view(n, q * t)  # Shape: [n, q*t]

            # Create MultitaskMultivariateNormal distribution
            mvn = MultitaskMultivariateNormal(
                mean_t, covariance_matrix=covar
            )  # Mean: [n, q*t], Covar: [n, q*t, q*t]

            # Construct the GPyTorchPosterior
            posterior = GPyTorchPosterior(mvn)

            # Apply any necessary transformations
            posterior = self._untransform_posterior(posterior)

            # Wrap the posterior to reshape mean appropriately
            # reshaped_posterior = ReshapedMultitaskPosterior(posterior, q=q, t=t)

            return posterior
