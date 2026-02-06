"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import List, Optional, Union

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior

from torch import nn as nn, Tensor, optim

from robrains.custom_models.custom_models import (
    OutcomeTransformMixin,
    PosteriorMakingMixin,
)


class BNNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
    ):
        super(BNNModel, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # Use Sigmoid if outputs are between 0 and 1
        self.model = nn.Sequential(*layers)
        self.model = self.model.double()

    def forward(self, x):
        return self.model(x)


class BayesianNeuralNetworkSurrogate(
    Model, OutcomeTransformMixin, PosteriorMakingMixin
):
    """
    Bayesian Neural Network surrogate model using Monte Carlo Dropout for uncertainty estimation.

    This model implements a Bayesian Neural Network by using dropout layers during both training
    and inference to approximate Bayesian inference. Multiple stochastic forward passes are performed
    to estimate the predictive mean and variance, which are then used to construct the posterior
    distribution required for Bayesian Optimization.

    **Attributes:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        T (int): Number of Monte Carlo samples for inference.
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output targets.
        hidden_dims (List[int]): List of hidden layer sizes.
        model (nn.Module): The Bayesian Neural Network model.

    **Args:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        num_epochs (int, optional): Number of training epochs. Default is `100`.
        batch_size (int, optional): Batch size for training. Default is `32`.
        learning_rate (float, optional): Learning rate for the optimizer. Default is `1e-3`.
        dropout_rate (float, optional): Dropout rate for the dropout layers. Default is `0.1`.
        hidden_layer_multiplier (float, optional): Multiplier to decrease the size of hidden layers. Default is `0.75`.
        T (int, optional): Number of Monte Carlo samples for inference. Default is `100`.
        **kwargs: Additional keyword arguments.

    **Methods:**
        __init__(self, train_X, train_Y, num_epochs=100, batch_size=32, learning_rate=1e-3,
                 dropout_rate=0.1, hidden_layer_multiplier=0.75, T=100, **kwargs):
            Initializes and trains the Bayesian Neural Network model.

        posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
            Computes the posterior over model outputs at the provided points.

    """

    param_dict = {
        "num_epochs": "int",
        "batch_size": "int",
        "learning_rate": "float",
        "dropout_rate": "float",
        "hidden_layer_multiplier": "float",
        "T": "int",
    }
    param_defaults = {
        "num_epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "dropout_rate": 0.1,
        "hidden_layer_multiplier": 0.75,
        "T": 100,
    }

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        dropout_rate: float = 1.0,
        hidden_layer_multiplier: float = 0.75,
        T: int = 100,  # Number of MC samples
        **kwargs,
    ):
        """
        Initializes and trains the Bayesian Neural Network surrogate model.

        **Args:**
            train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
            train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
            num_epochs (int, optional): Number of training epochs. Default is `100`.
            batch_size (int, optional): Batch size for training. Default is `32`.
            learning_rate (float, optional): Learning rate for the optimizer. Default is `1e-3`.
            dropout_rate (float, optional): Dropout rate for the dropout layers. Default is `0.1`.
            hidden_layer_multiplier (float, optional): Multiplier to decrease the size of hidden layers. Default is `0.75`.
            T (int, optional): Number of Monte Carlo samples for inference. Default is `100`.
            **kwargs: Additional keyword arguments.

        """
        super().__init__()
        OutcomeTransformMixin.__init__(
            self, train_Y, train_X, outcome_transform=outcome_transform
        )
        PosteriorMakingMixin.__init__(self, dtype=train_Y.dtype, device=train_Y.device)
        self._validate_tensor_args(train_X, self.train_Y)

        self.train_X = train_X

        self.T = T  # Number of MC samples for inference
        self.input_dim = train_X.shape[-1]
        self.output_dim = train_Y.shape[-1] if train_Y.ndim > 1 else 1

        # Generate hidden layer sizes
        self.hidden_dims = self._generate_hidden_dims(
            self.input_dim, self.output_dim, hidden_layer_multiplier
        )

        # Initialize the Bayesian Neural Network model
        self.model = BNNModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=dropout_rate,
        )

        # Train the model
        self._train(train_X, train_Y, num_epochs, batch_size, learning_rate)

    @property
    def num_outputs(self) -> int:
        return OutcomeTransformMixin.num_outputs.fget(self)

    @property
    def batch_shape(self) -> torch.Size:
        return OutcomeTransformMixin.batch_shape.fget(self)

    def _generate_hidden_dims(
        self, input_dim: int, output_dim: int, multiplier: float
    ) -> List[int]:
        """Generates a list of hidden layer sizes decreasing from input_dim to output_dim."""
        hidden_dims = []
        current_dim = input_dim
        while current_dim > output_dim:
            next_dim = max(output_dim, int(current_dim * multiplier))
            if next_dim == current_dim:
                break
            hidden_dims.append(next_dim)
            current_dim = next_dim
        return hidden_dims

    def _train(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        """Trains the Bayesian Neural Network model."""
        dataset = torch.utils.data.TensorDataset(train_X, train_Y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(num_epochs):
            for batch_X, batch_Y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()

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

        # Prepare input data
        X_flat = X.view(-1, X.size(-1))

        # Perform T stochastic forward passes
        predictions = []
        self.model.train()

        for _ in range(self.T):
            preds = self.model(X_flat)
            predictions.append(preds.cpu())
        predictions = torch.stack(
            predictions, dim=0
        )  # Shape: T x n_samples x n_outputs

        # Compute mean and variance
        mean = predictions.mean(axis=0).squeeze(-1)  # Shape: n_samples x n_outputs
        variance = predictions.var(axis=0).squeeze(-1)  # Shape: n_samples x n_outputs
        # add jitter to places where variance is zero
        jitter = 1.0e-6 * (variance == 0)
        variance += jitter

        posterior = self._build_posterior(
            mean=mean,
            variance=variance,
            X_shape=X.shape,
            observation_noise=observation_noise,
            output_indices=output_indices,
        )

        return posterior
