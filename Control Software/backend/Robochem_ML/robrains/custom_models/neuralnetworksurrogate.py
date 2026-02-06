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
from torch.utils.data import TensorDataset, DataLoader

from robrains.custom_models.custom_models import (
    OutcomeTransformMixin,
    PosteriorMakingMixin,
)


class SimpleNeuralNetwork(nn.Module):
    """
    Simple feedforward neural network with configurable hidden layers.

    **Args:**
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output targets.
        hidden_dims (List[int]): List of integers specifying the sizes of the hidden layers.

    **Methods:**
        forward(self, x):
            Performs a forward pass through the network.

    **Notes:**
        - Uses ReLU activation for hidden layers.
        - Uses Sigmoid activation for the output layer (modify if outputs are unbounded).
        - Layers are created in sequence from `input_dim` through `hidden_dims` to `output_dim`.

    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super(SimpleNeuralNetwork, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())  # Sigmoid activation on outputs
        self.model = nn.Sequential(*layers)
        self.model = self.model.double()

    def forward(self, x):
        """
        Performs a forward pass through the neural network.

        **Args:**
            x (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        **Returns:**
            Tensor: Output tensor of shape `(batch_size, output_dim)`.

        """
        return self.model(x)


class NeuralNetworkSurrogate(Model, OutcomeTransformMixin, PosteriorMakingMixin):
    """
    Neural Network surrogate model for Bayesian Optimization using BoTorch.

    This model trains an ensemble of feedforward neural networks to approximate the
    objective function. The ensemble approach allows estimation of predictive uncertainty
    by observing the variance across the ensemble's predictions.

    **Attributes:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        n_ensemble (int): Number of neural networks in the ensemble.
        input_dim (int): Dimensionality of the input features.
        output_dim (int): Dimensionality of the output targets.
        models (nn.ModuleList): List of trained neural network models.

    **Args:**
        train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
        train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
        n_ensemble (int, optional): Number of neural networks in the ensemble. Default is `5`.
        hidden_layer_multiplier (float, optional): Multiplier to decrease the size of hidden layers. Default is `0.75`.
        num_epochs (int, optional): Number of training epochs. Default is `100`.
        batch_size (int, optional): Batch size for training. Default is `32`.
        learning_rate (float, optional): Learning rate for the optimizer. Default is `1e-3`.
        **kwargs: Additional keyword arguments.

    **Methods:**
        __init__(self, train_X, train_Y, n_ensemble=5, hidden_layer_multiplier=0.75,
                 num_epochs=100, batch_size=32, learning_rate=1e-3, **kwargs):
            Initializes and trains the ensemble of neural networks.

        posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None, **kwargs):
            Computes the posterior over model outputs at the provided points.

    """

    param_dict = {
        "n_ensemble": "int",
        "hidden_layer_multiplier": "float",
        "num_epochs": "int",
        "batch_size": "int",
        "learning_rate": "float",
    }
    param_defaults = {
        "n_ensemble": 50,
        "hidden_layer_multiplier": 0.75,
        "num_epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
    }

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        n_ensemble: int = 100,
        hidden_layer_multiplier: float = 0.1,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        """
        Initializes and trains an ensemble of neural networks.

        **Args:**
            train_X (Tensor): Training input data of shape `(n_samples, n_features)`.
            train_Y (Tensor): Training target data of shape `(n_samples, n_outputs)`.
            n_ensemble (int, optional): Number of neural networks in the ensemble. Default is `5`.
            hidden_layer_multiplier (float, optional): Multiplier to decrease the size of hidden layers. Default is `0.75`.
            num_epochs (int, optional): Number of training epochs for each neural network. Default is `100`.
            batch_size (int, optional): Batch size for training. Default is `32`.
            learning_rate (float, optional): Learning rate for the optimizer. Default is `1e-3`.
            **kwargs: Additional keyword arguments.

        **Notes:**
            - Generates hidden layer sizes that decrease from `input_dim` to `output_dim` using the `hidden_layer_multiplier`.
            - Creates an ensemble of neural networks (`SimpleNeuralNetwork`) with the same architecture.
            - Trains each neural network in the ensemble independently on the training data.
            - Uses the Adam optimizer and Mean Squared Error (MSE) loss function for training.

        """
        super().__init__()
        OutcomeTransformMixin.__init__(
            self, train_Y, train_X, outcome_transform=outcome_transform
        )
        PosteriorMakingMixin.__init__(self, dtype=train_Y.dtype, device=train_Y.device)
        self._validate_tensor_args(train_X, self.train_Y)

        self.train_X = train_X
        self.n_ensemble = n_ensemble
        self.input_dim = train_X.shape[-1]
        self.output_dim = train_Y.shape[-1] if train_Y.ndim > 1 else 1

        # Define the hidden layer sizes decreasing from input_dim to output_dim
        hidden_dims = self._generate_hidden_dims(
            self.input_dim, self.output_dim, hidden_layer_multiplier
        )

        # Create an ensemble of neural networks
        self.models = nn.ModuleList(
            [
                SimpleNeuralNetwork(self.input_dim, self.output_dim, hidden_dims)
                for _ in range(n_ensemble)
            ]
        )
        self.hidden_dims = hidden_dims
        # Training the ensemble
        self._train_ensemble(train_X, train_Y, num_epochs, batch_size, learning_rate)

    @property
    def num_outputs(self) -> int:
        return OutcomeTransformMixin.num_outputs.fget(self)

    @property
    def batch_shape(self) -> torch.Size:
        return OutcomeTransformMixin.batch_shape.fget(self)

    def _generate_hidden_dims(
        self, input_dim: int, output_dim: int, multiplier: float
    ) -> List[int]:
        """Generates a list of hidden layer sizes decreasing from input_dim to output_dim.

        :param input_dim: (int) Dimension of input features.
        :param output_dim: (int) Dimension of output.
        :param multiplier: (float) Multiplier to decrease the size of hidden layers.
        :return: List of hidden layer sizes.
        """
        hidden_dims = []
        current_dim = input_dim
        while current_dim > output_dim:
            next_dim = max(output_dim, int(current_dim * multiplier))
            if next_dim == current_dim:
                break
            hidden_dims.append(next_dim)
            current_dim = next_dim
        return hidden_dims

    def _train_ensemble(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
    ):
        """Trains the ensemble of neural networks.

        :param train_X: Training inputs.
        :param train_Y: Training outputs.
        :param num_epochs: Number of epochs.
        :param batch_size: Batch size.
        :param learning_rate: Learning rate.
        """
        dataset = TensorDataset(train_X, train_Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for model in self.models:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            for epoch in range(num_epochs):
                for batch_X, batch_Y in loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
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

        **Notes:**
            - Flattens batched input `X` for prediction with the neural networks.
            - Collects predictions from all neural networks in the ensemble.
            - Computes the predictive mean and variance across the ensemble's predictions.
            - Assumes independence between outputs when constructing the covariance matrix.
            - Adds observation noise to the covariance matrix if specified.
            - Constructs a Multivariate Normal distribution with the predictive mean and covariance.

        """
        assert isinstance(
            X, Tensor
        ), f"Tensor X must be a tensor, actual type {type(X)}"

        # Move X to CPU for consistency
        X_cpu = X.cpu()
        original_shape = X_cpu.shape  # Exclude feature dimension

        # Flatten X for prediction
        X_flat = X_cpu.view(-1, X_cpu.size(-1))

        # Collect predictions from all models in the ensemble
        predictions = []
        for model in self.models:
            model.eval()

            preds = model(X_flat)
            predictions.append(preds.cpu())
        predictions = torch.stack(
            predictions, dim=0
        )  # Shape: n_ensemble x n_samples x n_outputs

        # Compute mean and variance across ensemble predictions
        mean = predictions.mean(axis=0).squeeze(-1)  # Shape: n_samples x n_outputs
        variance = predictions.var(axis=0).squeeze(-1)  # Shape: n_samples x n_outputs

        posterior = self._build_posterior(
            mean=mean,
            variance=variance,
            X_shape=original_shape,
            observation_noise=observation_noise,
            output_indices=output_indices,
        )

        return posterior

    def __repr__(self):
        """Representation of model parameters"""
        return (
            f"NeuralNetworkSurrogate(n_ensemble={self.n_ensemble}, input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dims={self.hidden_dims})"
        )
