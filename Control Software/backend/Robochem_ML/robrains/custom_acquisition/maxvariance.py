"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Optional

import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.models.model import Model


class MaxVariance(MCAcquisitionFunction):
    """
    A custom acquisition function that seeks to select points maximizing
    posterior variance. It is compatible with both single-objective and
    multi-objective models. If the model outputs multiple objectives, the
    variance is summed over all objectives. For a batch of q>1 points, the
    variance is also summed across those q points.

    The resulting acquisition value is a single scalar per batch,
    reflecting the total variance the model has over all points and objectives.
    """

    def __init__(
        self,
        model: Model,
        num_samples: int = 1000,
        sampler: Optional[torch.distributions.distribution.Distribution] = None,
        posterior_transform: Optional = None,
    ) -> None:
        """
        Args:
            model: A fitted BoTorch model (single- or multi-output).
            num_samples: Number of Monte Carlo samples to draw from the posterior
                         to estimate the variance.
            sampler: An optional sampler instance. If not provided, default sampling
                     via `posterior.rsample(...)` is used.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.num_samples = num_samples
        self.sampler = sampler  # If you need a custom sampler (Sobol, etc.)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the Max Variance acquisition on the candidate set X.

        Args:
            X: A `batch_size x q x d`-dim Tensor, where:
               - `batch_size` is the number of t-batches (for parallel evaluation),
               - `q` is the number of points chosen jointly (batch size for the design),
               - `d` is the dimensionality of each point.

        Returns:
            A `batch_size`-dim Tensor of acquisition values, where each element
            corresponds to the total variance of the model's posterior for that
            batch of q points, summed across all output dimensions (if multi-objective).
        """
        posterior = self.model.posterior(X)
        # samples shape: [num_samples, batch_size, q, m],
        # where m is the number of output dimensions (1 for single-objective)
        if self.sampler is not None:
            samples = self.sampler(posterior)
        else:
            samples = posterior.rsample(torch.Size([self.num_samples]))

        # Variance across the MC sample dimension -> shape: [batch_size, q, m]
        variance_per_output = samples.var(dim=0)

        # Sum across the q dimension (batch of design points)
        # and the m dimension (output objectives). Result shape: [batch_size].
        total_variance = variance_per_output.sum(dim=(-1, -2))

        return total_variance
