"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import torch
from torch import Tensor
from botorch.acquisition import (
    AcquisitionFunction,
    ScalarizedPosteriorTransform,
    MCAcquisitionFunction,
)
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.models.model import Model


class ThresholdedVarianceAcquisition(MCAcquisitionFunction):
    """
    Thresholded-variance: use high-fidelity acquisition when posterior uncertainty
    above tau; otherwise use low-fidelity acquisition.

    Args:
        model: Multi-fidelity GP model.
        tau: variance threshold for high-fidelity sampling.
        ei_low: acquisition targeting low-fidelity (using FixedFeatureAcquisition).
        ei_high: acquisition targeting high-fidelity.
        fidelity_col: index of fidelity in last dim.
    """

    def __init__(
        self,
        model: Model,
        base_acq: MCAcquisitionFunction,
        tau: float,
        s_low: float,
        s_high: float,
        fidelity_col: int = -1,
    ):
        if base_acq.model.num_outputs > 1:
            posterior_transform = ScalarizedPosteriorTransform(
                weights=torch.tensor([1.0] * base_acq.model.num_outputs)
            )
        else:
            posterior_transform = None
        super().__init__(model=base_acq.model, posterior_transform=posterior_transform)
        if model.num_outputs > 1:
            d = model.train_inputs[0][0].shape[-1]
        else:
            d = model.train_inputs[0].shape[-1]
        # fixed-feature acquisitions
        self.ei_low = FixedFeatureAcquisitionFunction(
            acq_function=base_acq,
            d=d,
            columns=[fidelity_col],
            values=[torch.tensor([s_low])],
        )
        self.ei_high = FixedFeatureAcquisitionFunction(
            acq_function=base_acq,
            d=d,
            columns=[fidelity_col],
            values=[torch.tensor([s_high])],
        )
        self.tau = tau
        self.fidelity_col = fidelity_col

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 3:
            raise ValueError("X must be of shape (obs, q, features)")

        obs, q, features = X.shape
        # split x only (without fidelity)
        mask = torch.ones(features, dtype=torch.bool)
        mask[self.fidelity_col] = False
        x = X[..., mask]
        # posterior variance at high fidelity
        X_high = X.clone()
        X_high[..., self.fidelity_col] = self.ei_high.values[0]
        post = self.model.posterior(X_high)
        var_high = post.variance.squeeze(-1)
        if self.model.num_outputs > 1:
            var_high = var_high.mean(dim=-1)
        # choose: if var_high > tau and s==s_high -> use high, else low
        mask_high = var_high > self.tau
        mask_high = mask_high.squeeze(-1)
        # evaluate both acquisitions on x
        a_low = self.ei_low(x)
        a_high = self.ei_high(x)
        # assemble
        return torch.where(mask_high, a_high, a_low)
