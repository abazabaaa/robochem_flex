import unittest
from unittest import skip

import torch
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from torch import Tensor
from botorch.models import SingleTaskMultiFidelityGP, ModelListGP
from botorch.acquisition import qLogExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from robrains.custom_acquisition import (
    CostAwareAcquisition,
    LagrangianAcquisition,
    EpsilonGreedyAcquisition,
    ThresholdedVarianceAcquisition,
)
from robrains.cost_functions.gpcostfunction import gp_cost_function


def branin_with_fidelity(X: Tensor) -> Tensor:
    x1 = X[..., 0] * 15 - 5
    x2 = X[..., 1] * 15
    s = X[..., 2]
    a, b, c, r, t = 1.0, 5.1 / (4 * torch.pi**2), 5 / torch.pi, 6.0, 10.0
    y = (
        a * (x2 - b * x1**2 + c * x1 - r) ** 2
        + t * (1 - 1 / (8 * torch.pi)) * torch.cos(x1)
        + t
    )
    bias = (1 - s) * 10
    noise = torch.randn_like(y) * 2 * (1 - s)
    return y + bias + noise


def linear_with_fidelity(X: Tensor) -> Tensor:
    x1, x2, s = X[..., 0], X[..., 1], X[..., 2]
    return (x1 + x2) * s


def cost_fn(s: Tensor) -> Tensor:
    return 1.0 + 9 * s.pow(2)


class TestCostAwareAcquisitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        cls.X_init = cls.sobol.draw(10, dtype=torch.float64)
        cls.Y_init = branin_with_fidelity(cls.X_init).unsqueeze(-1)

        def make_mf_model(X: Tensor, Y: Tensor) -> SingleTaskMultiFidelityGP:
            model = SingleTaskMultiFidelityGP(
                train_X=X, train_Y=Y, data_fidelities=[-1]
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            return model

        cls.model = make_mf_model(cls.X_init, cls.Y_init)

        Y1 = branin_with_fidelity(cls.X_init).unsqueeze(-1)
        Y2 = linear_with_fidelity(cls.X_init).unsqueeze(-1)
        m1 = make_mf_model(cls.X_init, Y1)
        m2 = make_mf_model(cls.X_init, Y2)
        mll1 = ExactMarginalLogLikelihood(m1.likelihood, m1)
        mll2 = ExactMarginalLogLikelihood(m2.likelihood, m2)
        fit_gpytorch_mll(mll1)
        fit_gpytorch_mll(mll2)
        cls.multiobj_model = ModelListGP(m1, m2)
        cls.ref = torch.tensor([30.0, 0.0])
        cls.partitioning = NondominatedPartitioning(
            ref_point=cls.ref, Y=torch.cat([Y1, Y2], dim=-1)
        )
        bounds_upper = [1.0] * cls.X_init.shape[-1]
        bounds_lower = [0.0] * cls.X_init.shape[-1]
        cls.bounds = torch.tensor([bounds_lower, bounds_upper], dtype=torch.float64)

    def test_cost_aware_acquisition_single_objective(self):
        X, Y, model = self.X_init, self.Y_init, self.model
        ei = qUpperConfidenceBound(model=model, beta=0.1)
        ca = CostAwareAcquisition(
            base_acq=ei, fidelity_col=-1, cost_fn=lambda s: torch.randn_like(s)
        )
        Xcand, _ = optimize_acqf(
            acq_function=ca, q=5, bounds=self.bounds, num_restarts=10, raw_samples=512
        )
        s_next = Xcand[0, -1]
        x_next = Xcand[0, :-1]
        self.assertGreaterEqual(s_next.item(), 0.0)
        self.assertLessEqual(s_next.item(), 1.0)
        self.assertEqual(x_next.shape, (2,))

    # @skip("Test not implemented yet")
    def test_lagrangian_acquisition_single_objective(self):
        X, Y, model = self.X_init, self.Y_init, self.model
        ei = qUpperConfidenceBound(model=model, beta=0.1)
        lacq = LagrangianAcquisition(
            base_acq=ei, cost_fn=lambda s: torch.randn_like(s), lambda_cost=2.0
        )
        Xcand, _ = optimize_acqf(
            acq_function=lacq, bounds=self.bounds, q=1, num_restarts=3, raw_samples=5
        )
        s_next = Xcand[0, -1]
        x_next = Xcand[0, :-1]
        self.assertGreaterEqual(s_next.item(), 0.0)
        self.assertLessEqual(s_next.item(), 1.0)
        self.assertEqual(x_next.shape, (2,))

    def test_epsilon_greedy_acquisition_single_objective(self):
        X, Y, model = self.X_init, self.Y_init, self.model
        ei = qUpperConfidenceBound(model=model, beta=0.1)
        eg = EpsilonGreedyAcquisition(
            base_acq=ei, s_low=0.0, s_high=1.0, fidelity_col=-1, epsilon=0.2
        )
        Xcand, _ = optimize_acqf(
            acq_function=eg, bounds=self.bounds, q=1, num_restarts=3, raw_samples=5
        )
        s_next = Xcand[0, -1]
        x_next = Xcand[0, :-1]
        self.assertGreaterEqual(s_next.item(), 0.0)
        self.assertLessEqual(s_next.item(), 1.0)
        self.assertEqual(x_next.shape, (2,))

    def test_thresholded_variance_acquisition_single_objective(self):
        X, Y, model = self.X_init, self.Y_init, self.model
        tau = 1.0
        ei = qUpperConfidenceBound(model=model, beta=0.1)
        tv = ThresholdedVarianceAcquisition(
            model=model,
            base_acq=ei,
            tau=tau,
            s_low=0.0,
            s_high=1.0,
            fidelity_col=-1,
        )
        Xcand, _ = optimize_acqf(
            acq_function=tv, bounds=self.bounds, q=1, num_restarts=3, raw_samples=5
        )
        s_next = Xcand[0, -1]
        x_next = Xcand[0, :-1]
        self.assertGreaterEqual(s_next.item(), 0.0)
        self.assertLessEqual(s_next.item(), 1.0)
        self.assertEqual(x_next.shape, (2,))

    # @skip("Test not implemented yet")
    def test_multiobjective_acquisitions(self):
        mlgp = self.multiobj_model
        bounds = self.bounds
        base_ehvi = qExpectedHypervolumeImprovement(
            model=mlgp, partitioning=self.partitioning, ref_point=self.ref
        )

        wrappers = [
            lambda base: CostAwareAcquisition(
                base, fidelity_col=-1, cost_fn=lambda s: torch.randn_like(s)
            ),
            lambda base: LagrangianAcquisition(
                base, cost_fn=lambda s: torch.randn_like(s), lambda_cost=2.0
            ),
            lambda base: EpsilonGreedyAcquisition(
                base, s_low=0.0, s_high=1.0, fidelity_col=-1, epsilon=0.2
            ),
            lambda base: ThresholdedVarianceAcquisition(
                model=mlgp,
                base_acq=base,  # exploitation = hypervolume‐improvement
                tau=1.0,
                s_low=0.0,
                s_high=1.0,
                fidelity_col=-1,
            ),
        ]

        for wrap in wrappers:
            acq = wrap(base_ehvi)
            # no more set_acq_fns needed!
            Xcand, _ = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            s_next = Xcand[0, -1]
            x_next = Xcand[0, :-1]
            self.assertGreaterEqual(s_next.item(), 0.0)
            self.assertLessEqual(s_next.item(), 1.0)
            self.assertEqual(x_next.shape, (2,))


if __name__ == "__main__":
    unittest.main()
