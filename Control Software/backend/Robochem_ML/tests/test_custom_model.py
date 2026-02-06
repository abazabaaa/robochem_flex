"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from botorch.posteriors import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
import torch
import numpy as np
from unittest import TestCase
import unittest
from robrains.custom_models import (
    RandomForestSurrogate,
    NeuralNetworkSurrogate,
    BayesianNeuralNetworkSurrogate,
    SVRSurrogate,
    BayesianLinearRegressionSurrogate,
    MultiOutputSurrogate,
    PosteriorMakingMixin,
)
from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf, optimize_acqf_discrete
from scipy.optimize import minimize
from botorch.models import SingleTaskGP
from unittest import skip
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.models import ModelListGP
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from filelock import FileLock


def test_function_single(X: Tensor, noise_level: float = 0.1) -> Tensor:
    """
    Function to test the model, this is a relatively complex mixed function with both categoricals and continuous variables.

    :param X: A 10-long tensor with 6 continuous variables and 2 vector-embedded variables (both 2-long).
    :param noise_level: The standard deviation of the Gaussian noise added to the output.
    :return: A single value tensor with the output of the function.
    """
    # Split the input into components
    continuous = X[:6]
    embed1 = X[6:8]
    embed2 = X[8:10]

    # Compute the primary sinusoidal component with multiple maxima
    continuous_part = (
        torch.sin(continuous[0] * 3.14)
        + torch.cos(continuous[1] * 2.71)
        + torch.sin(continuous[2] * 1.57)
        + continuous[3] ** 2
        - continuous[4] * continuous[5]
    )

    # Process the embedded vectors to add categorical decision
    embed1_decision = embed1.argmax()  # Get the index of the larger value
    embed2_decision = embed2.argmax()

    # Create an interaction term based on the categorical decisions
    interaction = (embed1_decision + 1) * (embed2_decision + 2)

    # Combine continuous and categorical parts into a final output
    result = continuous_part + interaction

    # Add Gaussian noise
    noise = torch.randn(1).item() * noise_level
    return result + noise


def test_function_multi(X: Tensor, noise_level: float = 0.1) -> Tensor:
    """
    Function to test the model, this is a relatively complex mixed function with both categoricals and continuous variables.

    :param X: A 10-long tensor with 6 continuous variables and 2 vector-embedded variables (both 2-long).
    :param noise_level: The standard deviation of the Gaussian noise added to the outputs.
    :return: A tensor with 4 outputs representing multiple responses.
    """
    # Split the input into components
    continuous = X[:6]
    embed1 = X[6:8]
    embed2 = X[8:10]

    # Compute the primary sinusoidal components with multiple maxima
    continuous_part1 = torch.sin(continuous[0] * 3.14) + continuous[3] ** 2
    continuous_part2 = torch.cos(continuous[1] * 2.71) - continuous[4] * continuous[5]
    continuous_part3 = continuous[2] ** 3 - torch.sin(continuous[5])
    continuous_part4 = torch.exp(-continuous[3]) + torch.sin(
        continuous[1] * continuous[4]
    )

    # Process the embedded vectors to add categorical decision
    embed1_decision = embed1.argmax()  # Get the index of the larger value
    embed2_decision = embed2.argmax()

    # Create interaction terms based on categorical decisions
    interaction1 = (embed1_decision + 1) * (embed2_decision + 2)
    interaction2 = (embed1_decision + embed2_decision) ** 2

    # Combine continuous and categorical parts into multiple outputs
    output1 = continuous_part1 + interaction1
    output2 = continuous_part2 + interaction2
    output3 = continuous_part3 - interaction1
    output4 = continuous_part4 + interaction2

    # Combine all outputs and add Gaussian noise
    results = torch.tensor([output1 + output2, output3 + output4])
    noise = torch.randn(2) * noise_level
    return results + noise


class TestSurrogateModels(unittest.TestCase):
    def setUp(self):
        """Set up common test data for all surrogate models."""
        torch.manual_seed(0)
        np.random.seed(0)

        # Generate synthetic training data
        self.train_X = torch.randn(
            100, 5, dtype=torch.float64
        )  # 100 samples, 5 features
        self.train_Y = torch.randn(100, 1, dtype=torch.float64)  # 100 samples, 1 output

        # Generate synthetic test data
        self.test_X = torch.randn(
            10, 5, dtype=torch.float64
        )  # 10 test samples, 5 features

        # List of surrogate models to test
        self.models = [
            SingleTaskGP,
            RandomForestSurrogate,
            NeuralNetworkSurrogate,
            BayesianNeuralNetworkSurrogate,
            SVRSurrogate,
            BayesianLinearRegressionSurrogate,
        ]

    @skip("Skip this test")
    def test_model_initialization(self):
        """Test that models can be initialized with the same training data."""
        for ModelClass in self.models:
            if ModelClass == SingleTaskGP:
                continue
            with self.subTest(Model=ModelClass.__name__):
                try:
                    model = ModelClass(train_X=self.train_X, train_Y=self.train_Y)
                    print(f"Model; {model} initialised")
                except Exception as e:
                    self.fail(f"Initialization failed for {ModelClass.__name__}: {e}")

    @skip("Skip this test")
    def test_posterior_computation(self):
        """Test that models can compute posterior given test data."""
        for ModelClass in self.models:
            if ModelClass == SingleTaskGP:
                pass
            elif ModelClass == RandomForestSurrogate:
                # continue
                pass
            else:
                continue
            with self.subTest(Model=ModelClass.__name__):
                print(f"Testing {ModelClass.__name__}")
                n, q, v, t = 10, [1, 3], 5, [1, 2]
                for q_, t_ in product(q, t):
                    print(f"Testing with q={q_} and t={t_}")
                    X = torch.randn((n, v), dtype=torch.float64)
                    Y = torch.randn((n, t_), dtype=torch.float64)

                    if q_ == 0:
                        X_test = torch.randn((n, v), dtype=torch.float64)
                    else:
                        X_test = torch.randn((n, q_, v), dtype=torch.float64)

                    model = ModelClass(train_X=X, train_Y=Y)
                    print(f"Model batch shape: {model.batch_shape}")
                    try:
                        posterior = model.posterior(X_test)
                        # self.assertIsInstance(
                        #     posterior,
                        #     Posterior,
                        #     f"Posterior is not a GPyTorchPosterior for {ModelClass.__name__}",
                        # )
                        # Check that posterior mean and covariance have correct shapes
                        mean = posterior.mean
                        covariance = posterior.mvn.covariance_matrix
                        mvn_mean = posterior.mvn.mean
                        print("model batch shape: ", model.batch_shape)
                        print(posterior.mvn)
                        print("with input of shape: ", X_test.shape)
                        print("mean shape: ", mean.shape)
                        print("mvn mean shape: ", mvn_mean.shape)
                        print("covariance shape: ", covariance.shape)

                        # expected_shape = self.test_X.shape[:-1] + (
                        #     self.train_Y.shape[-1]
                        # )
                        # self.assertEqual(
                        #     mean.shape,
                        #     expected_shape,
                        #     f"Posterior mean shape mismatch for {ModelClass.__name__}",
                        # )
                        # expected_cov_shape = self.test_X.shape[:-1] + (
                        #     self.train_Y.shape[-1],
                        #     self.train_Y.shape[-1],
                        # )
                        # self.assertEqual(
                        #     covariance.shape,
                        #     expected_cov_shape,
                        #     f"Posterior covariance shape mismatch for {ModelClass.__name__}",
                        # )
                    except Exception as e:
                        self.fail(
                            f"Posterior computation failed for {ModelClass.__name__}: {e}"
                        )

    @skip("Skip this test")
    def test_posterior_properties(self):
        """Test that the posterior has the expected properties."""
        for ModelClass in self.models:
            with self.subTest(Model=ModelClass.__name__):
                model = ModelClass(train_X=self.train_X, train_Y=self.train_Y)
                posterior = model.posterior(self.test_X)
                mean = posterior.mean
                covariance = posterior.mvn.covariance_matrix

                # Check that mean and covariance are tensors
                self.assertIsInstance(
                    mean,
                    Tensor,
                    f"Posterior mean is not a Tensor for {ModelClass.__name__}",
                )
                self.assertIsInstance(
                    covariance,
                    Tensor,
                    f"Posterior covariance is not a Tensor for {ModelClass.__name__}",
                )

                # Check that covariance matrix is positive semi-definite
                eigvals = torch.linalg.eigvalsh(covariance.cpu())
                self.assertTrue(
                    torch.all(eigvals >= -1e-6),
                    f"Covariance matrix is not positive semi-definite for {ModelClass.__name__}",
                )

    @skip("Skip this test")
    def test_calculate_optimise_acqf(self):
        """let's see if we can optimise a simple acqf function"""
        for ModelClass in self.models:

            if ModelClass == RandomForestSurrogate:
                options = {"with_grad": False}
                # continue
            elif ModelClass == NeuralNetworkSurrogate:
                options = {"with_grad": True}
                # continue
            elif ModelClass == BayesianNeuralNetworkSurrogate:
                options = {"with_grad": True}
                # continue
            elif ModelClass == SVRSurrogate:
                options = {"with_grad": False}
                # continue
            elif ModelClass == BayesianLinearRegressionSurrogate:
                options = {"with_grad": True}
                # continue
            elif ModelClass == SingleTaskGP:
                # continue
                options = {"with_grad": True}

            with self.subTest(Model=ModelClass.__name__):
                # train_Y = torch.randn(100,2, dtype=torch.float64)
                model = ModelClass(train_X=self.train_X, train_Y=self.train_Y)

                try:
                    acqf = UpperConfidenceBound(model, beta=0.1)
                    bound = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.float64)

                    next_point, acq_value = optimize_acqf(
                        acq_function=acqf,
                        bounds=bound,
                        q=1,
                        num_restarts=10,
                        raw_samples=10,
                        options=options,
                    )

                    self.assertIsInstance(
                        next_point,
                        Tensor,
                        f"Next point is not a Tensor for {ModelClass.__name__}",
                    )
                    self.assertIsInstance(
                        acq_value,
                        Tensor,
                        f"Acquisition value is not a Tensor for {ModelClass.__name__}",
                    )

                except Exception as e:
                    self.fail(f"MLL computation failed for {ModelClass.__name__}: {e}")

    @skip("Skip this test")
    def test_calculate_optimise_acqf_batched(self):
        """let's see if we can optimise a simple acqf function"""
        for ModelClass in self.models:
            if ModelClass == RandomForestSurrogate:
                # continue
                options = {"with_grad": False}
            elif ModelClass == NeuralNetworkSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == BayesianNeuralNetworkSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == SVRSurrogate:
                # continue
                options = {"with_grad": False}
            elif ModelClass == BayesianLinearRegressionSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == SingleTaskGP:
                continue
                options = {"with_grad": True}

            with self.subTest(Model=ModelClass.__name__):
                model = ModelClass(train_X=self.train_X, train_Y=self.train_Y)

                try:
                    acqf = qUpperConfidenceBound(model, beta=0.1)
                    bound = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.float64)

                    next_point, acq_value = optimize_acqf(
                        acq_function=acqf,
                        bounds=bound,
                        q=2,
                        num_restarts=10,
                        raw_samples=10,
                        options=options,
                    )

                    self.assertIsInstance(
                        next_point,
                        Tensor,
                        f"Next point is not a Tensor for {ModelClass.__name__}",
                    )
                    self.assertIsInstance(
                        acq_value,
                        Tensor,
                        f"Acquisition value is not a Tensor for {ModelClass.__name__}",
                    )

                except Exception as e:
                    self.fail(f"MLL computation failed for {ModelClass.__name__}: {e}")

    @skip("Skip this test")
    def test_multi_output_list(self):
        """Test that models can be trained with multi-output data."""
        train_Y = torch.randn(100, 2, dtype=torch.float64)
        for ModelClass in self.models:
            if ModelClass == RandomForestSurrogate:
                # continue
                options = {"with_grad": False}
            elif ModelClass == NeuralNetworkSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == BayesianNeuralNetworkSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == SVRSurrogate:
                # Svrsurrogate does not support multi-output
                # continue
                options = {"with_grad": False}
            elif ModelClass == BayesianLinearRegressionSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == SingleTaskGP:
                # continue
                options = {"with_grad": True}
            import time

            with self.subTest(Model=ModelClass.__name__):
                print(f"Testing {ModelClass.__name__}")
                start = time.time()
                models = []
                for I in range(train_Y.shape[1]):
                    model = ModelClass(
                        train_X=self.train_X, train_Y=train_Y[:, I : I + 1]
                    )
                    models.append(model)
                model = MultiOutputSurrogate(models)

                try:
                    ref_point = torch.tensor([0.0, 0.0], dtype=torch.float64)
                    partition = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
                    acqf = qLogExpectedHypervolumeImprovement(
                        model, ref_point=ref_point, partitioning=partition
                    )
                    bound = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.float64)

                    next_point, acq_value = optimize_acqf(
                        acq_function=acqf,
                        bounds=bound,
                        q=2,
                        num_restarts=10,
                        raw_samples=10,
                        options=options,
                    )

                    self.assertIsInstance(
                        next_point,
                        Tensor,
                        f"Next point is not a Tensor for {ModelClass.__name__}",
                    )
                    self.assertIsInstance(
                        acq_value,
                        Tensor,
                        f"Acquisition value is not a Tensor for {ModelClass.__name__}",
                    )

                except Exception as e:
                    self.fail(f"MLL computation failed for {ModelClass.__name__}: {e}")
                print(f"Time taken: {time.time()-start}")

    @skip("Skip this test")
    def test_multi_output_single(self):
        """Test that models can be trained with multi-output data."""
        train_Y = torch.randn(100, 2, dtype=torch.float64)
        for ModelClass in self.models:
            if ModelClass == RandomForestSurrogate:
                # continue
                options = {"with_grad": False}
            elif ModelClass == NeuralNetworkSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == BayesianNeuralNetworkSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == SVRSurrogate:
                # Svrsurrogate does not support multi-output
                continue
                options = {"with_grad": False}
            elif ModelClass == BayesianLinearRegressionSurrogate:
                # continue
                options = {"with_grad": True}
            elif ModelClass == SingleTaskGP:
                # continue
                options = {"with_grad": True}

            with self.subTest(Model=ModelClass.__name__):

                model = ModelClass(train_X=self.train_X, train_Y=train_Y)

                try:
                    ref_point = torch.tensor([0.0, 0.0], dtype=torch.float64)
                    partition = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
                    acqf = qExpectedHypervolumeImprovement(
                        model, ref_point=ref_point, partitioning=partition
                    )
                    bound = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.float64)

                    next_point, acq_value = optimize_acqf(
                        acq_function=acqf,
                        bounds=bound,
                        q=1,
                        num_restarts=10,
                        raw_samples=10,
                        options=options,
                    )

                    self.assertIsInstance(
                        next_point,
                        Tensor,
                        f"Next point is not a Tensor for {ModelClass.__name__}",
                    )
                    self.assertIsInstance(
                        acq_value,
                        Tensor,
                        f"Acquisition value is not a Tensor for {ModelClass.__name__}",
                    )

                except Exception as e:
                    self.fail(f"MLL computation failed for {ModelClass.__name__}: {e}")

    @skip("Skip this test")
    def test_single_taks_gp(self):
        """Test what shapes are reuqired for the single task GP for optimisation
        of t = 1 and 2 and q = 1 and 3"""

        n = 10
        v = 5
        q = [1, 3]
        t = [1, 2]

        for q_, t_ in product(q, t):
            X = torch.randn(n, v, dtype=torch.float64)
            Y = torch.randn(n, t_, dtype=torch.float64)
            model = SingleTaskGP(train_X=X, train_Y=Y)

            acqf = UpperConfidenceBound(model, beta=0.1)
            bound = torch.tensor([[0.0] * v, [1.0] * v], dtype=torch.float64)

            next_point, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=bound,
                q=1,
                num_restarts=10,
                raw_samples=10,
                options={"with_grad": True},
            )

    @skip("Skip this test")
    def test_make_posterior(self):
        """Test the posterior mixin for the model, should return a GPY posterior, we need to check tensor shapes"""

        posterior_maker = PosteriorMakingMixin()

        n_values = [10]
        q_values = [1, 2, 3]
        v_values = [5]
        t_values = [1, 2]

        # make all the combinations
        prod = product(n_values, q_values, v_values, t_values)

        for p in prod:
            # rules of the game, we make a mean and variance tensors and an X tensor of zeros in the right sh
            n, q, v, t = p
            print(f"Testing: n={n}, q={q}, v={v}, t={t}")
            if t == 1:
                print(
                    f"expected shapes: p.mean ({n,q,t}), mvn.mean ({n,q}), mvn.covar ({n,q*t,q*t})"
                )
            else:
                print(
                    f"expected shapes: p.mean ({n,q,t}), mvn.mean ({n,q,t}), mvn.covar ({n,q*t,q*t})"
                )
            X = torch.ones(n, q, v)
            mean = torch.ones(n * q, t)
            variance = torch.ones(n * q, t)
            print(f"X shape: {X.shape}")
            print(f"mean shape: {mean.shape}")
            print(f"variance shape: {variance.shape}")
            posterior_maker._num_outputs = t
            posterior_maker._untransform_posterior = lambda x: x
            # make the posterior
            posterior = posterior_maker._build_posterior(
                mean=mean, variance=variance, X_shape=X.shape
            )
            print(f"posterior mean shape: {posterior.mean.shape}")
            print(f"posterior variance shape: {posterior.variance.shape}")
            print(f"posterior mvn mean shape: {posterior.mvn.mean.shape}")
            print(
                f"posterior mvn covariance shape: {posterior.mvn.covariance_matrix.shape}"
            )
            continue
            self.assertIsInstance(posterior, GPyTorchPosterior)
            # now the posterior should have the right shapes:
            if t == 1:
                self.assertEqual(
                    posterior.mean.shape,
                    (n, q, t),
                    f"Mean shape is {posterior.mean.shape}",
                )
                self.assertEqual(
                    posterior.mvn.mean.shape,
                    (n, q),
                    f"MVN mean shape is {posterior.mvn.mean.shape}",
                )
                self.assertEqual(
                    posterior.mvn.covariance_matrix.shape,
                    (n, q * t, q * t),
                    f"MVN covar shape is {posterior.mvn.covariance_matrix.shape}",
                )
            else:
                self.assertEqual(
                    posterior.mean.shape,
                    (n, q, t),
                    f"Mean shape is {posterior.mean.shape}",
                )
                self.assertEqual(
                    posterior.mvn.mean.shape,
                    (n, q, t),
                    f"MVN mean shape is {posterior.mvn.mean.shape}",
                )
                self.assertEqual(
                    posterior.mvn.covariance_matrix.shape,
                    (n, q * t, q * t),
                    f"MVN covar shape is {posterior.mvn.covariance_matrix.shape}",
                )

    @skip("Skip this test")
    def test_optimise_single(self):
        """Test the optimisation of a single output function with parallel processing."""
        # Start set
        n = 10
        X = torch.randn(n, 10, dtype=torch.float64)
        Y = torch.tensor(
            [test_function_single(x, noise_level=0.05) for x in X], dtype=torch.float64
        ).reshape(-1, 1)

        def optimise_for_model(ModelClass):
            """Optimization routine for a specific model."""
            if ModelClass == RandomForestSurrogate:
                options = {"with_grad": False}
            elif ModelClass == NeuralNetworkSurrogate:
                options = {"with_grad": True}
            elif ModelClass == BayesianNeuralNetworkSurrogate:
                options = {"with_grad": True}
            elif ModelClass == SVRSurrogate:
                options = {"with_grad": False}
            elif ModelClass == BayesianLinearRegressionSurrogate:
                options = {"with_grad": True}
            else:
                options = {}

            Y_values = []

            try:
                local_X = X.clone()  # Avoid sharing mutable tensors between threads
                local_Y = Y.clone()
                print("Optimising for ", ModelClass.__name__)
                # print(str(ModelClass))
                for i in range(500):
                    # Create and fit the model
                    model = ModelClass(train_X=local_X, train_Y=local_Y)
                    print(model)
                    # Define acquisition function
                    acqf = UpperConfidenceBound(model, beta=0.01)
                    bounds = torch.tensor([[0.0] * 10, [1.0] * 10], dtype=torch.float64)

                    # Optimize the acquisition function
                    next_point, acq_value = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                        options=options,
                    )

                    # Evaluate the function at the new point
                    next_Y = torch.tensor(
                        [test_function_single(next_point[0], noise_level=0.01)],
                        dtype=torch.float64,
                    ).reshape(-1, 1)

                    # Concatenate the new data
                    local_X = torch.cat((local_X, next_point), dim=0)
                    local_Y = torch.cat((local_Y, next_Y), dim=0)

                    # Record Y for plotting
                    Y_values.append(next_Y.item())

                    # Convergence check
                    # if (
                    #     local_Y[-3:].std() < 0.1
                    # ):  # Check if the standard deviation of the last 3 values is small
                    #     print(f"Convergence achieved for {ModelClass.__name__}.")
                    #     break
                print(f"Optimisation for {ModelClass.__name__} completed.")
            except Exception as e:
                return ModelClass.__name__, Y_values, str(e)

            return ModelClass.__name__, Y_values, None

        # Use ThreadPoolExecutor to parallelize the optimization for each model
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(optimise_for_model, ModelClass)
                for ModelClass in self.models
                if not ModelClass
                in [
                    SVRSurrogate,
                    BayesianLinearRegressionSurrogate,
                    BayesianNeuralNetworkSurrogate,
                    SingleTaskGP,
                ]
            ]
            for future in futures:
                results.append(future.result())

        # Plot the results
        for model_name, Y_values, error in results:
            if error:
                print(f"Error in {model_name}: {error}")
            else:
                plt.scatter(range(len(Y_values)), Y_values, label=model_name)
                # Define the window size for the moving average
                window_size = 3  # You can adjust this value as needed

                # Calculate the moving average using numpy's convolve function
                moving_avg = np.convolve(
                    Y_values, np.ones(window_size) / window_size, mode="valid"
                )

                # Since the moving average array is shorter, adjust the x-axis accordingly
                moving_avg_x = np.arange(window_size - 1, len(Y_values))

                # Plot the moving average
                plt.plot(moving_avg, label=f"Moving Average_{model_name}")

        plt.xlabel("Iteration")

        plt.legend()
        plt.show()

    @skip("Skip this test")
    def test_optimise_multi(self):
        """Test the optimisation of a single output function with parallel processing."""
        # Start set
        n = 10
        X = torch.randn(n, 10, dtype=torch.float64)
        y = [test_function_multi(x, noise_level=0.05) for x in X]
        Y = torch.cat(y, dim=0).reshape(-1, 2)

        def optimise_for_model(ModelClass):
            """Optimization routine for a specific model."""
            if ModelClass == RandomForestSurrogate:
                options = {"with_grad": False}
            elif ModelClass == NeuralNetworkSurrogate:
                options = {"with_grad": True}
            elif ModelClass == BayesianNeuralNetworkSurrogate:
                options = {"with_grad": True}
            elif ModelClass == SVRSurrogate:
                options = {"with_grad": False}
            elif ModelClass == BayesianLinearRegressionSurrogate:
                options = {"with_grad": True}
            else:
                options = {}

            Y_values = []

            try:
                local_X = X.clone()  # Avoid sharing mutable tensors between threads
                local_Y = Y.clone()
                print("Optimising for ", ModelClass.__name__)
                for i in range(500):
                    models = []
                    for I in range(local_Y.shape[1]):
                        model = ModelClass(
                            train_X=local_X, train_Y=local_Y[:, I : I + 1]
                        )
                        models.append(model)
                    model = MultiOutputSurrogate(models)

                    ref_point = torch.tensor([0.0, 0.0], dtype=torch.float64)
                    partition = NondominatedPartitioning(ref_point=ref_point, Y=local_Y)
                    acqf = qLogExpectedHypervolumeImprovement(
                        model, ref_point=ref_point, partitioning=partition
                    )
                    bounds = torch.tensor([[0.0] * 10, [1.0] * 10], dtype=torch.float64)

                    # Optimize the acquisition function
                    next_point, acq_value = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                        options=options,
                    )

                    # Evaluate the function at the new point
                    next_Y = test_function_multi(
                        next_point[0], noise_level=0.01
                    ).reshape(-1, 2)

                    # Concatenate the new data
                    local_X = torch.cat((local_X, next_point), dim=0)
                    local_Y = torch.cat((local_Y, next_Y), dim=0)

                    # Record Y for plotting
                    Y_values.append(next_Y)

                    # Convergence check
                    if (
                        local_Y[-3:].std() < 0.05
                    ):  # Check if the standard deviation of the last 3 values is small
                        print(f"Convergence achieved for {ModelClass.__name__}.")
                        break
                print(f"Optimisation for {ModelClass.__name__} completed.")
            except Exception as e:
                Y_values = torch.cat(Y_values, dim=0)
                return ModelClass.__name__, Y_values, str(e)
            Y_values = torch.cat(Y_values, dim=0)
            return ModelClass.__name__, Y_values, None

        # Use ThreadPoolExecutor to parallelize the optimization for each model
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(optimise_for_model, ModelClass)
                for ModelClass in self.models
            ]
            for future in futures:
                results.append(future.result())

        fig, ax = plt.subplots(2, 1)
        # Plot the results
        for model_name, Y_values, error in results:
            if error:
                print(f"Error in {model_name}: {error}")
            else:

                ax[0].scatter(range(len(Y_values)), Y_values[:, 0], label=model_name)
                ax[1].scatter(range(len(Y_values)), Y_values[:, 1], label=model_name)
                # Define the window size for the moving average
                window_size = 3  # You can adjust this value as needed

                # Calculate the moving average using numpy's convolve function
                moving_avg_1 = np.convolve(
                    Y_values[:, 0], np.ones(window_size) / window_size, mode="valid"
                )
                moving_avg_2 = np.convolve(
                    Y_values[:, 1], np.ones(window_size) / window_size, mode="valid"
                )

                # Since the moving average array is shorter, adjust the x-axis accordingly
                moving_avg_x = np.arange(window_size - 1, len(Y_values))

                # Plot the moving average
                ax[0].plot(moving_avg_1, label=f"Moving Average_{model_name}")
                ax[1].plot(moving_avg_2, label=f"Moving Average_{model_name}")

        plt.legend()
        plt.show()

    def test_nets(self):
        """Test the neural network models with file locking for concurrent data writing."""
        # Start set
        n = 10
        X = torch.randn(n, 10, dtype=torch.float64)
        Y = torch.tensor(
            [test_function_single(x, noise_level=0.05) for x in X], dtype=torch.float64
        ).reshape(-1, 1)

        output_file = "optimisation_results.json"
        lock_file = "optimisation_results.lock"

        def optimise_for_model(ModelClass, kwargs={}):
            """Optimization routine for a specific model."""
            if ModelClass == RandomForestSurrogate:
                options = {"with_grad": False}
            elif ModelClass == NeuralNetworkSurrogate:
                options = {"with_grad": True}
            elif ModelClass == BayesianNeuralNetworkSurrogate:
                options = {"with_grad": True}
            elif ModelClass == SVRSurrogate:
                options = {"with_grad": False}
            elif ModelClass == BayesianLinearRegressionSurrogate:
                options = {"with_grad": True}
            else:
                options = {}

            Y_values = []

            try:
                local_X = X.clone()  # Avoid sharing mutable tensors between threads
                local_Y = Y.clone()
                print("Optimising for ", ModelClass.__name__)
                print("with kwargs: ", kwargs)

                for i in range(100):
                    # Create and fit the model
                    model = ModelClass(train_X=local_X, train_Y=local_Y, **kwargs)

                    # Define acquisition function
                    acqf = UpperConfidenceBound(model, beta=0.01)
                    bounds = torch.tensor([[0.0] * 10, [1.0] * 10], dtype=torch.float64)

                    # Optimize the acquisition function
                    next_point, acq_value = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        num_restarts=10,
                        raw_samples=512,
                        options=options,
                    )

                    # Evaluate the function at the new point
                    next_Y = torch.tensor(
                        [test_function_single(next_point[0], noise_level=0.01)],
                        dtype=torch.float64,
                    ).reshape(-1, 1)

                    # Concatenate the new data
                    local_X = torch.cat((local_X, next_point), dim=0)
                    local_Y = torch.cat((local_Y, next_Y), dim=0)

                    # Record Y for plotting
                    Y_values.append(next_Y.item())

                # Write results to file with a lock
                with FileLock(lock_file):
                    if not os.path.exists(output_file):
                        with open(output_file, "w") as f:
                            json.dump([], f)

                    with open(output_file, "r") as f:
                        data = json.load(f)

                    data.append(
                        {
                            "model": ModelClass.__name__,
                            "kwargs": kwargs,
                            "Y_values": Y_values,
                        }
                    )

                    with open(output_file, "w") as f:
                        json.dump(data, f, indent=4)

                print(f"Optimisation for {ModelClass.__name__} completed.")
            except Exception as e:
                return ModelClass.__name__, Y_values, str(e)

            return ModelClass.__name__, Y_values, None

        # Parameters for neural network testing
        kwarg_n_ensemble = [5, 10, 50, 100]
        kwarg_hidden_layer_multiplier = [0.1, 0.5, 0.75]
        kwarg_num_epochs = [50, 100, 200]
        kwarg_batch_size = [16, 32, 64]
        kwarg_learning_rate = [0.01, 0.001, 0.0001]
        kwargs_list_of_dicts = [
            {
                "n_ensemble": n,
                "hidden_layer_multiplier": h,
                "num_epochs": e,
                "batch_size": b,
                "learning_rate": lr,
            }
            for n, h, e, b, lr in product(
                kwarg_n_ensemble,
                kwarg_hidden_layer_multiplier,
                kwarg_num_epochs,
                kwarg_batch_size,
                kwarg_learning_rate,
            )
        ]

        # Execute in parallel
        with ThreadPoolExecutor() as executor:
            for kwargs in kwargs_list_of_dicts:
                executor.submit(optimise_for_model, NeuralNetworkSurrogate, kwargs)

            # results = [future.result() for future in futures

        # # Plot the results
        # for model_name, Y_values, error in results:
        #     if error:
        #         print(f"Error in {model_name}: {error}")
        #     else:
        #         plt.scatter(range(len(Y_values)), Y_values, label=model_name)
        #         # Define the window size for the moving average
        #         window_size = 3  # You can adjust this value as needed
        #
        #         # Calculate the moving average using numpy's convolve function
        #         moving_avg = np.convolve(
        #             Y_values, np.ones(window_size) / window_size, mode="valid"
        #         )
        #
        #         # Since the moving average array is shorter, adjust the x-axis accordingly
        #         moving_avg_x = np.arange(window_size - 1, len(Y_values))
        #
        #         # Plot the moving average
        #         plt.plot(moving_avg, label=f"Moving Average_{model_name}")

        plt.xlabel("Iteration")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
    # x = torch.tensor([0.5, 1.2, 0.8, 1.1, 0.9, 0.3, 0.6, 0.4, 0.7, 0.3], dtype = torch.float64)
    # print(test_function_single(x))
    # print(test_function_multi(x))
    # test_function_multi()
