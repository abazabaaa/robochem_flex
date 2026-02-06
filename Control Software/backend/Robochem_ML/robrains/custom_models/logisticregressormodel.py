"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import numpy as np
import torch
from botorch.models.model import Model
from sklearn.linear_model import LogisticRegression
from torch import Tensor


class LogisticRegressorModel(Model):
    """
    A BoTorch-compatible wrapper for a scikit-learn LogisticRegression.
    Provides `train_model()` and `predict()` methods, and a `posterior()`
    method returning a dummy Posterior that includes the predicted
    probability in its .mean.

    - Assumes a binary classification task where y in {0, 1}.
    - Interprets the probability of y=1 as "probability of failure".
    - If y has only one category (all 0s or all 1s), it skips fitting
      logistic regression and always predicts that same category.
    """

    def __init__(self):
        super().__init__()
        self.logreg = LogisticRegression()
        self._is_trained = False
        # Flags for single-class scenario
        self._is_single_class = False
        self._single_class_label = None  # 0 or 1

    def train_model(self, X: Tensor, y: Tensor) -> None:
        """
        Train (fit) the logistic regression model on data (X, y).

        Args:
            X: A (n x d) or possibly batched Tensor. We'll flatten any batch dims.
            y: A (n, ) Tensor of 0/1 labels (no batch dimension).
        """
        # 1. Flatten out any leading batch dimensions if present
        X_2d = self._flatten_to_2d(X)
        y_1d = y.view(-1).cpu().numpy()  # ensure shape (n,)

        # 2. Check if y contains only one unique value
        unique_classes = np.unique(y_1d)
        if len(unique_classes) == 1:
            # Single-class scenario
            self._is_single_class = True
            self._single_class_label = float(unique_classes[0])  # 0.0 or 1.0
            self._is_trained = True
        else:
            # 3. Fit the sklearn logistic regression as normal
            self.logreg.fit(X_2d, y_1d)
            self._is_trained = True
            self._is_single_class = False
            self._single_class_label = None

    def predict(self, X: Tensor) -> Tensor:
        """
        Predict the probability of failure (class=1) for input X.
        If X has shape (..., d), we flatten the ... dimension into a single batch,
        run predict_proba, then reshape back.

        Returns:
            A Tensor of probabilities with shape matching the leading dimensions of X.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before calling predict().")

        # If it's single-class, return all-0 or all-1 probabilities
        if self._is_single_class:
            # If single_class_label=1 => prob_of_failure=1
            # If single_class_label=0 => prob_of_failure=0
            orig_shape = X.shape[:-1]
            fill_val = self._single_class_label  # 0.0 or 1.0
            return torch.full(
                orig_shape, fill_val, dtype=torch.float32, device=X.device
            )

        # Otherwise, use logistic regression
        orig_shape = X.shape[:-1]  # all but the last dimension
        X_2d = self._flatten_to_2d(X)

        # predict_proba -> shape (num_samples, 2)
        # column 1 is p(class=1)
        probs_np = self.logreg.predict_proba(X_2d)
        p_fail_np = probs_np[:, 1]  # probability of failure

        # Reshape back to match leading dims
        p_fail = torch.from_numpy(p_fail_np).to(X.device).view(*orig_shape)
        return p_fail

    def posterior(self, X: Tensor):
        """
        A BoTorch Model requirement. Returns a 'Posterior' object that
        - has a .mean property with shape (..., 1)
        - can define an .rsample() if needed, but here we just do a dummy version.

        This allows integration with typical BoTorch acquisitions.
        We'll interpret .mean as the log-odds or direct probability.
        Here we provide direct probability p_fail in [0,1].
        """
        p_fail = self.predict(X)

        return _LogisticPosterior(p_fail)

    @staticmethod
    def _flatten_to_2d(X: Tensor) -> np.ndarray:
        """
        Flatten a Tensor of shape (..., d) into (N, d),
        converting to numpy float for scikit-learn.
        """
        # shape: (batch * q * ...), d
        d = X.shape[-1]
        return X.view(-1, d).cpu().detach().numpy()


class _LogisticPosterior:
    """
    A minimal Posterior object that:
      - exposes .mean: shape (..., 1)
      - has a dummy .rsample() method returning the same mean each time
        (no distributional uncertainty).
    """

    def __init__(self, p_fail: Tensor):
        # Make sure to have the last dimension as 1
        self._mean = p_fail
        # if p_fail.dim() == 0:
        #     # single scalar
        #     self._mean = p_fail.unsqueeze(-1)
        # else:
        #     self._mean = p_fail.unsqueeze(-1)

    @property
    def mean(self) -> Tensor:
        return self._mean

    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """
        Returns the same probabilities repeated for each sample in sample_shape.
        This is a dummy implementation (no real sampling).
        """
        return self._mean.expand(*sample_shape, *self._mean.shape)
