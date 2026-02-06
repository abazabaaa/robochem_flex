"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Union, Optional

from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import Posterior, PosteriorList
from torch import Tensor


class MultiOutputSurrogate:
    def __init__(self, models):
        self.models = models  # List of models, one per output

    def posterior(
        self,
        X: Tensor,
        observation_noise: Union[bool, Tensor] = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> Posterior:
        # Collect posteriors from each model
        posteriors = []
        for model in self.models:
            posterior = model.posterior(
                X=X,
                observation_noise=observation_noise,
                posterior_transform=None,  # Apply transform after combining
                **kwargs,
            )
            posteriors.append(posterior)

        # Combine posteriors
        combined_posteriors = PosteriorList(*posteriors)

        return combined_posteriors
