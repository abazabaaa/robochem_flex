"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Optional, Union, List

import pandas as pd
import torch

from robrains.communication_module.communication_module import baseMLBackend


class ML_Assistant(baseMLBackend):
    def run(self) -> None:
        """
        Mark all runs as finished and trigger the update process.

        This method is not called directly by ML_Assistant;
        the frontend invokes ML_Assistant and then observes `results_df`.
        """
        # Log run invocation
        self.log_mssg(
            "ML_Assistant run invoked: marking all statuses as 'finished'.", level="ok"
        )

        # Set all statuses to finished and perform any update logic
        if hasattr(self, "results_df"):
            self.results_df["status"] = "finished"
            self.log_mssg(
                f"Set {len(self.results_df)} rows to status 'finished'.",
                level="ok",
                indent=1,
            )
        else:
            self.log_mssg("No results_df to update in run().", level="warning")

        # Call update hook
        try:
            self.update()
            self.log_mssg("ML_Assistant update() completed.", level="ok")
        except Exception as e:
            self.log_mssg(f"Error during update() in run(): {e}", level="error")
            raise

    def com_prime(self) -> None:
        """
        Initialize communication layer for ML_Assistant.

        In the ML_Assistant version, communication is handled by the frontend,
        so no additional setup is required.
        """
        self._com_primed = True
        self.log_mssg("Communication primed for ML_Assistant.", level="ok")

    def in_data(self, results_df: pd.DataFrame) -> None:
        """
        Load experiment results from the frontend into the internal DataFrame.

        :param results_df: DataFrame of experimental results from the frontend.
        """
        self.log_mssg("Loading input data into ML_Assistant.")
        self.run_index = 0
        try:
            self._from_df(df=results_df)
            self.log_mssg("Input DataFrame processed into internal state.", level="ok")
        except Exception as e:
            self.log_mssg(f"Error processing input DataFrame: {e}", level="error")
            raise

    def out_data(
        self,
        tensor: List[torch.Tensor],
        predicted_y: Optional[Union[torch.Tensor, tuple]] = None,
    ) -> pd.DataFrame:
        """
        Convert output tensors into DataFrame rows and return the updated results.

        :param tensor: 2D tensor of input features for each new point (n_points x n_features).
        :param predicted_y: Optional tuple of (means, variances) tensors for predictions.
        :return: Copy of the updated results DataFrame.
        """
        self.log_mssg(f"Exporting {len(tensor)} data points to results_df.")
        try:
            for i in range(len(tensor)):
                point = tensor[i]
                preds = None
                if predicted_y is not None:
                    means, vars_ = predicted_y
                    preds = (means[i], vars_[i])
                self.log_mssg(f"Adding tensor index {i} to results_df.", indent=1)
                self.log_mssg(f"Point: {point}", indent=2)
                self._to_df(point, run_idx=None, predicted_y=preds)
                self.log_mssg(f"Added row for tensor index {i}.", level="ok", indent=1)
            result = self.results_df.copy()
            self.log_mssg("Output DataFrame generation completed.", level="ok")
            return result
        except Exception as e:
            self.log_mssg(f"Error exporting to DataFrame: {e}", level="error")
            raise
