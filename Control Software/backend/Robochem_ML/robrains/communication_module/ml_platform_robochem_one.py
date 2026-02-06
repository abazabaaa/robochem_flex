"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Literal, Union, Iterable

import torch
from typing import List, Any, Optional
from robrains.communication_module.ml_assistant import ML_Assistant


class ML_Platform_Robochem_One(ML_Assistant):
    """
    Single-threaded ML platform for the RoboChem_One project.

    Wraps ML_Assistant logic for a simpler, single-thread loop as preferred by Mr. Oliver.
    """

    def run(self) -> None:
        """
        Execute a single-threaded optimization step:
        1. If no runs have ever been submitted or written, perform first_run and mark status 'written'.
        2. Otherwise, update the model on completed data only.
        """
        self.log_mssg("RoboChem_One run invoked.", level="ok")
        if not self._initial_check():
            self.log_mssg(
                "No existing data; performing first_run().", level="ok", indent=1
            )
            self.first_run()
            self.results_df["status"] = "written"
            self.log_mssg("Marked all initial runs as 'written'.", level="ok", indent=1)
        else:
            self.log_mssg(
                "Existing data detected; calling update().", level="ok", indent=1
            )
            self.update()
        self.log_mssg("RoboChem_One run completed.", level="ok")

    def com_prime(self, **kwargs) -> None:
        """
        Initialize communication settings for RoboChem_One.

        Extracts required and optional parameter lists without threading or queues.
        """
        self.constants: List[Any] = kwargs.get("constants", [])
        self.robochem_params_required: List[Any] = kwargs.get(
            "robochem_params_required", []
        )
        self.robochem_params_optional: List[Any] = kwargs.get(
            "robochem_params_optional", []
        )
        self._com_primed = True
        self.log_mssg("Communication primed for RoboChem_One.", level="ok")

    def _get_robochem_param(self, name: str) -> Optional["ExperimentalParameter"]:
        """
        Retrieve an ExperimentalParameter by name from required or optional lists.

        :param name: Name of the parameter to find.
        :return: Matching ExperimentalParameter or None if not found.
        """
        for param in self.robochem_params_required + self.robochem_params_optional:
            if param.name == name:
                self.log_mssg(
                    f"Found RoboChem parameter '{name}'.", level="ok", indent=1
                )
                return param
        self.log_mssg(
            f"RoboChem parameter '{name}' not found.", level="warning", indent=1
        )
        return None

    def _initial_check(self) -> bool:
        """
        Determine if there are no pending 'submitted' or 'written' runs and some history exists.
        """
        exists = hasattr(self, "results_df") and self.run_index > 0
        pending = exists and (
            any(self.results_df["status"] == "submitted")
            or any(self.results_df["status"] == "written")
        )
        result = exists and not pending
        self.log_mssg(f"RoboChem initial_check -> {result}.", level="ok", indent=1)
        return result

    def in_data(
        self,
        run_index: int,
        x: torch.Tensor,
        y: torch.Tensor,
        y_var: torch.Tensor,
        vial_idx: str,
        save_file_name: Optional[str] = None,
    ) -> None:
        """
        Process incoming machine results: update DataFrame and mark status.
        """
        self.log_mssg(f"RoboChem in_data for run_index={run_index}.", level="ok")
        if x is None:
            self.set_status(run_index, "failed")
            self.log_mssg(
                f"Run {run_index} failed due to missing features.",
                level="warning",
                indent=1,
            )
            return
        self._to_df(x, run_idx=run_index)
        self.set_status(run_index, "finished")
        self.set_vial(run_index, vial_idx)
        self._add_y_to_df(run_index, y, y_var)
        if save_file_name:
            self._add_savename_to_df(run_index, save_file_name)
        self.log_mssg(
            f"RoboChem in_data processing complete for run_index={run_index}.",
            level="ok",
        )

    def out_data(
        self,
        next_points: torch.Tensor,
        predicted_y: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Add new design points to the submission DataFrame, marking them 'written'.
        """
        self.log_mssg("RoboChem out_data invoked.", level="ok")
        if (
            isinstance(next_points, str)
            and next_points == "failures"
            and self.parameters.get("Resubmission of Failed N", 0)
        ):
            self.resubmit_runs(statuses=["failed", "retry"])
            return
        if isinstance(next_points, str) and next_points== 'stop':
            self.log_mssg("Gotten the stop signal, stopping the experimentation.", level = "ok")
            self.set_status(self.run_index, "stop")
            return

        for i, point in enumerate(next_points):
            self.run_index = (self.run_index or 0) + 1
            preds = None
            if predicted_y is not None:
                preds = (predicted_y[0][i], predicted_y[-1][i])
            self.log_mssg(
                f"Adding design point: {point}  to DataFrame.", level="ok", indent=1
            )
            self._to_df(point, run_idx=self.run_index, predicted_y=preds)
            self.set_status(self.run_index, "written")
            self.log_mssg(
                f"Added design point for run_index={self.run_index}.",
                level="ok",
                indent=1,
            )

    def resubmit_runs(self, statuses: Union[str, Iterable[str]]) -> None:
        """
        Resubmit runs whose status is in `statuses` until policy limits are reached.

        - `statuses` can be a single status string or an iterable of statuses (e.g. ['failed','retry','check']).

        Marks runs as 'Dead' if attempts exceeds policy limit, logs each action.
        """
        # Normalize inputs
        if isinstance(statuses, str):
            statuses = [statuses]
        statuses = set(statuses)

        # Filter runs to resubmit
        to_resubmit = self.results_df[self.results_df["status"].isin(statuses)]
        # Log summary
        for status in statuses:
            count = len(self.results_df[self.results_df["status"] == status])
            self.log_mssg(f"Resubmitting {count} '{status}' runs.")

        # Process each run
        for run_index in to_resubmit["run_index"]:
            attempts = int(
                self.results_df.loc[
                    self.results_df["run_index"] == run_index, "attempts"
                ].iloc[0]
            )
            limit = self.parameters.get("Resubmission of Failed N", 0)

            if attempts >= limit:
                self.set_status(run_index, "Dead")
                self.log_mssg(
                    f"Run {run_index} marked as Dead after {attempts} attempts.",
                    level="warning",
                )
                continue

            try:
                point = self._extract_tensor_from_result_df(run_index)
                self._to_df(point, run_index=run_index, predicted_y="recover")
                self.log_mssg(
                    f"Resubmitted run {run_index} (previous status: '{self.results_df.loc[self.results_df['run_index'] == run_index, 'status'].iloc[0]}').",
                    level="ok",
                )
                self.set_status(run_index, "written")
                self.update_attempt(run_index)

            except Exception as e:
                self.log_mssg(f"Error resubmitting run {run_index}: {e}", level="error")
