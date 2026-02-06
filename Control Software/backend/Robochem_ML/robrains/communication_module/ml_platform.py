"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from queue import Queue, Empty
from threading import Thread, Event
from typing import Any, Optional, Union, Dict, Iterable

import pandas as pd
import torch

from robrains.communication_module.communication_module import baseMLBackend


class ML_Platform(baseMLBackend):
    """for the platform things are a bit more complex, here we have 4 queues,
    2 for communication with hardware and 2 for communication with the frontend

    for basic platform operations we care only about 3 queues, the input queue, the output queue and the visual queue

    the input queue is the data that the platform streams to the ML backend, the output queue is the data that the ML backend
    streams to the platform and the visual queue is the data that the ML backend streams to the frontend
    """

    _running_thread = None
    _stop_event = None
    _hitl_tag = False

    def run(self) -> None:
        """
        Start the initial platform run and launch a background thread for continuous operation.

        This method:
        1. Executes `first_run()`.
        2. If no background thread is active, spawns one targeting `_run()`.
        3. Logs progress and handles errors.

        :raises Exception: Propagates any exception from `first_run()`.
        """
        # Begin run
        self.log_mssg("Invoking run(): executing first_run().")
        try:
            self.first_run()
            self.log_mssg("first_run() completed successfully.", level="ok", indent=1)
        except Exception as e:
            self.log_mssg(f"Error during first_run(): {e}", level="error")
            raise

        # Check for existing thread
        if self._running_thread is not None and self._running_thread.is_alive():
            self.log_mssg(
                "Background thread already running; skipping spawn.", level="warning"
            )
            return

        # Spawn new background thread
        thread_name = f"{self.__class__.__name__}-worker"
        self._running_thread = Thread(target=self._run, name=thread_name)
        self._running_thread.daemon = True
        self._running_thread.start()
        self.log_mssg(f"Started background thread '{thread_name}'.", level="ok")

    def kill_thread(self) -> None:
        """
        Signal the background thread to stop and wait for its termination.

        Used for graceful shutdown (e.g., from Streamlit frontend).
        """
        self.log_mssg("Initiating kill_thread(): signalling stop event.")
        if self._stop_event is None:
            self.log_mssg(
                "No stop_event to set; thread may not have been started.",
                level="warning",
            )
            return

        # Signal stop and wait for thread to finish
        self._stop_event.set()
        if self._running_thread is not None:
            self._running_thread.join()
            self.log_mssg("Background thread joined successfully.", level="ok")

    def __del__(self) -> None:
        """
        Destructor: ensure background thread is stopped on object deletion.
        """
        try:
            self.kill_thread()
        except Exception:
            # Suppress exceptions during garbage collection
            pass

    def add_more_experiments(self, more_experiments: int) -> None:
        """
        Increase the total number of experiments and restart the run loop.

        :param more_experiments: Number of additional experiments to schedule.
        """
        self.log_mssg(f"Adding {more_experiments} more experiments.")
        if self._stop_event:
            self._stop_event.clear()
        # Update total points parameter
        self.parameters["Number of total points"] = (
            self.parameters.get("Number of total points", 0) + more_experiments
        )
        self.log_mssg(
            f"Updated 'Number of total points' to {self.parameters['Number of total points']}",
            indent=1,
        )
        # Restart the run loop
        self.run()

    def _run(self) -> None:
        """
        Main optimization loop: fetch results, process data, and trigger updates.
        """
        self.log_mssg("Starting optimization loop (_run).", level="ok")

        # Initial check and update
        try:
            if self._initial_check():
                self.log_mssg(
                    "Optimization starting from non-zero results_df.",
                    indent=1,
                )
                self.log_mssg(
                    f"Initial results_df:\n{self.results_df.head()}",
                    indent=2,
                )
                self.update()
        except Exception as e:
            self.log_mssg(f"Error during initial check: {e}", level="error")
            raise

        # Push initial DataFrame to visual queue
        if hasattr(self, "results_df"):
            self.visual_queue.put(self.results_df.copy())
            self.log_mssg("Pushed initial results_df to visual_queue.", indent=1)
            self.log_mssg(
                f"Initial results_df:\n{self.results_df.head()}", level="none", indent=2
            )

        # Loop until stop event is signalled
        while not self._stop_event.is_set():
            try:
                data = self.experiment.get_result(block=False)
            except Exception as e:
                self.log_mssg(
                    f"Error fetching result from experiment: {e}", level="error"
                )
                continue

            if data is None:
                self._stop_event.wait(1)
                continue

            try:
                self.log_mssg(f"Received data: {data}", indent=1)
                run_index, x, y, y_var, vial_idx, save_file_name = self._from_machine(
                    data
                )
                self.in_data(run_index, x, y, y_var, vial_idx)
                self.visual_queue.put(self.results_df.copy())
                self.log_mssg(
                    f"Processed result for run_index={run_index}.", level="ok", indent=1
                )

                if self.check_batch(run_index, x, y, vial_idx):
                    self.log_mssg(
                        "Batch complete; running update().", level="ok", indent=1
                    )
                    self.update()
            except Exception as e:
                self.log_mssg(f"Error processing machine data: {e}", level="error")

        self.log_mssg("Optimization loop stopped.", level="ok")

    def _initial_check(self) -> bool:
        """
        Determine if an initial update is required when resuming from existing data.
        """
        result = (
            hasattr(self, "results_df")
            and not any(self.results_df["status"] == "submitted")
            and getattr(self, "run_index", 0) > 0
        )
        self.log_mssg(f"_initial_check returns {result}.", level="ok", indent=1)
        return result

    def check_batch(
        self,
        run_index: Any,
        x: Any,
        y: Any,
        vial_idx: Any,
    ) -> bool:
        """
        Check if all submitted runs have completed (no 'submitted' status remains).
        """
        incomplete = any(self.results_df["status"] == "submitted")
        result = not incomplete
        self.log_mssg(f"check_batch for run_index={run_index} -> {result}.", indent=1)
        return result

    def com_prime(self, **kwargs):
        """Initialises the class communication for the ML platform version

        in the ml platform version we need to initialise the queues and the stop event
        """
        self.visual_queue = Queue()
        self.HITL_queue = Queue()
        self._stop_event = Event()
        self.experiment = kwargs.get("experiment", None)
        self.constants = kwargs.get("constants", [])
        self.parameter_machine = kwargs.get("parameter_machine", [])

        if self.experiment is None:
            self.log_mssg(
                "No experiment passed to the platform, IF YOU are reloading from json, ignore this",
                level="warning",
            )
        self._com_primed = True

    def in_data(
        self,
        run_index: int,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        y_var: Optional[torch.Tensor],
        vial_idx: Any,
        save_file_name: Optional[str] = None,
    ) -> None:
        """
        Process incoming experiment results and update `results_df`.

        :param run_index: Identifier of the completed run.
        :param x: Feature tensor for the run.
        :param y: Target tensor for the run.
        :param y_var: Variance tensor for the run.
        :param vial_idx: Identifier of the vial used in the experiment.
        :param save_file_name: Optional filename where results were saved.
        """
        # Handle failures
        self.log_mssg(f"In-Data processing run_index={run_index}.")
        if x is None:
            self.set_status(run_index, "failed")
            return
        # Insert features and finalize status
        self._to_df(x, run_idx=run_index, predicted_y="recover")
        self.set_status(run_index, "finished")
        self.set_vial(run_index, vial_idx)
        # Add targets and variances
        if y is not None:
            self._add_y_to_df(run_index, y, y_var)
        if save_file_name:
            self._add_savename_to_df(run_index=run_index, savename=save_file_name)
        self.log_mssg(
            f"Successfully processed in_data. run_index={run_index}", level="ok"
        )

    def out_data(
        self,
        next_points: Union[torch.Tensor, str],
        predicted_y: Optional[Union[torch.Tensor, tuple]] = None,
    ) -> None:
        """
        Prepare and submit new experimental runs to the platform.

        :param next_points: Tensor of feature points or control strings ('failures', 'stop').
        :param predicted_y: Optional predictions for each point.
        """
        # Handle control commands
        self.log_mssg(f"out_data received: {next_points}", level="ok")
        if (
            isinstance(next_points, str)
            and next_points == "failures"
            and self.parameters.get("Resubmission of Failed N", 0) != 0
        ):
            self.resubmit_runs(statuses=["failed", "retry"])
            return

        if isinstance(next_points, str) and next_points == "stop":
            self.visual_queue.put("stop")
            self._stop_event.set()
            return

        # Submit new runs
        recipes: Dict[int, Any] = {}
        for i, point in enumerate(
            next_points
            if isinstance(next_points, list) or isinstance(next_points, torch.Tensor)
            else []
        ):
            self.log_mssg(f"Processing point {i}: {point}")
            self.run_index = (self.run_index or 0) + 1
            preds = None
            if predicted_y is not None:
                means, vars_ = predicted_y
                preds = (means[i], vars_[i])
            self._to_df(point, run_idx=self.run_index, predicted_y=preds)
            recipe = self._to_machine(point)
            recipes[self.run_index] = recipe
            self.log_mssg(f"Generated recipe for run {self.run_index}", level="ok")

        try:
            order = sorted(
                recipes.keys(),
                key=lambda idx: next(
                    p.value for p in recipes[idx] if p.name == "temperature"
                ),
            )
            self.log_mssg(f"Sorted order of runs by temperature: {order}", level="ok")
        except StopIteration:
            # at least one run missing “temperature” → fall back to insertion order
            order = recipes.keys()

        # Submit them all exactly once
        for run_id in order:
            params = recipes[run_id]
            self.experiment.submit_run(run_id=run_id, parameters=params)
            self.log_mssg(f"Submitted run {run_id} with parameters:", level="ok")
            for p_index, param in enumerate(params):
                self.log_mssg(f"{p_index}: {param} ", indent=1)
            self.set_status(run_id, "submitted")
            self.update_attempt(run_id)

        # Trigger resubmission logic if needed
        if hasattr(self, "results_df"):
            self.resubmit_runs(statuses=["check"])

    def get_visual(self) -> Optional[pd.DataFrame]:
        """
        Retrieve the latest DataFrame from the visual queue for frontend display.

        :return: DataFrame from the queue, or None if no data is available.
        """
        data = None
        while True:
            try:
                data = self.visual_queue.get_nowait()
                self.visual_queue.task_done()
                self.log_mssg("Retrieved data from visual_queue.", level="ok")
            except Empty:
                break
        return data

    def res_df_is_valid(self, res_df: pd.DataFrame) -> bool:
        """
        Validate an external DataFrame before pushing to the machine.

        :param res_df: DataFrame the user intends to push.
        :return: True if all 'finished' target columns contain no NaNs.
        """
        df = res_df.copy()
        df = df[df["status"] == "finished"]
        for target in self.targets:
            if df[target].isnull().any():
                self.log_mssg(
                    f"Validation failed: NaN values in target '{target}'.",
                    level="error",
                )
                return False
        self.log_mssg("Validation passed for external results_df.", level="ok")
        return True

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
                recipe = self._to_machine(point)
                self.experiment.submit_run(run_id=run_index, parameters=recipe)
                self.log_mssg(
                    f"Resubmitted run {run_index} (previous status: '{self.results_df.loc[self.results_df['run_index'] == run_index, 'status'].iloc[0]}').",
                    level="ok",
                )
                self.set_status(run_index, "submitted")
                self.update_attempt(run_index)
            except Exception as e:
                self.log_mssg(f"Error resubmitting run {run_index}: {e}", level="error")
