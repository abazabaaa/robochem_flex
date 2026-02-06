"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from queue import Empty

import pandas as pd

from robrains.communication_module.ml_platform import ML_Platform
from typing import Any, Optional
import torch


class ML_Platform_HITL(ML_Platform):
    """
    Human-in-the-loop extension of ML_Platform.

    Waits for manual inputs (HITL data) before proceeding with updates.
    """

    _hitl_tag: bool = True

    def _run(self) -> None:
        """
        Main loop for HITL: process hardware results and wait for human inputs.
        """
        self.log_mssg("Starting HITL optimization loop (_run).")
        # Resubmit any failed runs at start
        self.resubmit_runs(statuses=["failed", "retry"])
        # Initial visual update if needed
        if self._initial_check():
            self.log_mssg(
                "HITL initial check passed; processing visual queue.",
                level="ok",
                indent=1,
            )
            self.log_mssg("Pushing results_df to visual_queue for HITL.")
            self.visual_queue.put(self.results_df.copy())
            self._process_visual_queue()

        # Main processing loop
        while not self._stop_event.is_set():
            try:
                data = self.experiment.get_result(block=False)
            except Exception as e:
                self.log_mssg(f"Error fetching result in HITL _run: {e}", level="error")
                continue

            if data is None:
                self._stop_event.wait(1)
                continue

            try:
                run_index, x, y, y_var, vial_idx, _ = self._from_machine(data)
                self.log_mssg(
                    f"Received machine data for run_index={run_index}.",
                    indent=1,
                )
                self.log_mssg("Pushing results_df to visual_queue for HITL.")
                # self.visual_queue.put(self.results_df.copy())
                if self.check_batch(run_index, x, y, vial_idx):
                    self.log_mssg(
                        f"Batch {run_index} finished; waiting for HITL input.",
                        level="ok",
                        indent=1,
                    )
                    self._process_visual_queue()
            except Exception as e:
                self.log_mssg(f"Error in HITL _run processing: {e}", level="error")

        self.log_mssg("HITL optimization loop stopped.", level="ok")

    def _process_visual_queue(self) -> None:
        """
        Pushes current results to visual queue and waits for HITL input.
        """
        # Wait for human input
        while not self._stop_event.is_set():
            df_hitl = self.get_HITL()
            if df_hitl is None:
                self._stop_event.wait(10)
                continue
            self.log_mssg("Received HITL data; processing in_data.")
            try:
                self.in_data(df_hitl)
                self.update()
                self.log_mssg("HITL data processed and update() called.", level="ok")
            except Exception as e:
                self.log_mssg(f"Error processing HITL data: {e}", level="error")
            break

    def check_batch(
        self,
        run_index: Any,
        x: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        vial_idx: Any,
    ) -> bool:
        """
        Determine if an update should be triggered: wait until all runs are complete.
        """
        # Log receipt
        self.log_mssg(f"HITL check_batch called for run_index={run_index}.")
        # Update individual run status
        if x is not None:
            self._to_df(x, run_idx=run_index)
            self.set_status(run_index, "finished")
            self.log_mssg(f"Run {run_index} marked finished.", level="ok", indent=1)
        else:
            self.set_status(run_index, "failed")
            self.log_mssg(f"Run {run_index} marked failed.", level="warning", indent=1)
            self.resubmit_runs(statuses=["failed", "retry"])

        self.set_vial(run_index, vial_idx)
        self.visual_queue.put(self.results_df.copy())
        # Check if any runs still pending or failed
        pending = any(self.results_df["status"].isin(["submitted"]))
        result = not pending
        self.log_mssg(f"HITL batch complete status: {result}.", level="ok", indent=1)
        return result

    def in_data(self, df: pd.DataFrame) -> None:
        """
        Process DataFrame input from human feedback (HITL).

        :param df: DataFrame provided by frontend with human-reviewed statuses.
        """
        self.log_mssg("HITL in_data received DataFrame.", level="ok")
        df_clean = self.validate_in_df(df)
        # Standardize status
        if "status" in df_clean.columns:
            df_clean["status"] = df_clean["status"].fillna("done").replace("", "done")
            self.log_mssg(
                "Normalized status column in HITL DataFrame.", level="ok", indent=1
            )
        # Convert to internal state and skip updating results_df
        _, _, _ = self._from_df(df=df_clean)
        self.log_mssg("HITL in_data processing complete.", level="ok")

    def put_HITL(self, results_df: pd.DataFrame) -> None:
        """
        Enqueue human-reviewed DataFrame for processing.

        :param results_df: DataFrame from frontend HITL interface.
        """
        self.log_mssg("Putting HITL DataFrame into HITL_queue.", level="ok")
        self.HITL_queue.put(results_df)

    def get_HITL(self) -> Optional[pd.DataFrame]:
        """
        Retrieve human-reviewed DataFrame from the queue.

        :return: DataFrame if available, otherwise None.
        """
        try:
            df = self.HITL_queue.get_nowait()
            self.log_mssg("Retrieved DataFrame from HITL_queue.", level="ok")
            return df
        except Empty:
            return None
