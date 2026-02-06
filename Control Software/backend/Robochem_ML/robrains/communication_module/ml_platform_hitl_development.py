"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import time

import pandas as pd
import torch

from robrains.communication_module.ml_platform_hitl import ML_Platform_HITL


class ML_Platform_HITL_Development(ML_Platform_HITL):
    """this class is a bit different and should only be used with the
    HITL_Development class
    The difference is that it won't put anything to the run queue until the user
    has put the first batch of runs in the results_df (as everything is manually
    done by the user)

    """

    def _run(self):
        """check the HITL queue for data, update the restults df and put requests
        in the other queue"""

        while not self.stop_event.is_set():
            HITL_data = self.HITL_queue.get()
            if HITL_data is None:
                self.stop_event.wait(1)
                continue

            self.in_data(HITL_data)
            self.update()

            while not self.stop_event.is_set():
                data = self.experiment.get_result()
                if data is None:
                    time.sleep(10)
                    continue
                self.log_mssg(f"Platform returned data: {data}", level="ok")
                if not self.out_data(data):
                    self.visual_queue.put(self.results_df.copy())
                    break

    def check_batch(self, run_index, x, y, vial_idx):
        """for the hitls we don't want to display results until the whole batch has been run, so we need to check
        when the platform has returned all of them and is waiting. To do this we keep accepting values until
        the vial_idx column has no more nans in it"""
        # check the tensor, if it's none it means that the experiment has not succeeded.
        # therefore we don't need to do much with it. just tell the user that the experiment has failed
        if not x is None:
            self._to_df(x, run_index)
            self.set_status(run_index=run_index, status="finished")
        else:
            self.set_status(run_index=run_index, status="failed")
        self.set_vial(run_index=run_index, vial=vial_idx)
        if any(self.results_df["status"] == "submitted"):
            return True
        else:
            return False

    def in_data(self, df: pd.DataFrame):
        """this function will take the df with emtpy target values
        make a tensor out of it, convert it to a recipe and put the recipe to
        the input_queue
        """
        # for the df set any missing or empty status value to written:

        df = self.validate_in_df(df)
        if "status" in df.columns:
            df["status"] = df["status"].fillna("written")
            df["status"] = df["status"].replace("", "written")
        _, _, _ = self._from_df(df, check_nan=False)

    def send_to_platform(self, run_id: str | int, recipe: list):
        """
        this function is used to send things to the platform
        run_index: int or string, the index of the run
        recipe: list of experimental parameters for the platform
        """
        self.log_mssg(
            f"Pushed these data to the platform. Run_ID: {run_id} ::: parameters:[{'::'.join(str(i) for i in recipe)}]",
            level="ok",
        )
        self.set_status(run_id, "submitted")
        self.experiment.submit_run(run_id=run_id, parameters=recipe)

    def out_data(self, next_points: torch.tensor):
        """
        this function takes in the next point from the machine, updates the results_df (by mostly updating the
        vial index) and checks if there are still things we need to wait for. Returns True if we need to wait
        and False if we don't
        """
        run_index, x, y, y_var, vial_idx, _ = self._from_machine(next_points)
        return self.check_batch(run_index, x, y, vial_idx)

    def res_df_is_valid(self, res_df: pd.DataFrame):
        """for dev hitl this should just check if the df is not empty and otherwise let the user push whatevs"""

        if res_df.empty:
            return False
        else:
            return True
