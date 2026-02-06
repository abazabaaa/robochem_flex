"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from robrains.communication_module import baseMLBackend
from robrains.utils import initialise_parameter


class DevelopmentHITL(baseMLBackend):
    """This class is to manually alter values to pass to the platform in a HITL fashion for dev purposes
    so the only thing it needs to do is to check that the data is correct, make it into a tensor,
    translate it into a recipe and put it in the output queue"""

    input_parameters = {
        "Model": ["Dummy"],
        "Initialisation Method": ["Dummy"],
    }
    input_defaults = {
        "Model": "Dummy",
        "Initialisation Method": "Dummy",
    }
    ml_parameters = {}
    tags = ["ML", "dev", "HITL"]

    def __init__(self):
        super().__init__()

    def validate_and_update(self, key, value):
        self.parameters[key] = value

    def first_run(self):
        """
        the initial run for this experiment needs to
        not do anything but initialise an empty results_df
        with the right columns
        """
        initial_x = initialise_parameter(
            self.ML_parameters,
            number_of_points=2,
            method="Random",
            force_categorical=False,
        )
        # make the results_df full of empty values:
        for i in range(len(initial_x)):
            self._to_df(initial_x[i])

        self.results_df.dropna(inplace=True)
        self.visual_queue.put(self.results_df.copy())

    def update(self):
        """
        gets the current results_df, checks if there are any NaN values in the vial_idx column, if there are
        it will make a tensor out of that row and put it to the machine queue.

        :return:
        then it will put the results_df back in the results_df

        """

        dummy_results_df = self.results_df.copy()
        # find all rows where the status is "written"
        nan_rows = self.results_df[self.results_df["status"] == "written"]
        # reset index to make sure we can iterate over the rows
        nan_rows.reset_index(inplace=True)

        # find the tensor for that:
        x, y, y_var = self._from_df(nan_rows, check_nan=False, keyword="written")
        x = x.reshape(len(nan_rows), -1)
        # put the tensor in the output queue
        for i, row in nan_rows.iterrows():
            recipe = self._to_machine(x[i])
            run_index = row["run_index"]
            dummy_results_df.loc[
                dummy_results_df["run_index"] == row["run_index"], "status"
            ] = "submitted"
            self.send_to_platform(run_id=run_index, recipe=recipe)
        # put back the results_df
        self.results_df = dummy_results_df
