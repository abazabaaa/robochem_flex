"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: unit tasks for the measuring module, includes raman spectrometer connection and control,
data acquisition (processing and analysis done separately)

"""

import time

import pandas as pd
from omniplatypus.procedures.unit_tasks.base_unit_task import BaseUnitTaskTemplate
from omniplatypus.devices.nrg.rama_berry import RamaBerry


# TODO: implement the unittasks:
# TODO: 1. handle parameter setting
# TODO: 2. send data acqquisition and start the polling
# TODO: 3. handle the returning data
# TODO: 4. handle the stop acquisition


def handle_error(func):
    """decorator to handle errors"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise e

    return wrapper


class SetRamanParameters(BaseUnitTaskTemplate):
    """SetRamanParameters class. Sets the parameters for the Raman spectrometer.

    Usage: SetRamanParameters.run(raman, parameters) ->void

    :param raman: Raman spectrometer object (RamaBerry)
    :param parameters: dictionary of parameters to set
    example:
    parameters = {
        "integration_time": 1000, # ms
        "n_averages": 1,
        "delay": 0,
        "n_scans": 1,
        "return_as": "DataFrame",
        "dark_correction": False,
    """

    @classmethod
    def _validate_input(cls, raman: RamaBerry, parameters: dict):
        """Validate input arguments."""
        cls._validate_input_log(raman, RamaBerry, "raman must be a RamaBerry object.")
        cls._validate_input_log(parameters, dict, "parameters must be a dictionary.")
        cls._validate_input_log(
            parameters.keys(),
            lambda x: all([isinstance(i, str) for i in x]),
            "keys of parameters must be strings.",
        )

    @classmethod
    def _execute(cls, raman: RamaBerry, parameters: dict):
        # valid parametres:
        valid_parameters = [
            "integration_time",
            "n_averages",
            "delay",
            "n_scans",
            "return_as",
            "dark_correction",
        ]

        # write the parameters to the spectrometer:
        for param in parameters:
            if param in valid_parameters:
                raman[param] = parameters[param]
            else:
                print(f"Parameter {param} not valid")


class StartAcquisition(BaseUnitTaskTemplate):
    """StartAcquisition class. Starts the acquisition of data.

    Usage: StartAcquisition.run(raman) ->void

    :param raman: Raman spectrometer object (RamaBerry)
    """

    @classmethod
    def _validate_input(cls, raman: RamaBerry):
        """Validate input arguments."""
        cls._validate_input_log(raman, RamaBerry, "raman must be a RamaBerry object.")

    @classmethod
    def _execute(cls, raman: RamaBerry):
        raman["acq_data"] = True


class StopAcquisition(BaseUnitTaskTemplate):
    """StopAcquisition class. Stops the acquisition of data.

    Usage: StopAcquisition.run(raman) ->void

    :param raman: Raman spectrometer object (RamaBerry)
    """

    @classmethod
    def _validate_input(cls, raman: RamaBerry):
        """Validate input arguments."""
        cls._validate_input_log(raman, RamaBerry, "raman must be a RamaBerry object.")

    @classmethod
    def _execute(cls, raman: RamaBerry):
        raman["stop_acq"] = True


class GetRamanData(BaseUnitTaskTemplate):
    """GetData class. Gets the data from the Raman spectrometer.

    Usage: GetData.run(raman, delay_time, aggregate) ->void

    :param raman: Raman spectrometer object (RamaBerry)
    :param delay_time: delay time between data polling
    :param aggregate: aggregate the data into a single DataFrame
    """

    @classmethod
    def _validate_input(cls, raman: RamaBerry, delay_time: float, aggregate: bool):
        """Validate input arguments."""
        cls._validate_input_log(raman, RamaBerry, "raman must be a RamaBerry object.")
        cls._validate_input_log(delay_time, float, "delay_time must be a float.")
        cls._validate_input_log(aggregate, bool, "aggregate must be a boolean.")

    @staticmethod
    def _check_dataframe(data):
        """Check if data is a DataFrame with 'Rshift' column and at least 2 columns"""
        return (
            isinstance(data, pd.DataFrame)
            and "Rshift" in data.columns
            and len(data.columns) >= 2
        )

    @staticmethod
    def _process_dataframe(data, data_frame, aggregate=False):
        """Process the DataFrame based on aggregation preference"""
        if data_frame is None:
            return data
        else:
            if aggregate:
                # Aggregate by concatenating the last column of new data to the existing dataframe
                return pd.concat([data_frame, data.iloc[:, -1]], axis=1)
            else:
                # If not aggregating, just return the new data frame
                return data

    @staticmethod
    def _process_data(data, data_frame, aggregate=False):
        """Process the data based on type and aggregation preference"""

        if GetRamanData._check_dataframe(data):
            return GetRamanData._process_dataframe(data, data_frame, aggregate)

        return data_frame

    @staticmethod
    def _poll_data(raman, delay_time):
        """Poll data from the raman object"""
        time.sleep(delay_time)
        try:
            return raman["data"]
        except Exception as e:
            print(f"Error while polling data: {e}")
            return "done"

    @classmethod
    def _execute(cls, raman: RamaBerry, delay_time: float, aggregate: bool):
        """poll the data from the spectrometer until you get 'done' as a response
        returns the data in a data frame
        The 'aggregate' parameter determines if multiple spectra should be aggregated into a single DataFrame.
        """
        data_frame = None

        while True:
            data = cls._poll_data(raman, delay_time)
            if data == "done":
                break  # Exit loop if 'done' or None
            elif data is not None:
                data_frame = cls._process_data(data, data_frame, aggregate)
                if not aggregate:
                    yield data_frame  # Yield the data frame immediately if not aggregating

        if aggregate:
            yield data_frame
