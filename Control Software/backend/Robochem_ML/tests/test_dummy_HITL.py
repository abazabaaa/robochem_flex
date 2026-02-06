"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""


import unittest
from unittest.mock import patch, MagicMock
from robrains.ml_modules import DevelopmentHITL
from robrains.communication_module import ML_Platform_HITL_Development
from robrains.parameter_backends import ML_parameter

from queue import Queue
from threading import Event
import threading
from unittest import TestCase
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from unittest import skip
import time


# define two functions, one single objective and one double objective:


class ChildClass(DevelopmentHITL, ML_Platform_HITL_Development):
    def __init__(self):
        super().__init__()


class TestSingleBayesianOptiBackend(TestCase):

    def translate(self, row):
        """translate the recipe from the backend to a tensor to use in the objective function
        default value: [('Chemical', 'A', 'A', 27), ('Physical', 'B', 0.5), ('Physical', 'C', 'C')]
        """
        # translate the recipe from the backend to a tensor to use in the objective function

        # translate the recipe from the backend to a tensor to use in the objective function
        tensor = torch.zeros(8, dtype=torch.float64)

        # Assume df has one row, extract the first (and only) row

        # Process 'A' column
        if row["A"] == "A":
            tensor[0] = self.param_A.translation_continuous(row["A_conc"])
            tensor[1] = 1
        elif row["A"] == "B":
            tensor[0] = self.param_A.translation_continuous(row["A_conc"])
            tensor[2] = 1
        elif row["A"] == "C":
            tensor[0] = self.param_A.translation_continuous(row["A_conc"])
            tensor[3] = 1

        # Process 'B' column
        tensor[4] = self.param_B.translation_continuous(row["B"])

        # Process 'C' column
        if row["C"] == "A":
            tensor[5] = 1
        elif row["C"] == "B":
            tensor[6] = 1
        elif row["C"] == "C":
            tensor[7] = 1

        return tensor

    def setUp(self):
        # define parameters:
        self.param_A = ML_parameter(
            "A", "penguins", default_min=0, default_max=200, phy_chem="Chemical"
        )
        self.param_A.update_values(
            discrete=["A", "B", "C"], min_value=27, max_value=101
        )
        # only continuous
        self.param_B = ML_parameter(
            "B", "penguins", default_min=0, default_max=200, phy_chem="Physical"
        )
        self.param_B.update_values(min_value=27, max_value=101)
        # only categorical
        self.param_C = ML_parameter("C", "penguins", phy_chem="Physical")
        self.param_C.update_values(discrete=["A", "B", "C"])

        # make the tuple:
        self.parameters = (self.param_A, self.param_B, self.param_C)
        for param in self.parameters:
            param.generate_translators()

        # make the queues and the event set:
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.visualisation_queue = Queue()
        self.HITL_queue = Queue()
        self.stop_event = Event()

        # generate the kwargs for the ML backend:
        # kwargs dictionary should look like this:
        # self.kwargs = {
        # "Model": ["SingleTaskGP", "FixedNoiseGP", "NoisySingleTaskGP"],
        # "Acquisition Function": ["EI", "PI", "UCB"],
        # "Initialisation Method": ["LHS", "Random"],
        # "Number of initial points": "int",
        # "Number of total points": "int",
        # "Number of Experiments per batch": "int",
        # "Explorative Factor": "float",
        # "force_categorical": "bool",
        # "Termination criterion": ["max_iter", "max_time", "performance"],
        # }

        self.kwargs = {
            "Model": "Dummy",
            "Initialisation Method": "Random",
        }

        # instantiate the backend:
        self.backend = ChildClass()

    def run_backend(self):
        self.backend.first_run()
        self.backend.run()

    def platform_mock(self):
        while not self.stop_event.is_set():
            time.sleep(1)
            if not self.output_queue.empty():
                data = self.output_queue.get()
                print(f"Platform receives from ML: {data}")
                if data == "stop":
                    self.stop_event.set()
                    break
                run_index, x = data
                self.input_queue.put((run_index, x, np.nan, "xyz"))
                print(f"Sent data to ML: {run_index, x, np.nan, 'xyz'}")
            else:
                time.sleep(1)

    def consume_and_produce_single(self):
        """in this case we're looking at the visualisation queue
        we will get a df, in which we have to fill the target
        columns with the objective function value

        """
        while not self.stop_event.is_set():
            time.sleep(1)
            if not self.visualisation_queue.empty():
                data = self.visualisation_queue.get()
                print(f"HITL from ML recieved:")
                print(data)
                if isinstance(data, str) and data == "stop":
                    self.stop_event.set()
                    break
                if len(data) > 30:
                    self.stop_event.set()
                    break
                # choose a random number of rows to add:
                number = np.random.randint(1, 5)
                for i in range(number):
                    row = {
                        "A": "A",
                        "A_conc": np.random.randint(27, 101),
                        "B": np.random.randint(27, 101),
                        "C": "C",
                        "Objective": np.nan,
                        "vial_idx": "",
                    }
                    data = pd.concat(
                        [data, pd.DataFrame(row, index=[0])], ignore_index=True
                    )

                self.HITL_queue.put(data)
                print(f"HITL sent data to ML:")
                print(data)
            else:
                time.sleep(1)

    def test_run_single(self):
        self.backend.ML_prime(
            ML_parameters=self.parameters,
            targets=["Objective"],
            save_path="test",
        )
        self.backend.com_prime(
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            visual_queue=self.visualisation_queue,
            stop_event=self.stop_event,
            HITL_queue=self.HITL_queue,
        )

        for key, value in self.kwargs.items():
            self.backend.validate_and_update(key, value)

        backend_thread = threading.Thread(target=self.run_backend)
        platform_thread = threading.Thread(target=self.platform_mock)
        consumer_thread = threading.Thread(target=self.consume_and_produce_single)

        backend_thread.start()
        consumer_thread.start()
        platform_thread.start()

        backend_thread.join()
        consumer_thread.join()
        platform_thread.join()

        print(self.backend.results_df)
