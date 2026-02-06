"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from unittest import TestCase
import unittest
from robrains.parameter_backends import ML_parameter
from robrains.ml_modules import SingleBayesianOptiBackend, initialise_parameter
from robrains.communication_module import ML_Platform
import numpy as np
from queue import Queue
from threading import Event
import torch
import threading
import warnings
import traceback
import matplotlib.pyplot as plt


# define two functions, one single objective and one double objective:
def single_objective_function(x):
    # Ensure the input tensor is in the correct format and shape
    assert len(x) == 8
    assert -1 <= x[0] <= 1
    assert x[1] in [0, 1]
    assert x[2] in [0, 1]
    assert x[3] in [0, 1]
    assert -1 <= x[4] <= 1
    assert x[5] in [0, 1]
    assert x[6] in [0, 1]
    assert x[7] in [0, 1]
    # Ensure the input tensor is in the correct format and shape
    assert len(x) == 8
    assert -1 <= x[0] <= 1
    assert x[1] in [0, 1]
    assert x[2] in [0, 1]
    assert x[3] in [0, 1]
    assert -1 <= x[4] <= 1
    assert x[5] in [0, 1]
    assert x[6] in [0, 1]
    assert x[7] in [0, 1]

    # Extract continuous and categorical values
    continuous_values = [x[0], x[4]]
    categorical_set_1 = [x[1], x[2], x[3]]
    categorical_set_2 = [x[5], x[6], x[7]]

    # Ensure only one hot encoding per categorical set
    assert sum(categorical_set_1) == 1
    assert sum(categorical_set_2) == 1

    # Map categorical variables to their Gaussian means
    gaussian_means_set_1 = [-0.5, 0.0, 0.5]
    gaussian_means_set_2 = [0.5, -0.5, 0.0]

    # Select the means for the Gaussian based on categorical variables
    mean_1 = gaussian_means_set_1[categorical_set_1.index(1)]
    mean_2 = gaussian_means_set_2[categorical_set_2.index(1)]

    # Apply sigmoid function to continuous values to get standard deviations
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    std_1 = sigmoid(continuous_values[0])  # Standard deviation for the first Gaussian
    std_2 = sigmoid(continuous_values[1])  # Standard deviation for the second Gaussian

    # Compute the Gaussian values
    gaussian_1 = np.exp(-0.5 * ((continuous_values[0] - mean_1) / std_1) ** 2) / (
        std_1 * np.sqrt(2 * np.pi)
    )
    gaussian_2 = np.exp(-0.5 * ((continuous_values[1] - mean_2) / std_2) ** 2) / (
        std_2 * np.sqrt(2 * np.pi)
    )

    # Combine both Gaussian values
    total_score = gaussian_1 + gaussian_2

    return total_score


def double_objective_function(x):
    # Ensure the input tensor is in the correct format and shape
    # Ensure the input tensor is in the correct format and shape
    assert len(x) == 8
    assert -1 <= x[0] <= 1
    assert x[1] in [0, 1]
    assert x[2] in [0, 1]
    assert x[3] in [0, 1]
    assert -1 <= x[4] <= 1
    assert x[5] in [0, 1]
    assert x[6] in [0, 1]
    assert x[7] in [0, 1]
    # Ensure the input tensor is in the correct format and shape
    assert len(x) == 8
    assert -1 <= x[0] <= 1
    assert x[1] in [0, 1]
    assert x[2] in [0, 1]
    assert x[3] in [0, 1]
    assert -1 <= x[4] <= 1
    assert x[5] in [0, 1]
    assert x[6] in [0, 1]
    assert x[7] in [0, 1]

    # Extract continuous and categorical values
    continuous_values = [x[0], x[4]]
    categorical_set_1 = [x[1], x[2], x[3]]
    categorical_set_2 = [x[5], x[6], x[7]]

    # Ensure only one hot encoding per categorical set
    assert sum(categorical_set_1) == 1
    assert sum(categorical_set_2) == 1

    # Map categorical variables to their Gaussian means
    gaussian_means_set_1 = [-0.5, 0.0, 0.5]
    gaussian_means_set_2 = [0.5, -0.5, 0.0]

    # Select the means for the Gaussian based on categorical variables
    mean_1 = gaussian_means_set_1[categorical_set_1.index(1)]
    mean_2 = gaussian_means_set_2[categorical_set_2.index(1)]

    # Apply sigmoid function to continuous values to get standard deviations
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    std_1 = sigmoid(continuous_values[0])  # Standard deviation for the first Gaussian
    std_2 = sigmoid(continuous_values[1])  # Standard deviation for the second Gaussian

    # Compute the Gaussian values
    gaussian_1 = np.exp(-0.5 * ((continuous_values[0] - mean_1) / std_1) ** 2) / (
        std_1 * np.sqrt(2 * np.pi)
    )
    gaussian_2 = np.exp(-0.5 * ((continuous_values[1] - mean_2) / std_2) ** 2) / (
        std_2 * np.sqrt(2 * np.pi)
    )

    # Combine both Gaussian values
    # return as a dim 2 tensor
    return torch.tensor([gaussian_1, gaussian_2], dtype=torch.float64)


class ChildClass(SingleBayesianOptiBackend, ML_Platform):
    def __init__(self):
        super().__init__()


class TestSingleBayesianOptiBackend(TestCase):

    def translate(self, x):
        """translate the recipe from the backend to a tensor to use in the objective function
        default value: [('Chemical', 'A', 'A', 27), ('Physical', 'B', 0.5), ('Physical', 'C', 'C')]
        """
        # translate the recipe from the backend to a tensor to use in the objective function
        tensor = torch.zeros(8, dtype=torch.float64)
        for value in x:
            match value[1]:
                case "A":
                    tensor[0] = self.param_A.translation_continuous(value[3])
                    match value[2]:
                        case "A":
                            tensor[1] = 1
                        case "B":
                            tensor[2] = 1
                        case "C":
                            tensor[3] = 1
                case "B":
                    tensor[4] = self.param_B.translation_continuous(value[2])
                case "C":
                    match value[2]:
                        case "A":
                            tensor[5] = 1
                        case "B":
                            tensor[6] = 1
                        case "C":
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
            "Model": "SingleTaskGP",
            "Acquisition Function": "EI",
            "Initialisation Method": "LHS",
            "Number of initial points": 5,
            "Number of total points": 30,
            "Number of Experiments per batch": 1,
            "Explorative Factor": 0.1,
            "force_categorical": False,
            "Termination criterion": "max_iter",
        }

        # instantiate the backend:
        self.backend = ChildClass()

    def run_backend(self):
        self.backend.first_run()
        self.backend.run()

    def consume_and_produce_single(self):
        while not self.stop_event.is_set():
            try:
                data = self.output_queue.get(timeout=1)
                print(f"data from ML recieved: {data}")
                if data == "stop":
                    self.stop_event.set()
                    break
                run_index, x = data
                x_tensor = self.translate(x)
                y = single_objective_function(x_tensor)
                self.input_queue.put((run_index, x, y, "xyz"))
                print(f"sent data to ML: {run_index, x, y, 'xyz'}")
            except Exception as e:
                continue

    def consume_and_produce_double(self):
        while not self.stop_event.is_set():
            try:
                data = self.output_queue.get(timeout=1)
                print(f"data from ML recieved: {data}")
                if data == "stop":
                    self.stop_event.set()
                    break
                run_index, x = data
                x_tensor = self.translate(x)
                y = double_objective_function(x_tensor)
                self.input_queue.put((run_index, x, y, "xyz"))
                print(f"sent data to ML: {run_index, x, y}")
            except Exception as e:
                continue

    def test_run_single(self):
        self.backend.ML_prime(
            ML_parameters=self.parameters,
            targets=["Objective 1"],
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
        consumer_thread = threading.Thread(target=self.consume_and_produce_single)

        backend_thread.start()
        consumer_thread.start()

        backend_thread.join()
        consumer_thread.join()
        print(self.backend.train_x, self.backend.train_y)
        x = np.arange(0, len(self.backend.train_y))
        plt.scatter(
            x, self.backend.train_y, s=5, c="red", label="Single Task GP Function"
        )
        plt.hlines(1.7, 0, x[-1], color="blue", label="Objective Function Maximum")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")

        plt.show()

        self.assertTrue(self.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
