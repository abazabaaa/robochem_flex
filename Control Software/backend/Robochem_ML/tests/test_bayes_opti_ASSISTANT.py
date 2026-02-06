"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from unittest import TestCase
from robrains.parameter_backends import ML_parameter, PhysicalParameter
from robrains.ml_modules import (
    SingleBayesianOptiBackend,
    ScopeAcceleratorFidelityBackend,
    ScopeAcceleratorTaskBackend,
    AllScopeTaskBackend,
)
from robrains.communication_module.ml_assistant import ML_Assistant
import torch
import matplotlib.pyplot as plt
import pandas as pd
from unittest import skip

# Function to handle warnings as exceptions and print full traceback
# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#     log = traceback.format_stack()
#     warning_message = warnings.formatwarning(message, category, filename, lineno, line)
#     print(f"{warning_message}\n{''.join(log)}")
#
# # Set the custom warning handler
# warnings.showwarning = warn_with_traceback


# define two functions, one single objective and one double objective:
import numpy as np
from scipy.optimize import minimize


def maximize_over_combinations(func):
    # Define the 3 categorical values for x[0] and x[3]
    categorical_values_1 = [0.0, 0.5, 1.0]  # For x[0] (3 categories)
    categorical_values_2 = [0.0, 0.5, 1.0]  # For x[3] (3 categories)

    max_value = -np.inf  # Initialize to a very low value
    best_combination = None

    # Loop over all 9 combinations of categorical values
    for cat1 in categorical_values_1:
        for cat2 in categorical_values_2:
            # Function to optimize over continuous values x[1] and x[2]
            def func_to_optimize(continuous_values):
                x = [cat1, continuous_values[0], continuous_values[1], cat2]
                return -func(
                    x
                )  # Minimize the negative of the function to find the maximum

            # Initial guess for x[1] and x[2]
            initial_guess = [0.5, 0.5]

            # Optimize over continuous variables (x[1] and x[2])
            result = minimize(func_to_optimize, initial_guess, bounds=[(0, 1), (0, 1)])

            # Calculate the maximum value for this combination
            max_for_combination = (
                -result.fun
            )  # Negate the result since we minimized the negative

            # Check if this is the new maximum
            if max_for_combination > max_value:
                max_value = max_for_combination
                best_combination = [cat1, result.x[0], result.x[1], cat2]

    return max_value, best_combination


def plot_single(func, cat_1=0.5, cat_2=0.5):
    # Set fixed values for the categorical variables
    single_objective_function = func
    categorical_1 = cat_1  # This represents the middle of the three categories for x[0]
    categorical_2 = cat_2  # This represents the middle of the three categories for x[3]

    # Create a grid of continuous values for x[1] and x[2]
    x1_values = np.linspace(0, 1, 100)  # x[1] ranges between 0 and 1
    x2_values = np.linspace(0, 1, 100)  # x[2] ranges between 0 and 1

    X1, X2 = np.meshgrid(x1_values, x2_values)

    # Compute the output for each combination of x[1] and x[2]
    Z = np.array(
        [
            [
                single_objective_function([categorical_1, x1, x2, categorical_2])
                for x1 in x1_values
            ]
            for x2 in x2_values
        ]
    )

    # Plot the result as a 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X1, X2, Z, cmap="viridis")

    ax.set_xlabel("x[1] (Continuous)")
    ax.set_ylabel("x[2] (Continuous)")
    ax.set_zlabel("Output")

    plt.show()


def single_objective_function(x: torch.tensor) -> float:
    """
    Takes as input a tensor (list) of shape [6] with values between 0 and 1,
    and returns a single objective value by combining Gaussian values.

    Args:
        x (List[float]): A list of 6 float values between 0 and 1.
            - x[0:2] and x[3:5] are two-dimensional embeddings (for categorical variables)
            - x[2] and x[5] are continuous values used in sigmoid for standard deviations.

    Returns:
        float: The combined value of two Gaussian functions.
    """
    # Ensure the input tensor is in the correct format and shape
    assert len(x) == 6, "Input list must have exactly 6 elements."
    assert all(0 <= i <= 1 for i in x), "All values must be between 0 and 1."

    # Extract continuous and categorical (embedded) values
    continuous_values = [x[2], x[5]]
    categorical_embedding_1 = x[0:2]
    categorical_embedding_2 = x[3:5]

    # Map categorical embeddings to their Gaussian means
    gaussian_means_set_1 = [-0.5, 0.0, 0.5]
    gaussian_means_set_2 = [0.5, -0.5, 0.0]

    # Compute the mean for each categorical embedding as the sum of the two embedding dimensions
    transform = lambda embedding, array: array[int(min(sum(embedding) * 1.5, 2))]
    mean_1 = transform(categorical_embedding_1, gaussian_means_set_1)
    mean_2 = transform(categorical_embedding_2, gaussian_means_set_2)

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


def multitask_objective_function(x):
    """
    Function for a multitask model where the second item (x[1]) encodes the task.
    Each task has a different function maximum position and magnitude.

    :param x: Input tensor of shape [5].
    :return: A single value representing the output of the objective function for the given task.
    """
    assert len(x) == 5, "Input tensor must have exactly 5 elements."
    # assert all(0 <= i <= 1 for i in x), "All tensor values must be between 0 and 1."

    # Task is encoded in x[1]
    task = int(x[1])  # Task derived from x[1] (discretize into 3 tasks)

    # Extract continuous values from x[0] and x[2]
    continuous_values = [x[0], x[2]]

    # Extract categorical embedding vector from x[3] and x[4]
    categorical_embedding = x[3:5]

    # Define different behavior for different tasks (varying Gaussian means and magnitude)
    if task == 0:
        gaussian_means_set_1 = [-0.5, 0.0, 0.5]
        gaussian_means_set_2 = [0.5, -0.5, 0.0]
        magnitude_factor = 1.0  # Task 0 has the default magnitude
    elif task == 1:
        gaussian_means_set_1 = [-0.6, 0.1, 0.4]
        gaussian_means_set_2 = [0.4, -0.6, 0.1]
        magnitude_factor = 1.2  # Task 1 has a slightly larger magnitude
    elif task == 2:
        gaussian_means_set_1 = [-0.5, 0.0, 0.5]
        gaussian_means_set_2 = [0.5, -0.5, 0.0]
        magnitude_factor = 1  # Task 2 has a slightly smaller magnitude
    else:
        raise ValueError("Unsupported task identifier.")

    # Select the means for the Gaussian based on categorical variable
    transform = lambda embedding, array: array[int(min(sum(embedding) * 1.5, 2))]
    mean_1 = transform(categorical_embedding, gaussian_means_set_1)
    mean_2 = transform(categorical_embedding, gaussian_means_set_2)

    # Apply sigmoid function to continuous values to get standard deviations
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    std_1 = sigmoid(continuous_values[0])
    std_2 = sigmoid(continuous_values[1])

    # Compute the Gaussian values
    gaussian_1 = np.exp(-0.5 * ((continuous_values[0] - mean_1) / std_1) ** 2) / (
        std_1 * np.sqrt(2 * np.pi)
    )
    gaussian_2 = np.exp(-0.5 * ((continuous_values[1] - mean_2) / std_2) ** 2) / (
        std_2 * np.sqrt(2 * np.pi)
    )

    # Combine both Gaussian values and apply the magnitude factor
    total_score = (gaussian_1 + gaussian_2) * magnitude_factor

    return total_score


def multifidelity_objective_function(x):
    """
    Function for a multifidelity model where the fidelity is encoded at position 1 of the tensor.
    - Fidelity at x[1] == 1: high-fidelity evaluation.
    - Fidelity at x[1] < 1: low-fidelity evaluation.

    :param x: Input tensor of shape [5].
    :return: A single value representing the output of the objective function.
    """
    assert len(x) == 5, "Input tensor must have exactly 5 elements."
    assert all(0 <= i <= 1 for i in x), "All tensor values must be between 0 and 1."

    # Fidelity is encoded at position 1 (x[1])
    fidelity = x[1]  # Fidelity level (1.0 = high fidelity, <1.0 = low fidelity)

    # Extract continuous values from x[0] and x[2]
    continuous_values = [x[0], x[2]]

    # Extract categorical embedding vector from x[3] and x[4]
    categorical_embedding = x[3:5]

    # Gaussian means for the categorical embeddings
    gaussian_means_set_1 = [-0.5, 0.0, 0.5]
    gaussian_means_set_2 = [0.5, -0.5, 0.0]

    # Calculate the mean for each categorical embedding as the sum of the embedding dimensions
    transform = lambda embedding, array: array[int(min(sum(embedding) * 1.5, 2))]
    mean_1 = transform(categorical_embedding, gaussian_means_set_1)
    mean_2 = transform(categorical_embedding, gaussian_means_set_2)

    # Apply sigmoid function to continuous values to get standard deviations
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    std_1 = sigmoid(continuous_values[0])
    std_2 = sigmoid(continuous_values[1])

    # Compute the Gaussian values
    gaussian_1 = np.exp(-0.5 * ((continuous_values[0] - mean_1) / std_1) ** 2) / (
        std_1 * np.sqrt(2 * np.pi)
    )
    gaussian_2 = np.exp(-0.5 * ((continuous_values[1] - mean_2) / std_2) ** 2) / (
        std_2 * np.sqrt(2 * np.pi)
    )

    # Combine both Gaussian values
    total_score = gaussian_1 + gaussian_2

    # Introduce fidelity-based noise or scaling if fidelity is low
    if fidelity < 1.0:
        noise = np.random.normal(0, 0.1) * (
            0.1 - fidelity
        )  # Add noise based on fidelity
        # total_score *= fidelity  # Scale the function output based on fidelity
        total_score += noise  # Add noise to the function output

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


class ChildClass(SingleBayesianOptiBackend, ML_Assistant):
    def __init__(self):
        super().__init__()


class FidelityChildClass(ScopeAcceleratorFidelityBackend, ML_Assistant):
    def __init__(self):
        super().__init__()


class TaskChildClass(ScopeAcceleratorTaskBackend, ML_Assistant):
    def __init__(self):
        super().__init__()


class AllTaskChildClass(AllScopeTaskBackend, ML_Assistant):
    def __init__(self):
        super().__init__()


class TestSingleBayesianOptiBackend(TestCase):

    def translate(self, row):
        """translate the recipe from the backend to a tensor to use in the objective function
        default value: [('Chemical', 'A', 'A', 27), ('Physical', 'B', 0.5), ('Physical', 'C', 'C')]
        """
        # translate the recipe from the backend to a tensor to use in the objective function
        tensor = torch.zeros(6, dtype=torch.float64)
        # tensor shape should be now [A, A_conc, B, C]
        # Assume df has one row, extract the first (and only) row

        # Process 'A' column
        tensor[[0, 1]] = self.param_A.translation_discrete(row["A"])
        tensor[2] = self.param_A.translation_continuous(row["A_conc"])

        # Process 'B' column
        tensor[3] = self.param_B.translation_continuous(row["B"])
        # process 'C' column
        tensor[[4, 5]] = self.param_C.translation_discrete(row["C"])

        return tensor

    def translate_fidelity(self, row):
        # translate the recipe from the backend to a tensor to use in the objective function
        tensor = torch.zeros(4, dtype=torch.float64)
        # tensor shape should be now [A, A_conc, B, C]
        tensor[1] = self.param_A.translation_fidelity(row["A"])
        tensor[0] = self.param_A.translation_continuous(row["A_conc"])
        tensor[2] = self.param_B.translation_continuous(row["B"])
        tensor[[3, 4]] = self.param_C.translation_discrete(row["C"])

    def translate_multitask(self, row):
        # translate the recipe from the backend to a tensor to use in the objective function
        tensor = torch.zeros(4, dtype=torch.float64)
        # tensor shape should be now [A, A_conc, B, C]
        tensor[1] = self.param_A.translation_task(row["A"])
        tensor[0] = self.param_A.translation_continuous(row["A_conc"])
        tensor[2] = self.param_B.translation_continuous(row["B"])
        tensor[[3, 4]] = self.param_C.translation_discrete(row["C"])

    def setUp(self):
        # define parameters:
        self.param_A = ML_parameter(name="A_ml")
        self.param_A.update_values(
            discrete=["A", "B", "C"], min_value=27, max_value=101
        )
        b_phisical = PhysicalParameter(
            name="B", min_value=27, max_value=101, unit="penguins"
        )
        # only continuous
        self.param_B = ML_parameter("B", b_phisical)
        self.param_B.update_values(min_value=27, max_value=101)
        # only categorical
        c_physical = PhysicalParameter(
            name="C", unit="penguins", min_value=27, max_value=101
        )
        self.param_C = ML_parameter("C", c_physical)

        self.param_C.update_values(discrete=["A", "B", "C"])

        # make the tuple:
        self.parameters = (self.param_A, self.param_B, self.param_C)
        for param in self.parameters:
            param.generate_translators()

        # make the queues and the event set:

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
            "Acquisition Function": "UCB",
            "Initialisation Method": "LHS",
            "Number of initial points": 20,
            "Number of total points": 50,
            "Number of Experiments per batch": 1,
            "Explorative Factor": 2.0,
            "force_categorical": False,
            "Termination criterion": "max_iter",
        }

        # instantiate the backend:
        self.backend = ChildClass()

        properties = dir(self.backend)

    def test_run_single(self):
        # Call first_run to get the first DataFrame
        self.backend.ML_prime(
            ML_parameters=[self.param_A, self.param_B, self.param_C],
            targets=["target"],
            save_path="test",
        )
        print(self.backend.primed)
        self.backend.com_prime()
        print(self.backend.primed)
        for key, value in self.kwargs.items():
            self.backend.validate_and_update(key, value)

        first_df = self.backend.first_run()

        # Calculate the output for each row and store it in the target column
        while (
            max(first_df["run_index"])
            <= self.backend.parameters["Number of total points"]
        ):
            for index, row in first_df.iterrows():
                if pd.isna(row["target"]):
                    x = self.translate(row)
                    first_df.at[index, "target"] = single_objective_function(x).item()
                    first_df.at[index, "target"] = np.random.random()
                    first_df.at[index, "status"] = "finished"
            # Push the results back to the backend
            self.backend.in_data(first_df)

            self.backend.update()
            first_df = self.backend.res_df
            print(first_df)
        first_df.to_csv("test_2.csv")
        # Calculate the output for rows where target is NaN
        plt.plot(
            first_df["run_index"],
            first_df["target"],
            c="red",
            label="Single Task GP Function",
        )
        # plt.hlines(1.7, 0, x[-1], color="blue", label="Objective Function Maximum")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.show()

    def test_monte_carlo(self):
        """tests the optimisation a bunch of times and collects the optimised conditions"""
        # Call first_run to get the first DataFrame
        for i in range(100):
            self.backend = ChildClass()
            self.backend.ML_prime(
                ML_parameters=[self.param_A, self.param_B, self.param_C],
                targets=["target"],
                save_path="test",
            )
            print(self.backend.primed)
            self.backend.com_prime()
            print(self.backend.primed)
            for key, value in self.kwargs.items():
                self.backend.validate_and_update(key, value)

            first_df = self.backend.first_run()

            # Calculate the output for each row and store it in the target column
            while (
                max(first_df["run_index"])
                <= self.backend.parameters["Number of total points"]
            ):
                for index, row in first_df.iterrows():
                    if pd.isna(row["target"]):
                        x = self.translate(row)
                        first_df.at[index, "target"] = single_objective_function(
                            x
                        ).item()
                        first_df.at[index, "status"] = "finished"
                # Push the results back to the backend
                self.backend.in_data(first_df)

                self.backend.update()
                first_df = self.backend.res_df

            print(first_df)
            first_df.to_csv(f"test_data/optimisation_{i}.csv")

    @skip("Multitask not implemented")
    def test_run_single_fidelity(self):
        # Call first_run to get the first DataFrame
        self.param_A.fidelity_feature = True
        self.param_A.fidelity_value = ["A"]
        self.param_A.generate_translators()

        self.backend = FidelityChildClass()
        self.kwargs = {
            "Model": "SingleTaskMultiFidelityGP",
            "Acquisition Function": "EI",
            "Initialisation Method": "LHS",
            "Number of initial points": 10,
            "Number of total points": 30,
            "Number of Experiments per batch": 1,
            "Explorative Factor": 0.1,
            "force_categorical": False,
            "Termination criterion": "max_iter",
            "Resubmission of Failed N": 0,
        }
        self.backend.ML_prime(
            ML_parameters=[self.param_A, self.param_B, self.param_C],
            targets=["target"],
            save_path="test",
        )
        print(self.backend.primed)
        self.backend.com_prime()
        print(self.backend.primed)
        for key, value in self.kwargs.items():
            self.backend.validate_and_update(key, value)

        first_df = self.backend.first_run()
        # randomly change some values to B and C on the first column:

        # Calculate the output for each row and store it in the target column
        while (
            max(first_df["run_index"])
            <= self.backend.parameters["Number of total points"]
        ):
            for index, row in first_df.iterrows():
                if pd.isna(row["target"]):
                    x = self.translate(row)
                    first_df.at[index, "target"] = multifidelity_objective_function(
                        x
                    ).item()
                    first_df.at[index, "status"] = "finished"
            # Push the results back to the backend
            self.backend.in_data(first_df)

            self.backend.update()
            first_df = self.backend.res_df
            print(first_df)

        # Calculate the output for rows where target is NaN
        plt.plot(
            first_df["run_index"],
            first_df["target"],
            c="red",
            label="Single Task GP Function",
        )
        # plt.hlines(1.7, 0, x[-1], color="blue", label="Objective Function Maximum")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.show()

    @skip("Double objective function not implemented")
    def test_run_double(self):
        # change kwargs to double objective
        self.kwargs["Acquisition Function"] = "qEHVI"
        self.kwargs["number_of_objectives"] = 2
        self.backend.ML_prime(
            ML_parameters=[self.param_A, self.param_B, self.param_C],
            targets=["target_1", "target_2"],
            save_path="test",
        )
        self.backend.com_prime()
        for key, value in self.kwargs.items():
            self.backend.validate_and_update(key, value)

        first_df = self.backend.first_run()

        # Calculate the output for each row and store it in the target column
        while len(first_df) < self.backend.parameters["Number of total points"]:
            for index, row in first_df.iterrows():
                if pd.isna(row["target_1"]):
                    x = self.translate(row)
                    res = double_objective_function(x)
                    first_df.at[index, "target_1"] = res[0].item()
                    first_df.at[index, "target_2"] = res[1].item()

            # Push the results back to the backend
            self.backend.in_data(first_df)
            self.backend.update()
            first_df = self.backend.res_df
            print(first_df)

        plt.scatter(
            first_df["target_1"],
            first_df["target_2"],
            c="red",
            label="Double Task GP Function",
        )
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.title("Pareto Front")
        plt.show()


class TestFidelityBayesianOptiBackend(TestCase):

    def translate_fidelity(self, row):
        """
        Translate the recipe from the backend to a tensor to use in the objective function
        for the fidelity case. The tensor includes the fidelity value as the first element.
        """
        tensor = torch.zeros(5, dtype=torch.float64)
        # Assume df has one row, extract the first (and only) row

        # Process 'A' column (continuous) and fidelity value
        tensor[0] = self.param_A.translation_continuous(row["A_conc"])
        tensor[1] = self.param_A.translation_fidelity(row["A"])

        # Process 'B' column (continuous)
        tensor[2] = self.param_B.translation_continuous(row["B"])

        # Process 'C' column (discrete)
        tensor[[3, 4]] = self.param_C.translation_discrete(row["C"])

        return tensor

    def setUp(self):
        # Setup parameters for fidelity testing:
        self.param_A = ML_parameter(name="A_ml")
        self.param_A.update_values(
            discrete=["A", "B", "C"], min_value=27, max_value=101
        )

        # Setting fidelity as a feature in param_A
        self.param_A.fidelity_feature = True
        self.param_A.fidelity_value = ["A"]
        self.param_A.generate_translators()

        self.param_B = ML_parameter(
            "B",
            PhysicalParameter(name="B", min_value=27, max_value=101, unit="penguins"),
        )
        self.param_B.update_values(min_value=27, max_value=101)

        self.param_C = ML_parameter(
            "C",
            PhysicalParameter(name="C", unit="penguins", min_value=27, max_value=101),
        )
        self.param_C.update_values(discrete=["A", "B", "C"])

        # Store parameters in a tuple
        self.parameters = (self.param_A, self.param_B, self.param_C)

        # Instantiate the backend
        self.backend = FidelityChildClass()

        # Define backend kwargs
        self.kwargs = {
            "Model": "SingleTaskMultiFidelityGP",
            "Acquisition Function": "UCB",
            "Initialisation Method": "LHS",
            "Number of initial points": 5,
            "Number of total points": 100,
            "Number of Experiments per batch": 1,
            "Explorative Factor": 5.0,
            "force_categorical": False,
            "Termination criterion": "max_iter",
        }

    def test_run_fidelity(self):
        # Start ML prime
        self.backend.ML_prime(
            ML_parameters=[self.param_A, self.param_B, self.param_C],
            targets=["target"],
            save_path="test",
        )

        self.backend.com_prime()

        for key, value in self.kwargs.items():
            self.backend.validate_and_update(key, value)

        first_df = self.backend.first_run()
        first_df["A"] = "A"

        first_df_2 = pd.read_csv("test.csv", index_col=0)
        # first_df_2["A"] = np.random.choice(["B", "C"], size=len(first_df_2))
        first_df_2["vial_idx"] = ""
        # remove the last row:
        first_df_2 = first_df_2[:-1]

        # concatenate the two dataframes
        first_df = pd.concat([first_df_2, first_df])

        # Calculate the output for each row and store it in the target column
        while (
            max(first_df["run_index"])
            <= self.backend.parameters["Number of total points"]
        ):
            for index, row in first_df.iterrows():
                if pd.isna(row["target"]):
                    x = self.translate_fidelity(row)
                    first_df.at[index, "target"] = multifidelity_objective_function(
                        x
                    ).item()
                    first_df.at[index, "status"] = "finished"

            # Push the results back to the backend
            self.backend.in_data(first_df)
            self.backend.update()
            first_df = self.backend.res_df
            print(first_df)

        # Plot results
        # plot the results:
        # first plot the data of the first run

        # Plot first_df_2 against its index (which represents the line number)
        plt.plot(
            first_df_2.index,
            first_df_2["target"],
            c="blue",
            label="Initial Low Fidelity",
        )

        # Remove the first 50 rows from first_df
        first_df_trimmed = first_df.iloc[50:]

        # Reset the index of the trimmed first_df and add 50 to it to continue from the 51st position
        # Adjust the index: if the index is less than 50, add 50, otherwise keep it as is
        adjusted_index = first_df_trimmed.index.map(lambda x: x + 50 if x < 50 else x)

        # Plot the trimmed first_df against the adjusted index
        plt.plot(adjusted_index, first_df_trimmed["target"], c="red", label="Learning")

        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.legend()
        plt.show()


class TestMultitaskBayesianOptiBackend(TestCase):

    def translate_multitask(self, row):
        """
        Translate the recipe from the backend to a tensor to use in the objective function
        for the multitask case. The tensor includes the task identifier as the first element.
        """
        tensor = torch.zeros(5, dtype=torch.float64)
        # Process the task (task identifier based on 'A_task')
        tensor[1] = self.param_A.translation_task(row["A"])

        # Process 'A' continuous value
        tensor[0] = self.param_A.translation_continuous(row["A_conc"])

        # Process 'B' continuous value
        tensor[2] = self.param_B.translation_continuous(row["B"])

        # Process 'C' column (discrete)
        tensor[[3, 4]] = self.param_C.translation_discrete(row["C"])[0]

        return tensor

    def setUp(self):
        # Setup parameters for multitask testing:
        self.param_A = ML_parameter(name="A_ml")
        self.param_A.update_values(
            discrete=["A", "B", "C"], min_value=27, max_value=101
        )

        # Setting task as a feature in param_A
        self.param_A.task_feature = True
        self.param_A.task_value = "A"
        self.param_A.generate_translators()

        self.param_B = ML_parameter(
            "B",
            PhysicalParameter(name="B", min_value=27, max_value=101, unit="penguins"),
        )
        self.param_B.update_values(min_value=27, max_value=101)

        self.param_C = ML_parameter(
            "C",
            PhysicalParameter(name="C", unit="penguins", min_value=27, max_value=101),
        )
        self.param_C.update_values(discrete=["A", "B", "C"])

        # Store parameters in a tuple
        self.parameters = (self.param_A, self.param_B, self.param_C)

        # Instantiate the backend
        self.backend = TaskChildClass()

        # Define backend kwargs
        self.kwargs = {
            "Model": "MultiTaskGP",
            "Acquisition Function": "UCB",
            "Initialisation Method": "LHS",
            "Number of initial points": 2,
            "Number of total points": 100,
            "Number of Experiments per batch": 1,
            "Explorative Factor": 1.0,
            "force_categorical": False,
            "Termination criterion": "max_iter",
        }

    def test_run_multitask(self):
        # Start ML prime
        self.backend.ML_prime(
            ML_parameters=[self.param_A, self.param_B, self.param_C],
            targets=["target"],
            save_path="test",
        )

        self.backend.com_prime()

        for key, value in self.kwargs.items():
            self.backend.validate_and_update(key, value)

        first_df = self.backend.first_run()
        first_df["A"] = "A"

        first_df_2 = pd.read_csv("test.csv", index_col=0)
        # first_df_2["A"] = np.random.choice(["B", "C"], size=len(first_df_2))
        first_df_2["vial_idx"] = ""
        # remove the last row:
        first_df_2 = first_df_2[:-1]

        # concatenate the two dataframes
        first_df = pd.concat([first_df_2, first_df])

        # Calculate the output for each row and store it in the target column
        while (
            max(first_df["run_index"])
            <= self.backend.parameters["Number of total points"]
        ):
            for index, row in first_df.iterrows():
                if pd.isna(row["target"]):
                    x = self.translate_multitask(row)
                    first_df.at[index, "target"] = multitask_objective_function(
                        x
                    ).item()
                    first_df.at[index, "status"] = "finished"

            # Push the results back to the backend
            self.backend.in_data(first_df)
            self.backend.update()
            first_df = self.backend.res_df
            print(first_df)

        # Plot results

        # Plot first_df_2 against its index (which represents the line number)
        plt.scatter(
            first_df_2.index,
            first_df_2["target"],
            color="blue",
            label="Initial Low Other Tasks",
        )

        # Remove the first 50 rows from first_df
        first_df_trimmed = first_df.iloc[50:]

        # Reset the index of the trimmed first_df and add 50 to it to continue from the 51st position
        # Adjust the index: if the index is less than 50, add 50, otherwise keep it as is
        adjusted_index = first_df_trimmed.index.map(lambda x: x + 50 if x < 50 else x)

        # Plot the trimmed first_df against the adjusted index
        plt.plot(adjusted_index, first_df_trimmed["target"], c="red", label="Learning")

        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.legend()
        plt.show()


class TestAllTaskChildClass(TestCase):

    def translate_multitask(self, row):
        """
        Translate the recipe from the backend to a tensor to use in the objective function
        for the multitask case.
        """
        tensor = torch.zeros(5, dtype=torch.float64)
        # Assume df has one row, extract the task and continuous values
        tensor[1] = self.param_A.translation_task(row["A"])  # Task is at position 1
        tensor[0] = self.param_A.translation_continuous(
            row["A_conc"]
        )  # Continuous A at position 0
        tensor[2] = self.param_B.translation_continuous(
            row["B"]
        )  # Continuous B at position 2
        tensor[[3, 4]] = self.param_C.translation_discrete(
            row["C"]
        )  # Categorical embedding at positions 3 and 4

        return tensor

    def setUp(self):
        # Setup parameters for multitask testing:
        self.param_A = ML_parameter(name="A_ml")
        self.param_A.update_values(
            discrete=["A", "B", "C"], min_value=27, max_value=101
        )

        # Setting task as a feature in param_A
        self.param_A.task_feature = True
        # self.param_A.task_value = ["A_task"]
        self.param_A.generate_translators()

        self.param_B = ML_parameter(
            "B",
            PhysicalParameter(name="B", min_value=27, max_value=101, unit="penguins"),
        )
        self.param_B.update_values(min_value=27, max_value=101)

        self.param_C = ML_parameter(
            "C",
            PhysicalParameter(name="C", unit="penguins", min_value=27, max_value=101),
        )
        self.param_C.update_values(discrete=["A", "B", "C"])

        # Store parameters in a tuple
        self.parameters = (self.param_A, self.param_B, self.param_C)

        # Instantiate the backend for AllTaskChildClass
        self.backend = AllTaskChildClass()

        # Define backend kwargs
        self.kwargs = {
            "Model": "MultiTaskGP",
            "Acquisition Function": "UCB",
            "Initialisation Method": "LHS",
            "Number of initial points": 20,
            "Number of total points": 70,
            "Number of Experiments per batch": 1,
            "Explorative Factor": 1.0,
            "force_categorical": False,
            "Termination criterion": "max_iter",
        }

    def test_run_multitask(self):
        # Start ML prime
        self.backend.ML_prime(
            ML_parameters=[self.param_A, self.param_B, self.param_C],
            targets=["target"],
            save_path="test",
        )

        self.backend.com_prime()

        for key, value in self.kwargs.items():
            self.backend.validate_and_update(key, value)

        first_df = self.backend.first_run()

        # Calculate the output for each row and store it in the target column
        while (
            max(first_df["run_index"])
            <= self.backend.parameters["Number of total points"]
        ):
            for index, row in first_df.iterrows():
                if pd.isna(row["target"]):
                    x = self.translate_multitask(row)
                    first_df.at[index, "target"] = multitask_objective_function(
                        x
                    ).item()
                    first_df.at[index, "status"] = "finished"

            # Push the results back to the backend
            self.backend.in_data(first_df)
            self.backend.update()
            first_df = self.backend.res_df
            print(first_df)

            # Plot results: group by task and plot a different line for each task
        tasks = first_df["A"].unique()
        colors = ["blue", "green", "red"]  # Define a color for each task

        plt.figure(figsize=(10, 6))

        for i, task in enumerate(tasks):
            task_df = first_df[first_df["A"] == task]
            plt.scatter(
                task_df["run_index"],
                task_df["target"],
                s=5,
                color=colors[i],
                label=f"Task {task}",
            )

        # Add labels and legend
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function Value")
        plt.title("Multitask GP Function - Different Tasks")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    max_v, best_v = maximize_over_combinations(single_objective_function)
    print(max_v, best_v)
