"""
Author: Elia Savino
gitHub: github.com/EliaSavino

Happy Hacking!

Descr:
test case for the embedding methodology for high categorical screening
"""

import unittest
import torch
from robrains.custom_models import EmbeddingModel
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class TestEmbeddingModel(unittest.TestCase):
    def setUp(self):
        """Set up a small synthetic dataset for testing."""
        self.train_X = torch.randint(
            0, 5, (100, 3)
        )  # 100 samples, 3 categorical features
        self.train_Y = torch.rand(100, 1)  # Continuous target for regression
        self.bounds = [4, 4, 4]  # 5 categories per feature
        self.model = EmbeddingModel(
            train_X=self.train_X,
            train_Y=self.train_Y,
            bounds=self.bounds,
            task="regression",
            num_epochs=10,  # Short training for testing
            learning_rate=0.01,
        )

    def test_generate_hidden_dims(self):
        """Test the hidden dimension generator."""
        hidden_dims = self.model._generate_hidden_dims(
            input_dim=6, output_dim=1, hidden_layers=0
        )
        print(hidden_dims)
        self.assertTrue(
            all(x > 1 for x in hidden_dims),
            "Hidden dims should be greater than output_dim.",
        )
        # self.assertTrue(len(hidden_dims) > 0, "Should generate at least one hidden layer.")

    def test_train_reduces_loss(self):
        """Test that training reduces the loss."""
        initial_loss = None
        final_loss = None
        for epoch in range(2):
            epoch_loss = 0.0
            for batch_X, batch_Y in self.model.loader:
                batch_X = batch_X.to(self.model.device).long()
                batch_Y = batch_Y.to(self.model.device).float()

                self.model.optimizer.zero_grad()
                outputs = self.model.model(batch_X)
                loss = self.model.criterion(outputs, batch_Y)
                loss.backward()
                self.model.optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)

            epoch_loss /= len(self.model.loader.dataset)
            if epoch == 0:
                initial_loss = epoch_loss
            else:
                final_loss = epoch_loss

        self.assertIsNotNone(initial_loss, "Initial loss should be set.")
        self.assertIsNotNone(final_loss, "Final loss should be set.")
        self.assertLess(final_loss, initial_loss, "Training should reduce the loss.")

    def test_predict(self):
        """Test the predict function."""
        test_X = torch.tensor([[1, 2, 3]])
        prediction = self.model.predict(test_X)
        self.assertEqual(prediction.shape, (1, 1), "Prediction shape mismatch.")

    def test_predict_all(self):
        """Test the predict_all function."""
        results = self.model.predict_all(top_k=5, middle_k=5)
        self.assertIn("top", results, "Result should contain 'top'.")
        self.assertIn("middle", results, "Result should contain 'middle'.")
        self.assertEqual(results["top"].shape[0], 5, "Top performers count mismatch.")
        self.assertEqual(
            results["middle"].shape[0], 5, "Middle performers count mismatch."
        )

    def test_get_embeddings(self):
        """Test that embeddings are correctly retrieved."""
        embeddings = self.model.get_embeddings()
        self.assertEqual(
            len(embeddings),
            len(self.bounds),
            "Embedding count should match the number of bounds.",
        )
        for emb in embeddings:
            self.assertEqual(
                emb.shape[0],
                max(self.bounds) + 1,
                "Embedding size mismatch for categorical variables.",
            )


class TestEmbeddingModelWorkflow(unittest.TestCase):
    def test_nothing(self):
        pass

    def test_full_workflow(self):
        """Simulate the full optimization workflow for a fully categorical space."""
        num_categories = 5
        num_initial_points = 20
        num_samples_per_step = 20
        bounds = [
            np.random.randint(2, 10) for _ in range(num_categories)
        ]  # Random levels between 2 and 10
        space_size = np.prod(bounds)

        # Define a simulated metric function
        def simulate_metric(x):
            """Simulate a metric function for the categorical space."""
            return (
                torch.sum(x.float(), dim=1) + torch.randn(x.size(0)) * 0.1
            )  # Sum + small noise

        def simulate_complex_metric(x: torch.Tensor) -> torch.Tensor:
            """
            Simulate a more complex metric function for the categorical space.

            Args:
                x (torch.Tensor): Input tensor of categorical variables.

            Returns:
                torch.Tensor: Simulated metric values.
            """
            # Convert x to float for calculations
            x_float = x.float()

            # Maximization terms: Favor higher values in first and second columns
            max_term = torch.sum(x_float[:, :2] ** 2, dim=1)  # Quadratic scaling

            # Minimization terms: Favor lower values in third column
            min_term = -torch.sum(x_float[:, 2:3] * 2, dim=1)  # Linear penalty

            # Random penalties for specific combinations
            penalty = torch.zeros_like(max_term)
            for idx in range(x.size(1)):
                penalty -= torch.where(
                    x[:, idx] % 3 == 0, torch.rand_like(max_term) * 5, 0
                )

            # Interaction terms: Nonlinear dependency between columns
            interaction_term = torch.prod(
                torch.sin(x_float + 1.0), dim=1
            )  # Sine-based interaction

            # Add noise for realism
            noise = torch.randn(x.size(0)) * 0.1

            # Combine terms to create the final metric
            return max_term + min_term + interaction_term + penalty + noise

        # find the max score for the metric:
        max_score = 0
        for x in product(*[range(b + 1) for b in bounds]):
            score = simulate_complex_metric(torch.tensor(x).unsqueeze(0)).item()
            if score > max_score:
                max_score = score
        print(f"Max score: {max_score:.3f}")

        # Generate initial points
        train_X = torch.cat(
            [torch.randint(0, b + 1, (num_initial_points, 1)) for b in bounds], dim=1
        )
        train_Y = simulate_complex_metric(train_X).unsqueeze(1)

        # Initialize the model
        model = EmbeddingModel(
            train_X=train_X,
            train_Y=train_Y,
            bounds=bounds,
            task="regression",
            num_epochs=50,
            learning_rate=0.01,
            batch_size=5,
            hidden_layers=0,
        )

        top_scores = []
        middle_scores = []
        best_score = []
        train_size = []
        for iteration in range(30):  # Simulate 5 optimization steps
            # Predict all possible combinations
            num_samples = num_samples_per_step // 2
            predictions = model.predict_all(top_k=num_samples, middle_k=num_samples)

            # Select new points
            exploitative_points = predictions["top"]
            explorative_points = predictions["middle"]

            # Calculate metrics for new points
            new_points = torch.cat([exploitative_points, explorative_points], dim=0)
            new_Y = simulate_complex_metric(new_points).unsqueeze(1)

            # Add new points to the dataset
            train_X = torch.cat([train_X, new_points], dim=0)
            train_Y = torch.cat([train_Y, new_Y], dim=0)

            # Retrain the model
            model = EmbeddingModel(
                train_X=train_X,
                train_Y=train_Y,
                bounds=bounds,
                task="regression",
                num_epochs=50,
                learning_rate=0.01,
                hidden_layers=0,
            )

            # Check if all exploitative points rank high
            top_predictions = model.predict(exploitative_points)
            middle_predictions = model.predict(explorative_points)
            avg_top_score = top_predictions.mean().item()
            avg_middle_score = middle_predictions.mean().item()
            print(
                f"Iteration {iteration + 1}: Avg top exploitative score = {avg_top_score:.3f}"
            )
            top_scores.append(avg_top_score)
            best_score.append(top_predictions.max().item())
            middle_scores.append(avg_middle_score)
            train_size.append(len(train_X))
            # Convergence condition (optional): Stop if exploitative points rank high enough

        print(top_scores)

        plt.hlines(
            max_score,
            0,
            max(train_size),
            label="Max Score",
            color="red",
            linestyle="--",
        )
        plt.plot(train_size, top_scores, label="Top Exploitative Points")
        plt.plot(train_size, middle_scores, label="Middle Explorative Points")
        plt.plot(train_size, best_score, label="Best Score")
        plt.xlabel("Dataset Size")
        plt.ylabel("Avg Top Score (arbitrary units)")
        plt.title(
            f"Convergence of exploitative points with complex metric, size of space({space_size})"
        )
        plt.legend()
        plt.show()

        # Get embeddings after optimization
        embeddings = model.get_embeddings()
        self.assertEqual(
            len(embeddings),
            len(bounds),
            "Embedding count should match number of categorical variables.",
        )

        # Final assertions for exploitative points
        self.assertTrue(
            avg_top_score > 8.0, "Final exploitative points should rank high."
        )


if __name__ == "__main__":
    unittest.main()
