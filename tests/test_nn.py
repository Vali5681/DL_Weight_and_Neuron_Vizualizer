"""Unit tests for MLP forward pass, analytical gradients, and XOR convergence."""

import unittest

from dl_viz.datasets import get_dataset
from dl_viz.nn import MLP


class TestMLP(unittest.TestCase):
    def test_gradient_check_tanh(self) -> None:
        self._assert_gradients_match("tanh")

    def test_gradient_check_sigmoid(self) -> None:
        self._assert_gradients_match("sigmoid")

    def _assert_gradients_match(self, activation: str) -> None:
        mlp = MLP(hidden_layers=[3, 2], activation=activation, seed=7)
        dataset = [([0.4, -0.6], 1), ([-0.5, 0.3], 0)]
        mlp.compute_loss_and_grads(dataset)

        eps = 1e-5
        n = len(dataset)

        for l_idx, layer in enumerate(mlp.layers):
            for j in range(layer.out_features):
                # Check bias gradient
                orig_b = layer.biases[j]
                layer.biases[j] = orig_b + eps
                loss_plus, _ = self._eval_loss_only(mlp, dataset)
                layer.biases[j] = orig_b - eps
                loss_minus, _ = self._eval_loss_only(mlp, dataset)
                layer.biases[j] = orig_b

                num_gb = (loss_plus - loss_minus) / (2.0 * eps)
                ana_gb = layer.grad_b[j] / n
                self.assertAlmostEqual(
                    ana_gb,
                    num_gb,
                    places=5,
                    msg=f"Bias grad mismatch at layer {l_idx}, neuron {j}",
                )

                # Check weight gradients
                for i in range(layer.in_features):
                    orig_w = layer.weights[j][i]
                    layer.weights[j][i] = orig_w + eps
                    loss_plus, _ = self._eval_loss_only(mlp, dataset)
                    layer.weights[j][i] = orig_w - eps
                    loss_minus, _ = self._eval_loss_only(mlp, dataset)
                    layer.weights[j][i] = orig_w

                    num_gw = (loss_plus - loss_minus) / (2.0 * eps)
                    ana_gw = layer.grad_w[j][i] / n
                    self.assertAlmostEqual(
                        ana_gw,
                        num_gw,
                        places=5,
                        msg=f"Weight grad mismatch at layer {l_idx}, ({j},{i})",
                    )

    @staticmethod
    def _eval_loss_only(mlp: MLP, dataset):
        import math

        total = 0.0
        eps = 1e-7
        for x, y in dataset:
            p = min(max(mlp.predict(x), eps), 1.0 - eps)
            total += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        return total / len(dataset), 0.0

    def test_xor_convergence(self) -> None:
        dataset = get_dataset("xor", n_samples=80, seed=12)
        mlp = MLP(hidden_layers=[4, 4], activation="tanh", seed=12)
        first = mlp.train_epoch(dataset, lr=0.08, optimizer="adam")
        for _ in range(149):
            last = mlp.train_epoch(dataset, lr=0.08, optimizer="adam")

        self.assertLess(last["loss"], first["loss"])
        self.assertGreaterEqual(last["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
