"""Lightweight Multi-Layer Perceptron (MLP) with weight, gradient, and delta tracking."""

import math
import random
from typing import Dict, List, Sequence, Tuple


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _activate(z: float, kind: str) -> float:
    if kind == "tanh":
        return math.tanh(z)
    if kind == "relu":
        return z if z > 0.0 else 0.0
    if kind == "sigmoid":
        return _sigmoid(z)
    if kind == "linear":
        return z
    raise ValueError(f"Unsupported activation: {kind}")


def _activate_deriv_from_out(a: float, z: float, kind: str) -> float:
    if kind == "tanh":
        return 1.0 - a * a
    if kind == "relu":
        return 1.0 if z > 0.0 else 0.0
    if kind == "sigmoid":
        return a * (1.0 - a)
    if kind == "linear":
        return 1.0
    raise ValueError(f"Unsupported activation: {kind}")


class Layer:
    """Dense neural network layer with gradient and update-delta tracking."""

    def __init__(self, in_features: int, out_features: int, activation: str, seed: int = 0) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        rng = random.Random(seed)
        if activation == "relu":
            scale = math.sqrt(2.0 / max(1, in_features))
        else:
            scale = math.sqrt(2.0 / max(1, in_features + out_features))

        self.weights: List[List[float]] = [
            [rng.gauss(0.0, scale) for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.biases: List[float] = [0.0 for _ in range(out_features)]

        self.grad_w: List[List[float]] = [[0.0] * in_features for _ in range(out_features)]
        self.grad_b: List[float] = [0.0] * out_features

        self.last_delta_w: List[List[float]] = [[0.0] * in_features for _ in range(out_features)]
        self.last_delta_b: List[float] = [0.0] * out_features
        self.mean_activations: List[float] = [0.0] * out_features

        # Adam state
        self._m_w: List[List[float]] = [[0.0] * in_features for _ in range(out_features)]
        self._v_w: List[List[float]] = [[0.0] * in_features for _ in range(out_features)]
        self._m_b: List[float] = [0.0] * out_features
        self._v_b: List[float] = [0.0] * out_features

        # Cached single-sample forward pass values
        self._last_inputs: List[float] = []
        self._last_z: List[float] = []
        self._last_a: List[float] = []

    def forward(self, inputs: Sequence[float]) -> List[float]:
        self._last_inputs = list(inputs)
        z_vec: List[float] = []
        a_vec: List[float] = []
        for j in range(self.out_features):
            w_row = self.weights[j]
            z = self.biases[j]
            for i in range(self.in_features):
                z += w_row[i] * inputs[i]
            a = _activate(z, self.activation)
            z_vec.append(z)
            a_vec.append(a)
        self._last_z = z_vec
        self._last_a = a_vec
        return a_vec

    def zero_grad(self) -> None:
        for j in range(self.out_features):
            self.grad_b[j] = 0.0
            row = self.grad_w[j]
            for i in range(self.in_features):
                row[i] = 0.0

    def backward(self, d_out: Sequence[float], is_output_bce_sigmoid: bool = False) -> List[float]:
        """Accumulate gradients from upstream dL/da (or dL/dz if output BCE+sigmoid)."""
        d_inputs = [0.0] * self.in_features
        for j in range(self.out_features):
            if is_output_bce_sigmoid:
                dz = d_out[j]
            else:
                dz = d_out[j] * _activate_deriv_from_out(
                    self._last_a[j], self._last_z[j], self.activation
                )
            self.grad_b[j] += dz
            w_row = self.weights[j]
            gw_row = self.grad_w[j]
            for i in range(self.in_features):
                gw_row[i] += dz * self._last_inputs[i]
                d_inputs[i] += dz * w_row[i]
        return d_inputs

    def step(self, lr: float, batch_size: int, optimizer: str = "adam", step_num: int = 1) -> None:
        inv_n = 1.0 / max(1, batch_size)
        if optimizer == "sgd":
            for j in range(self.out_features):
                gb = self.grad_b[j] * inv_n
                self.grad_b[j] = gb
                db = -lr * gb
                self.biases[j] += db
                self.last_delta_b[j] = db
                for i in range(self.in_features):
                    gw = self.grad_w[j][i] * inv_n
                    self.grad_w[j][i] = gw
                    dw = -lr * gw
                    self.weights[j][i] += dw
                    self.last_delta_w[j][i] = dw
            return

        if optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            b1_corr = 1.0 - (beta1 ** step_num)
            b2_corr = 1.0 - (beta2 ** step_num)
            for j in range(self.out_features):
                gb = self.grad_b[j] * inv_n
                self.grad_b[j] = gb
                self._m_b[j] = beta1 * self._m_b[j] + (1.0 - beta1) * gb
                self._v_b[j] = beta2 * self._v_b[j] + (1.0 - beta2) * (gb * gb)
                m_hat = self._m_b[j] / b1_corr
                v_hat = self._v_b[j] / b2_corr
                db = -lr * m_hat / (math.sqrt(v_hat) + eps)
                self.biases[j] += db
                self.last_delta_b[j] = db

                for i in range(self.in_features):
                    gw = self.grad_w[j][i] * inv_n
                    self.grad_w[j][i] = gw
                    self._m_w[j][i] = beta1 * self._m_w[j][i] + (1.0 - beta1) * gw
                    self._v_w[j][i] = beta2 * self._v_w[j][i] + (1.0 - beta2) * (gw * gw)
                    mw_hat = self._m_w[j][i] / b1_corr
                    vw_hat = self._v_w[j][i] / b2_corr
                    dw = -lr * mw_hat / (math.sqrt(vw_hat) + eps)
                    self.weights[j][i] += dw
                    self.last_delta_w[j][i] = dw
            return

        raise ValueError(f"Unsupported optimizer: {optimizer}")


class MLP:
    """Multi-Layer Perceptron for 2D binary classification with full state inspection."""

    def __init__(
        self,
        hidden_layers: Sequence[int] = (4, 4),
        activation: str = "tanh",
        seed: int = 42,
    ) -> None:
        self.hidden_layers = list(hidden_layers)
        self.activation = activation
        self.seed = seed
        self.epoch = 0
        self.step_num = 0
        self.loss_history: List[float] = []
        self.acc_history: List[float] = []

        sizes = [2] + self.hidden_layers + [1]
        self.layer_sizes = sizes
        self.layers: List[Layer] = []
        for idx in range(len(sizes) - 1):
            is_last = idx == len(sizes) - 2
            act = "sigmoid" if is_last else activation
            self.layers.append(Layer(sizes[idx], sizes[idx + 1], act, seed=seed + idx * 101))

    def forward(self, x: Sequence[float]) -> float:
        out = list(x)
        for layer in self.layers:
            out = layer.forward(out)
        return out[0]

    def predict(self, x: Sequence[float]) -> float:
        return self.forward(x)

    def compute_loss_and_grads(self, dataset: Sequence[Tuple[List[float], int]]) -> Tuple[float, float]:
        """Run forward + backward over dataset without stepping optimizer."""
        for layer in self.layers:
            layer.zero_grad()

        act_sums = [[0.0] * layer.out_features for layer in self.layers]
        total_loss = 0.0
        correct = 0
        eps = 1e-7

        for x, y in dataset:
            p = self.forward(x)
            p_clamped = min(max(p, eps), 1.0 - eps)
            loss = -(y * math.log(p_clamped) + (1.0 - y) * math.log(1.0 - p_clamped))
            total_loss += loss
            if (p >= 0.5 and y == 1) or (p < 0.5 and y == 0):
                correct += 1

            for l_idx, layer in enumerate(self.layers):
                for j, a_val in enumerate(layer._last_a):
                    act_sums[l_idx][j] += a_val

            # dL/dz for BCE + sigmoid output is simply (p - y)
            grad: List[float] = [p - float(y)]
            for l_idx in range(len(self.layers) - 1, -1, -1):
                is_out = l_idx == len(self.layers) - 1
                grad = self.layers[l_idx].backward(grad, is_output_bce_sigmoid=is_out)

        n = max(1, len(dataset))
        for l_idx, layer in enumerate(self.layers):
            layer.mean_activations = [s / n for s in act_sums[l_idx]]

        return total_loss / n, correct / n

    def train_epoch(
        self,
        dataset: Sequence[Tuple[List[float], int]],
        lr: float = 0.05,
        optimizer: str = "adam",
    ) -> Dict[str, float]:
        loss, acc = self.compute_loss_and_grads(dataset)
        self.step_num += 1
        self.epoch += 1
        for layer in self.layers:
            layer.step(lr=lr, batch_size=len(dataset), optimizer=optimizer, step_num=self.step_num)

        self.loss_history.append(loss)
        self.acc_history.append(acc)
        if len(self.loss_history) > 500:
            self.loss_history = self.loss_history[-500:]
            self.acc_history = self.acc_history[-500:]

        return {"epoch": self.epoch, "loss": loss, "accuracy": acc}

    def snapshot(self) -> Dict[str, object]:
        return {
            "epoch": self.epoch,
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "loss": self.loss_history[-1] if self.loss_history else None,
            "accuracy": self.acc_history[-1] if self.acc_history else None,
            "loss_history": self.loss_history[-120:],
            "acc_history": self.acc_history[-120:],
            "layers": [
                {
                    "index": idx,
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "activation": layer.activation,
                    "weights": [[round(w, 6) for w in row] for row in layer.weights],
                    "biases": [round(b, 6) for b in layer.biases],
                    "grad_w": [[round(g, 6) for g in row] for row in layer.grad_w],
                    "grad_b": [round(g, 6) for g in layer.grad_b],
                    "last_delta_w": [[round(d, 6) for d in row] for row in layer.last_delta_w],
                    "last_delta_b": [round(d, 6) for d in layer.last_delta_b],
                    "mean_activations": [round(a, 6) for a in layer.mean_activations],
                }
                for idx, layer in enumerate(self.layers)
            ],
        }
