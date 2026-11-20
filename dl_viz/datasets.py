"""Synthetic 2D classification datasets for visualizing neural network training."""

import math
import random
from typing import List, Tuple

Sample = Tuple[List[float], int]


def make_xor(n_samples: int = 120, noise: float = 0.12, seed: int = 42) -> List[Sample]:
    rng = random.Random(seed)
    data: List[Sample] = []
    centers = [(-0.7, -0.7, 0), (0.7, 0.7, 0), (-0.7, 0.7, 1), (0.7, -0.7, 1)]
    for i in range(n_samples):
        cx, cy, label = centers[i % 4]
        x = cx + rng.gauss(0.0, noise)
        y = cy + rng.gauss(0.0, noise)
        data.append(([x, y], label))
    rng.shuffle(data)
    return data


def make_circle(n_samples: int = 160, noise: float = 0.08, seed: int = 42) -> List[Sample]:
    rng = random.Random(seed)
    data: List[Sample] = []
    half = n_samples // 2
    for i in range(n_samples):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        if i < half:
            r = rng.uniform(0.0, 0.45)
            label = 1
        else:
            r = rng.uniform(0.75, 1.15)
            label = 0
        x = r * math.cos(angle) + rng.gauss(0.0, noise)
        y = r * math.sin(angle) + rng.gauss(0.0, noise)
        data.append(([x, y], label))
    rng.shuffle(data)
    return data


def make_moons(n_samples: int = 160, noise: float = 0.10, seed: int = 42) -> List[Sample]:
    rng = random.Random(seed)
    data: List[Sample] = []
    half = n_samples // 2
    for i in range(half):
        t = math.pi * i / max(1, half - 1)
        x = math.cos(t) - 0.5 + rng.gauss(0.0, noise)
        y = math.sin(t) - 0.25 + rng.gauss(0.0, noise)
        data.append(([x, y], 0))
    for i in range(n_samples - half):
        t = math.pi * i / max(1, (n_samples - half) - 1)
        x = 1.0 - math.cos(t) - 0.5 + rng.gauss(0.0, noise)
        y = 0.5 - math.sin(t) - 0.25 + rng.gauss(0.0, noise)
        data.append(([x, y], 1))
    rng.shuffle(data)
    return data


def make_spiral(n_samples: int = 180, noise: float = 0.07, seed: int = 42) -> List[Sample]:
    rng = random.Random(seed)
    data: List[Sample] = []
    half = n_samples // 2
    for label in (0, 1):
        for i in range(half):
            r = (i / max(1, half - 1)) * 1.15 + 0.08
            t = 1.75 * (i / max(1, half - 1)) * 2.0 * math.pi + (label * math.pi)
            x = r * math.sin(t) + rng.gauss(0.0, noise)
            y = r * math.cos(t) + rng.gauss(0.0, noise)
            data.append(([x, y], label))
    rng.shuffle(data)
    return data


DATASETS = {
    "xor": make_xor,
    "circle": make_circle,
    "moons": make_moons,
    "spiral": make_spiral,
}


def get_dataset(name: str, n_samples: int = 160, seed: int = 42) -> List[Sample]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name](n_samples=n_samples, seed=seed)
