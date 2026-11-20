# DL Weight & Neuron Visualizer

<img width="1689" height="898" alt="image" src="https://github.com/user-attachments/assets/c6db8101-bf16-4c3d-8fb0-e4842f0540e5" />


A lightweight, zero-dependency Deep Learning project in Python with an interactive web GUI that visualizes **weights, biases, gradients, and live weight updates ($\Delta w$)** during training.

## Features
- **Interactive Neural Network Graph**:
  - **Edges (Weights)**: Colored by sign (cyan = positive, coral = negative), thickness scaled by $|w|$, optional numeric labels, and live amber highlight for $|\Delta w|$ updates.
  - **Neurons**: Fill intensity reflects mean activation; ring color reflects bias sign; displays bias $b$.
  - **Click-to-Inspect**: Click any edge or neuron to inspect its exact value, gradient ($\partial L/\partial w$ or $\partial L/\partial b$), and latest update ($\Delta w$ or $\Delta b$).
- **Live 2D Decision Boundary**: Watch the classification boundary warp in real time as weights update.
- **Training Loss & Accuracy Curve** + **Largest Weight Updates ($|\Delta w|$)** table.
- **Configurable**: Datasets (`XOR`, `Circle`, `Two Moons`, `Spiral`), hidden layer architectures, activations (`tanh`, `relu`, `sigmoid`), optimizers (`Adam`, `SGD`), learning rate, and playback speed.

## Running the App

```bash
python3 main.py
```

Then open `http://127.0.0.1:8765` in your browser (it opens automatically by default; pass `--no-browser` to disable).

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
