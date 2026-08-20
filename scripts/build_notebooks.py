"""Generates all 8 NeuroTrain Lab notebooks (4 topics x Spanish/English).

Each notebook is defined as a list of cells assembled by a topic module in
``notebook_builders/``. Regenerate after editing any topic module:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

from notebook_builders.common import write_notebook
from notebook_builders import (
    topic1_perceptron,
    topic2_loss_backprop,
    topic3_optimizers,
    topic4_training_lab,
)

NOTEBOOKS = [
    ("es/01_perceptron.ipynb", topic1_perceptron.build_es_cells),
    ("en/01_perceptron.ipynb", topic1_perceptron.build_en_cells),
    ("es/02_perdida_y_backpropagation.ipynb", topic2_loss_backprop.build_es_cells),
    ("en/02_loss_and_backpropagation.ipynb", topic2_loss_backprop.build_en_cells),
    ("es/03_optimizadores.ipynb", topic3_optimizers.build_es_cells),
    ("en/03_optimizers.ipynb", topic3_optimizers.build_en_cells),
    ("es/04_entrenamiento_y_sobreajuste.ipynb", topic4_training_lab.build_es_cells),
    ("en/04_training_and_overfitting.ipynb", topic4_training_lab.build_en_cells),
]


def main() -> None:
    for relative_path, builder in NOTEBOOKS:
        path = write_notebook(relative_path, builder())
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
