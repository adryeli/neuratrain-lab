"""Consistent plots for notebooks and the app, in Spanish or English."""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


COLORS = {"train": "#2563EB", "validation": "#F97316", "accent": "#7C3AED"}

LABELS = {
    "es": {
        "loss": "Pérdida",
        "accuracy": "Exactitud",
        "epoch": "Época",
        "train": "Entrenamiento",
        "validation": "Validación",
        "benign": "Benigno",
        "malignant": "Maligno",
        "confusion_title": "Matriz de confusión",
        "threshold": "umbral",
        "roc_title": "Curva ROC en test",
        "feature_1": "Variable 1",
        "feature_2": "Variable 2",
        "layer_distance": "Distancia desde la salida hacia capas anteriores",
        "gradient_magnitude": "Magnitud relativa del gradiente",
    },
    "en": {
        "loss": "Loss",
        "accuracy": "Accuracy",
        "epoch": "Epoch",
        "train": "Training",
        "validation": "Validation",
        "benign": "Benign",
        "malignant": "Malignant",
        "confusion_title": "Confusion matrix",
        "threshold": "threshold",
        "roc_title": "ROC curve on test",
        "feature_1": "Feature 1",
        "feature_2": "Feature 2",
        "layer_distance": "Distance from output toward earlier layers",
        "gradient_magnitude": "Relative gradient magnitude",
    },
}


def plot_training_history(history: dict[str, list[float]], lang: str = "es"):
    """Draws training/validation loss and accuracy curves."""

    labels = LABELS[lang]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label=labels["train"], color=COLORS["train"])
    axes[0].plot(epochs, history["val_loss"], label=labels["validation"], color=COLORS["validation"])
    axes[0].set(title=labels["loss"], xlabel=labels["epoch"], ylabel="Binary cross-entropy")

    axes[1].plot(epochs, history["accuracy"], label=labels["train"], color=COLORS["train"])
    axes[1].plot(epochs, history["val_accuracy"], label=labels["validation"], color=COLORS["validation"])
    axes[1].set(title=labels["accuracy"], xlabel=labels["epoch"], ylabel=labels["accuracy"], ylim=(0, 1.02))

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    return fig


def plot_training_history_comparison(
    history_a: dict[str, list[float]],
    history_b: dict[str, list[float]],
    label_a: str,
    label_b: str,
    lang: str = "es",
):
    """Overlays two runs' validation loss curves for a guided A/B comparison."""

    labels = LABELS[lang]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    epochs_a = range(1, len(history_a["val_loss"]) + 1)
    epochs_b = range(1, len(history_b["val_loss"]) + 1)
    axis.plot(epochs_a, history_a["val_loss"], label=label_a, color=COLORS["train"])
    axis.plot(epochs_b, history_b["val_loss"], label=label_b, color=COLORS["validation"])
    axis.set(
        title=f"{labels['validation']} {labels['loss'].lower()}: A vs B",
        xlabel=labels["epoch"],
        ylabel="Binary cross-entropy",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    return fig


def plot_confusion(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5, lang: str = "es"):
    labels = LABELS[lang]
    y_pred = (np.asarray(probabilities).ravel() >= threshold).astype(int)
    fig, axis = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=[labels["benign"], labels["malignant"]],
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    axis.set_title(f"{labels['confusion_title']} · {labels['threshold']} {threshold:.2f}")
    fig.tight_layout()
    return fig


def plot_roc(y_true: np.ndarray, probabilities: np.ndarray, lang: str = "es"):
    labels = LABELS[lang]
    fig, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        curve_kwargs={"color": COLORS["accent"]},
        plot_chance_level=True,
        chance_level_kw={"color": "#64748B", "alpha": 0.7, "linestyle": "--"},
        ax=axis,
    )
    axis.set_title(labels["roc_title"])
    axis.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    title: str = "",
    lang: str = "es",
    resolution: int = 300,
    ax=None,
):
    """Plots a 2D decision boundary for a 2-feature dataset (e.g. ``make_moons``).

    ``predict_fn`` maps an ``(N, 2)`` array to predicted probabilities or
    classes in ``[0, 1]``, so it works equally for a hand-written NumPy
    perceptron, a scikit-learn model, or a Keras model's ``.predict``.
    """

    labels = LABELS[lang]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    predictions = np.asarray(predict_fn(grid)).reshape(xx.shape)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.contourf(xx, yy, predictions, levels=20, cmap="RdBu_r", alpha=0.6, vmin=0, vmax=1)
    ax.contour(xx, yy, predictions, levels=[0.5], colors="#172033", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="white", s=35, linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(labels["feature_1"])
    ax.set_ylabel(labels["feature_2"])
    if fig is not None:
        fig.tight_layout()
        return fig
    return ax


def plot_gradient_magnitude_by_layer(magnitudes_by_label: dict[str, list[float]], lang: str = "es"):
    """Plots mean-|gradient| per Dense layer (log scale) for the vanishing-gradient demo.

    ``magnitudes_by_label`` maps a series name (e.g. ``"sigmoid"``/``"relu"``)
    to a list of magnitudes ordered from the layer closest to the output to
    the one closest to the input.
    """

    labels = LABELS[lang]
    palette = [COLORS["accent"], COLORS["validation"], COLORS["train"]]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for index, (series_label, magnitudes) in enumerate(magnitudes_by_label.items()):
        distance = range(1, len(magnitudes) + 1)
        axis.plot(distance, magnitudes, marker="o", label=series_label, color=palette[index % len(palette)])
    axis.set_yscale("log")
    axis.set(xlabel=labels["layer_distance"], ylabel=labels["gradient_magnitude"])
    axis.grid(alpha=0.2, which="both")
    axis.legend()
    fig.tight_layout()
    return fig
