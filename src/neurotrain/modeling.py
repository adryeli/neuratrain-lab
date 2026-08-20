"""Baseline model and neural network for NeuroTrain Lab."""

from __future__ import annotations

import importlib.util
import random
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import TrainingConfig
from .data import PreparedData


def tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


def make_progress_callback(on_epoch_end: Callable[[int, int, dict[str, float]], None], total_epochs: int) -> Any:
    """Builds a Keras callback that reports per-epoch progress via a plain function.

    Kept import-lazy like the rest of this module, so importing ``neurotrain.modeling``
    never requires TensorFlow. ``on_epoch_end(epoch, total_epochs, logs)`` fires after
    every epoch; this module has no UI-framework dependency — the caller (e.g. a
    Streamlit page) decides how to render that progress.
    """

    import tensorflow as tf

    class _ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
            on_epoch_end(epoch + 1, total_epochs, logs or {})

    return _ProgressCallback()


def fit_logistic_baseline(data: PreparedData, random_state: int = 42) -> LogisticRegression:
    """Fits a simple baseline on the same scaled data."""

    model = LogisticRegression(max_iter=2_000, random_state=random_state)
    model.fit(data.X_train, data.y_train.astype(int))
    return model


def build_dense_classifier(input_dim: int, config: TrainingConfig) -> Any:
    """Builds and compiles a binary MLP in TensorFlow/Keras."""

    config.validate()
    if not tensorflow_available():
        raise ModuleNotFoundError(
            "TensorFlow is not installed. Run: pip install -r requirements.txt"
        )

    import tensorflow as tf

    random.seed(config.random_state)
    np.random.seed(config.random_state)
    tf.keras.utils.set_random_seed(config.random_state)

    layers: list[Any] = [tf.keras.layers.Input(shape=(input_dim,))]
    for units in config.hidden_units:
        layers.append(tf.keras.layers.Dense(units, activation="relu"))
        if config.dropout_rate > 0:
            layers.append(tf.keras.layers.Dropout(config.dropout_rate))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

    model = tf.keras.Sequential(layers, name="neurotrain_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="sensitivity"),
        ],
    )
    return model


def train_dense_classifier(
    data: PreparedData,
    config: TrainingConfig,
    verbose: int = 0,
    extra_callbacks: list[Any] | None = None,
) -> tuple[Any, dict[str, list[float]]]:
    """Trains the MLP and returns the model plus a serializable history."""

    import tensorflow as tf

    model = build_dense_classifier(data.X_train.shape[1], config)
    callbacks: list[Any] = list(extra_callbacks or [])
    if config.use_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
                verbose=verbose,
            )
        )

    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )
    return model, {key: list(values) for key, values in history.history.items()}

