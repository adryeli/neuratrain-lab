"""Typed configuration for training experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Controllable hyperparameters for the dense model."""

    hidden_units: tuple[int, ...] = (32, 16)
    dropout_rate: float = 0.30
    learning_rate: float = 1e-3
    epochs: int = 150
    batch_size: int = 32
    patience: int = 12
    use_early_stopping: bool = True
    random_state: int = 42

    def validate(self) -> None:
        if not self.hidden_units or any(units <= 0 for units in self.hidden_units):
            raise ValueError("hidden_units must contain positive integers.")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout_rate must be between 0 (inclusive) and 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.epochs <= 0 or self.batch_size <= 0 or self.patience < 0:
            raise ValueError("epochs/batch_size must be positive and patience non-negative.")
