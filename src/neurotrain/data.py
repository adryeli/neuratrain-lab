"""Loading, validation, splitting, and scaling for the dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
TARGET_COLUMN = "diagnosis"


@dataclass
class PreparedData:
    """Leak-free splits plus the fitted preprocessing objects."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_train_raw: pd.DataFrame
    X_val_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    scaler: StandardScaler
    feature_names: list[str]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Loads the bundled CSV and checks its minimal contract."""

    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    frame = pd.read_csv(csv_path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}'.")
    if frame[TARGET_COLUMN].isna().any() or frame.drop(columns=TARGET_COLUMN).isna().any().any():
        raise ValueError("The dataset contains unexpected missing values.")
    if set(frame[TARGET_COLUMN].unique()) != {"B", "M"}:
        raise ValueError("diagnosis must contain exactly the labels B and M.")
    if frame.shape[1] != 31:
        raise ValueError("Expected 30 predictor variables plus one label.")

    return frame


def prepare_data(frame: pd.DataFrame, random_state: int = 42) -> PreparedData:
    """Builds train/validation/test splits and fits the scaler on train only.

    The positive class is explicitly defined as malignant (M = 1).
    """

    X = frame.drop(columns=TARGET_COLUMN)
    y = frame[TARGET_COLUMN].eq("M").astype("int8")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return PreparedData(
        X_train=X_train_scaled.astype("float32"),
        X_val=X_val_scaled.astype("float32"),
        X_test=X_test_scaled.astype("float32"),
        y_train=y_train.to_numpy(dtype="float32"),
        y_val=y_val.to_numpy(dtype="float32"),
        y_test=y_test.to_numpy(dtype="float32"),
        X_train_raw=X_train.copy(),
        X_val_raw=X_val.copy(),
        X_test_raw=X_test.copy(),
        scaler=scaler,
        feature_names=X.columns.tolist(),
    )
