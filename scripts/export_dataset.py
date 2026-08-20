"""Exports scikit-learn's copy of the UCI WDBC dataset to a reproducible CSV."""

from pathlib import Path

from sklearn.datasets import load_breast_cancer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "breast_cancer_wisconsin.csv"


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    features.columns = [name.replace(" ", "_") for name in features.columns]
    features.insert(0, "diagnosis", dataset.target.map({0: "M", 1: "B"}))
    features.to_csv(OUTPUT, index=False)
    print(f"Exported {len(features)} rows and {features.shape[1]} columns to {OUTPUT}")


if __name__ == "__main__":
    main()

