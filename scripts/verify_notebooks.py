"""Executes every notebook end-to-end with exercises pre-solved.

Exercise-stub cells (marked with ✏️✏️✏️) are *designed* to fail if run
unedited, so a plain ``nbconvert --execute`` can't verify a whole notebook.
This script builds a temporary "solved" copy — each exercise-stub cell's
source is swapped for the solution stashed in its own metadata by
``notebook_builders/exercises.py`` — and executes that copy, proving the
complete, correct learning path runs top to bottom.

Usage:
    python scripts/verify_notebooks.py                  # all notebooks
    python scripts/verify_notebooks.py notebooks/es/01_perceptron.ipynb
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def solved_copy(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Returns a deep-copied notebook with every exercise stub's source
    replaced by its stashed solution."""

    solved = nbformat.from_dict(notebook)
    for cell in solved.cells:
        if cell.get("cell_type") == "code" and cell.get("metadata", {}).get("exercise_stub"):
            solution = cell["metadata"].get("solution_source")
            if solution is None:
                raise ValueError("exercise_stub cell has no stashed solution_source")
            cell["source"] = solution
    return solved


def verify(path: Path) -> None:
    print(f"Verifying {path.relative_to(ROOT)} ...", end=" ", flush=True)
    original = nbformat.read(path, as_version=4)
    solved = solved_copy(original)

    with tempfile.TemporaryDirectory() as tmp_dir:
        client = NotebookClient(
            solved,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()

    for cell in solved.cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise RuntimeError(f"{output['ename']}: {output['evalue']}")

    print("OK")


def main() -> None:
    args = sys.argv[1:]
    targets = [Path(a).resolve() for a in args] if args else sorted(NOTEBOOK_DIR.rglob("*.ipynb"))

    failures = []
    for path in targets:
        try:
            verify(path)
        except Exception as exc:  # noqa: BLE001 - report every failure, don't stop at the first
            print("FAILED")
            print(f"  {type(exc).__name__}: {exc}")
            failures.append(path)

    if failures:
        print(f"\n{len(failures)} notebook(s) failed verification.")
        sys.exit(1)
    print(f"\nAll {len(targets)} notebook(s) verified.")


if __name__ == "__main__":
    main()
