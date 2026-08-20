"""Low-level cell/notebook builders shared by every topic module.

Moved verbatim (behavior unchanged) from the old single-file
``scripts/build_notebooks.py`` so all 8 notebooks share one implementation.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = ROOT / "notebooks"


def _cell_id() -> str:
    # nbformat 4.5+ expects every cell to carry a unique id.
    return uuid.uuid4().hex[:8]


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def exercise_stub(source: str) -> dict:
    """Code cell containing an unfinished ✏️✏️✏️ exercise.

    Tagged so ``tests/test_notebook_builders.py`` can skip ``compile()``
    checks on it (the whole point is that it does not run as-is).
    """

    cell = code(source)
    cell["metadata"]["exercise_stub"] = True
    return cell


def notebook(cells: list[dict], language_name: str = "Python 3 (.venv)") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": language_name,
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(relative_path: str, cells: list[dict]) -> Path:
    """Write a notebook under ``notebooks/<relative_path>``, creating parents."""

    path = NOTEBOOK_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    return path
