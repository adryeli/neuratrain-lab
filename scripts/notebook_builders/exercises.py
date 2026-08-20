"""The ✏️✏️✏️ fill-in-the-blank exercise pattern used inside every notebook.

Distinct from ``assessment.py``: these are hands-on code exercises spaced
through the middle of a notebook, not the end-of-notebook self-assessment.
"""

from __future__ import annotations

from textwrap import dedent

from .common import exercise_stub, markdown

_HEADER = {"es": "✏️ Ejercicio", "en": "✏️ Exercise"}
_SOLUTION_LABEL = {"es": "Ver solución", "en": "Show solution"}


def exercise_cell(lang: str, prompt: str, starter_code: str, solution_code: str) -> list[dict]:
    """Build the 3-cell exercise block: prompt, ✏️✏️✏️ stub, hidden solution.

    ``starter_code`` must contain the literal marker ``✏️✏️✏️`` where the
    student is expected to write code. Two deliberate placements are
    available to the caller depending on which failure mode is more
    instructive for that particular exercise:

    - Bare-expression position (e.g. ``Dense(✏️✏️✏️, ...)``) fails
      immediately with a clear ``SyntaxError`` if left unedited.
    - Inside a string literal (e.g. ``activation="✏️✏️✏️"``) is valid
      Python but fails later with a framework-level ``ValueError``.
    """

    intro = markdown(f"### {_HEADER[lang]}\n\n{dedent(prompt).strip()}")
    stub = exercise_stub(starter_code)
    # Stashed (not shown to the student) so scripts/verify_notebooks.py can build a
    # "solved" copy of the notebook and confirm the full, correct path executes end-to-end.
    stub["metadata"]["solution_source"] = dedent(solution_code).strip()
    label = _SOLUTION_LABEL[lang]
    solution = markdown(
        f"<details>\n<summary><b>{label}</b></summary>\n\n"
        f"```python\n{dedent(solution_code).strip()}\n```\n\n</details>"
    )
    return [intro, stub, solution]
