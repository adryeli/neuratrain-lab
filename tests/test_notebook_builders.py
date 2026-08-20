"""Structural sanity checks for the 8 generated notebooks.

Does not execute cells (see scripts/verify_notebooks.py for the real
end-to-end run) — this only checks the generator output shape: syntax,
required closing cells, and ES/EN parity.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import notebook_builders  # noqa: E402


def _discover_topic_modules():
    modules = []
    for info in pkgutil.iter_modules(notebook_builders.__path__):
        if info.name.startswith("topic"):
            modules.append(importlib.import_module(f"notebook_builders.{info.name}"))
    return sorted(modules, key=lambda m: m.__name__)


TOPIC_MODULES = _discover_topic_modules()
LANGS = ["es", "en"]


class NotebookBuilderStructureTests(unittest.TestCase):
    def test_at_least_one_topic_module_is_registered(self):
        self.assertGreaterEqual(len(TOPIC_MODULES), 1)

    def test_every_topic_exposes_both_languages(self):
        for module in TOPIC_MODULES:
            self.assertTrue(hasattr(module, "build_es_cells"), f"{module.__name__} missing build_es_cells")
            self.assertTrue(hasattr(module, "build_en_cells"), f"{module.__name__} missing build_en_cells")

    def test_non_exercise_code_cells_compile(self):
        for module in TOPIC_MODULES:
            for lang in LANGS:
                cells = getattr(module, f"build_{lang}_cells")()
                for index, cell in enumerate(cells):
                    if cell["cell_type"] != "code":
                        continue
                    if cell.get("metadata", {}).get("exercise_stub"):
                        continue
                    source = "".join(cell["source"])
                    try:
                        compile(source, f"<{module.__name__}:{lang}:cell{index}>", "exec")
                    except SyntaxError as exc:
                        self.fail(f"{module.__name__} ({lang}) cell {index} failed to compile: {exc}")

    def test_exercise_stubs_carry_a_working_solution(self):
        for module in TOPIC_MODULES:
            for lang in LANGS:
                cells = getattr(module, f"build_{lang}_cells")()
                for index, cell in enumerate(cells):
                    if not cell.get("metadata", {}).get("exercise_stub"):
                        continue
                    solution = cell["metadata"].get("solution_source")
                    self.assertTrue(solution, f"{module.__name__} ({lang}) cell {index} has no solution_source")
                    try:
                        compile(solution, f"<{module.__name__}:{lang}:cell{index}:solution>", "exec")
                    except SyntaxError as exc:
                        self.fail(f"{module.__name__} ({lang}) cell {index} solution failed to compile: {exc}")

    def test_last_cell_calls_celebrate(self):
        for module in TOPIC_MODULES:
            for lang in LANGS:
                cells = getattr(module, f"build_{lang}_cells")()
                last_source = "".join(cells[-1]["source"])
                self.assertIn("celebrate(", last_source, f"{module.__name__} ({lang}) last cell must call celebrate()")

    def test_assessment_has_quiz_and_exactly_two_open_questions(self):
        for module in TOPIC_MODULES:
            for lang in LANGS:
                cells = getattr(module, f"build_{lang}_cells")()
                kinds = [c["metadata"].get("assessment_kind") for c in cells if c["cell_type"] == "markdown"]
                quiz_count = kinds.count("quiz")
                open_count = kinds.count("open")
                self.assertGreaterEqual(quiz_count, 4, f"{module.__name__} ({lang}) needs >=4 quiz questions, found {quiz_count}")
                self.assertEqual(open_count, 2, f"{module.__name__} ({lang}) needs exactly 2 open questions, found {open_count}")

    def test_es_and_en_editions_have_matching_cell_counts(self):
        for module in TOPIC_MODULES:
            es_cells = module.build_es_cells()
            en_cells = module.build_en_cells()
            self.assertEqual(
                len(es_cells),
                len(en_cells),
                f"{module.__name__}: ES has {len(es_cells)} cells, EN has {len(en_cells)} — should match structurally",
            )


if __name__ == "__main__":
    unittest.main()
