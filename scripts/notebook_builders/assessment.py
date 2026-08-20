"""End-of-notebook self-assessment: multiple-choice quiz + open questions.

Distinct from ``exercises.py`` (mid-notebook fill-in-the-blank code) — this
is the closing "Autoevaluación"/"Self-assessment" section every notebook
ends with, right before the final ``celebrate()`` cell.
"""

from __future__ import annotations

from textwrap import dedent

from .common import markdown

_SECTION_TITLE = {"es": "## 🎯 Autoevaluación", "en": "## 🎯 Self-assessment"}
_SECTION_INTRO = {
    "es": "Respóndelas sin mirar atrás. No necesitas frases perfectas: explica el mecanismo con tus palabras.",
    "en": "Answer without looking back. You don't need perfect phrasing: explain the mechanism in your own words.",
}
_ANSWER_LABEL = {"es": "Ver respuesta", "en": "Show answer"}
_GUIDANCE_LABEL = {
    "es": "Qué debería incluir una buena respuesta",
    "en": "What a good answer should include",
}
_LETTERS = ["A", "B", "C", "D", "E"]


def section_header(lang: str) -> dict:
    return markdown(f"{_SECTION_TITLE[lang]}\n\n{_SECTION_INTRO[lang]}")


def quiz_question(
    lang: str,
    number: int,
    question: str,
    options: list[str],
    correct_index: int,
    explanation: str,
) -> dict:
    """A single multiple-choice question with the answer hidden in <details>.

    ``options`` is a list of 2-5 short answer strings; ``correct_index`` is
    the 0-based index of the correct one.
    """

    option_lines = "\n".join(
        f"{_LETTERS[i]}. {opt}" for i, opt in enumerate(options)
    )
    correct_letter = _LETTERS[correct_index]
    body = (
        f"**{number}. {dedent(question).strip()}**\n\n"
        f"{option_lines}\n\n"
        f"<details>\n<summary><b>{_ANSWER_LABEL[lang]}</b></summary>\n\n"
        f"**{correct_letter}.** {dedent(explanation).strip()}\n\n</details>"
    )
    cell = markdown(body)
    cell["metadata"]["assessment_kind"] = "quiz"
    return cell


def open_question(lang: str, number: int, question: str, guidance: list[str]) -> dict:
    """An open-ended question answered in the student's own words.

    ``guidance`` is a short bullet list of what a strong answer would
    mention — not a single fixed correct answer, since these have no one
    right phrasing.
    """

    bullets = "\n".join(f"- {point}" for point in guidance)
    body = (
        f"**{number}. {dedent(question).strip()}**\n\n"
        f"<details>\n<summary><b>{_GUIDANCE_LABEL[lang]}</b></summary>\n\n"
        f"{bullets}\n\n</details>"
    )
    cell = markdown(body)
    cell["metadata"]["assessment_kind"] = "open"
    return cell
