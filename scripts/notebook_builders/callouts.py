"""Boxed callouts shared by all notebooks (concept / doubt / mistake / remember).

Styled with the same purple/blue/orange/green palette as the Streamlit app
(see ``app.py``'s ``.hero``/``.note`` CSS) so notebooks and app read as one
visual system. Raw HTML-in-markdown is already an established pattern in
this codebase (the original notebook's ``<details>`` blocks), so this is
not a new technique for the project.
"""

from __future__ import annotations

from textwrap import dedent

from .common import markdown

CALLOUTS = {
    "concept": {
        "icon": "🧠",
        "label": {"es": "CONCEPTO CLAVE", "en": "KEY CONCEPT"},
        "border": "#7C3AED",
        "background": "#F5F3FF",
    },
    "doubt": {
        "icon": "❓",
        "label": {"es": "DUDA PROBABLE", "en": "LIKELY QUESTION"},
        "border": "#2563EB",
        "background": "#EFF6FF",
    },
    "mistake": {
        "icon": "⚠️",
        "label": {"es": "ERROR TÍPICO", "en": "TYPICAL MISTAKE"},
        "border": "#F97316",
        "background": "#FFF7ED",
    },
    "remember": {
        "icon": "📌",
        "label": {"es": "PARA RECORDAR", "en": "REMEMBER THIS"},
        "border": "#22C55E",
        "background": "#F0FDF4",
    },
}


def callout(kind: str, lang: str, title: str, body: str) -> dict:
    """A single boxed callout markdown cell.

    ``kind`` is one of ``concept``/``doubt``/``mistake``/``remember``.
    ``title`` is the short headline inside the box; ``body`` is the
    explanation (markdown allowed).
    """

    style = CALLOUTS[kind]
    html = (
        f'<div style="border-left:4px solid {style["border"]}; '
        f'background:{style["background"]}; border-radius:.4rem; '
        f'padding:.85rem 1.1rem; margin:.7rem 0;">\n'
        f'<b>{style["icon"]} {style["label"][lang]} — {title}</b><br><br>\n'
        f"{dedent(body).strip()}\n"
        f"</div>"
    )
    return markdown(html)
