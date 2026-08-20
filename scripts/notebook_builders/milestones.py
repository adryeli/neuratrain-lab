"""Playful bilingual progress captions ("seeds") sprinkled through a notebook.

Placed by each topic module at roughly the 1/3 and 2/3 progress marks of
that notebook, snapped to the nearest natural section boundary rather than
a mechanical cell count — interrupting mid-explanation would hurt more
than it helps. The goal is purely emotional pacing: make a dense topic
feel lighter through a small, friendly interruption.
"""

from __future__ import annotations

from .common import markdown

MESSAGES = {
    "es": [
        "🌱 Acabas de sembrar la primera semilla de tu red neuronal.",
        "🔥 Mitad del camino: de aquí a nada desentrañarás los misterios de la mente artificial.",
        "🚀 La red está tomando forma bajo tus manos.",
        "🏁 Última recta antes del final del notebook.",
    ],
    "en": [
        "🌱 You just planted the first seed of your neural network.",
        "🔥 Halfway there: from here it's a short hop to unravelling the mysteries of the artificial mind.",
        "🚀 The network is taking shape under your hands.",
        "🏁 Final stretch before the end of this notebook.",
    ],
}


def milestone(lang: str, index: int) -> dict:
    """A centered, italic progress caption. ``index`` picks the message (0-3)."""

    text = MESSAGES[lang][index]
    html = (
        '<div style="text-align:center; opacity:.85; font-style:italic; '
        f'margin:1.1rem 0; font-size:1.05rem;">{text}</div>'
    )
    return markdown(html)
