"""NeuroTrain Lab — entry point. Wires up the guided journey + the real lab."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app_pages.shared import ensure_progress_state

st.set_page_config(page_title="NeuroTrain Lab", page_icon="🧠", layout="wide")
ensure_progress_state()

pages = st.navigation(
    {
        "NeuroTrain Lab": [
            st.Page("app_pages/home.py", title="Inicio", icon=":material/home:", default=True),
        ],
        "Recorrido guiado": [
            st.Page("app_pages/topic1_perceptron.py", title="1 · Perceptrón", icon=":material/hub:"),
            st.Page("app_pages/topic2_loss.py", title="2 · Pérdida y backprop", icon=":material/trending_down:"),
            st.Page("app_pages/topic3_optimizers.py", title="3 · Optimizadores", icon=":material/explore:"),
            st.Page("app_pages/topic4_training.py", title="4 · Entrenamiento", icon=":material/tune:"),
        ],
        "Material adicional": [
            st.Page("app_pages/material_adicional.py", title="Recursos extra", icon=":material/library_books:"),
        ],
        "Laboratorio": [
            st.Page("app_pages/laboratory.py", title="Modo Experimento", icon=":material/experiment:"),
        ],
    },
    position="sidebar",
)
pages.run()
