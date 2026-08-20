"""Inicio — journey overview and progress checklist."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_pages.shared import (
    TOPIC_TITLES,
    ensure_progress_state,
    inject_base_styles,
    progress_fraction,
    render_dataset_download,
    render_hero,
)

ROOT = Path(__file__).resolve().parent.parent

inject_base_styles()
ensure_progress_state()

render_hero(
    "NeuroTrain Lab",
    "Un recorrido guiado para entender, desde cero, cómo aprende una red neuronal — "
    "con analogías, un notebook por tema y un laboratorio real al final donde entrenas "
    "un modelo de verdad sobre datos clínicos.",
)

st.subheader("Cómo funciona el recorrido")
st.markdown(
    """
    1. **Recorrido guiado** — 4 temas en la barra lateral, cada uno con teoría contada con
       analogías del día a día, un vídeo, una mini-demo interactiva, y un botón para
       descargar el notebook de ese tema (en español o en inglés) y hacer los ejercicios reales.
    2. **Material adicional** — Notebook resuelto ya resuelto uno de cómo se haría manualmente y 
        y otro con librerias, que es la manera más usada hoy en día.
        RECOMENDACIÓN: leer antes de hacer los ejercicios del notebook y acceder al laboratorio
        y al terminar los ejercicios volver a leer para terminar de comprender.
    3. **Laboratorio · Modo Experimento** — cuando termines (o cuando quieras, no hay
       bloqueos), entra al laboratorio: es una app de verdad donde entrenas una red
       neuronal sobre un dataset clínico real y comparas sus resultados con un modelo
       más simple.
    """
)
st.caption(
    "No hay nada bloqueado — puedes saltar directamente al laboratorio si quieres. "
    "El recorrido existe para que, cuando llegues ahí, entiendas exactamente qué está pasando."
)
st.subheader("Antes de empezar")
st.markdown(
    """
    1.Lee la teoría de cada tema en la barra lateral y mira los vídeos. 
        Luego, **haz los ejercicios** en el notebook de cada tema.
        Contienen ejercicios marcados con ✏️✏️✏️. Descárgalos en español o en
        inglés desde cada página de tema.
    2. **Ábrelos en Google Colab** si no tienes Jupyter instalado localmente: entra en
       [colab.research.google.com](https://colab.research.google.com), *Archivo → Subir
       cuaderno* y selecciona el `.ipynb` que descargaste.
       - Los Notebooks 1, 2 y 3 son autocontenidos: no necesitan ningún archivo extra.
       - El Notebook 4 necesita el dataset real. Descárgalo aquí abajo y súbelo a Colab
         con el panel de archivos (icono de carpeta a la izquierda) antes de ejecutar las
         celdas que lo cargan.
    3. Pasa por **Material adicional** (barra lateral) para ver 2 ejemplos de notebooks resueltos.
        y termina en el **Laboratorio**.
    """
)
render_dataset_download()

st.divider()

st.subheader("Tu progreso")
progress = st.session_state["progress"]
st.progress(progress_fraction())

checklist_cols = st.columns(4)
for column, topic_key in zip(checklist_cols, TOPIC_TITLES):
    with column:
        done = progress[topic_key]
        icon = ":material/check_circle:" if done else ":material/radio_button_unchecked:"
        st.markdown(f"{icon} {TOPIC_TITLES[topic_key]}")

st.divider()


pdf_guide_path = ROOT / "docs" / "GUIA_DEL_PROYECTO.pdf"
if pdf_guide_path.exists():
    st.download_button(
        "Descargar guía del proyecto (PDF)",
        data=pdf_guide_path.read_bytes(),
        file_name="GUIA_DEL_PROYECTO.pdf",
        mime="application/pdf",
        icon=":material/description:",
    )

st.divider()
st.caption("¿Dudas? Contacta con Elizabeth Sena en LinkedIn.")
st.link_button(
    "LinkedIn — Elizabeth Sena",
    "https://www.linkedin.com/in/elizabeth-sena",
    icon=":material/person:",
)
