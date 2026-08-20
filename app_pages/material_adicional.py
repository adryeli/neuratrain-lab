"""Material adicional — resumen de los notebooks, notebook resuelto de bonus,
y una vista extra de qué pasa a nivel de batch frente a nivel de epoch.

El recorrido completo es: teoría + vídeos (Recorrido guiado) -> aquí, para ver
el flujo de principio a fin en un ejemplo ya resuelto -> Laboratorio.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_pages.shared import (
    NOTEBOOK_SUMMARIES,
    TOPIC_TITLES,
    inject_base_styles,
    render_bonus_notebook,
)

ROOT = Path(__file__).resolve().parent.parent

inject_base_styles()

st.title("📚 Material adicional")
st.caption(
    "Primero la teoría y los vídeos de cada tema (Recorrido guiado). Aquí tienes un resumen "
    "de cada notebook y dos ejemplos completos ya resueltos para ver el flujo de principio a "
    "fin, antes de entrar al Laboratorio."
)

st.subheader("Qué contiene cada notebook")
for topic_key, title in TOPIC_TITLES.items():
    st.markdown(f"**{title}**")
    st.markdown(NOTEBOOK_SUMMARIES[topic_key])

st.divider()

st.subheader("Notebooks resueltos")
st.caption(
    "Dos ejemplos completos del mismo problema: primero a mano, sin frameworks; luego con "
    "las herramientas que se usan de verdad en la industria."
)

bonus_col1, bonus_col2 = st.columns(2)
with bonus_col1:
    render_bonus_notebook("punto_de_partida")
with bonus_col2:
    render_bonus_notebook("tres_maneras")

st.divider()

st.subheader("Qué pasa a nivel de batch frente a nivel de epoch")
st.caption("Adaptado de la Masterclass. Son dos escalas distintas del mismo bucle de entrenamiento.")

batch_col, epoch_col = st.columns(2)
with batch_col:
    with st.container(border=True):
        st.markdown("**A nivel de BATCH** _(en cada lote de datos)_")
        st.caption("Ocurre miles de veces por época. Puramente matemático y de optimización.")
        st.markdown(
            """
            - **Forward pass** — el modelo procesa un pequeño subconjunto de datos y genera predicciones.
            - **Batch loss** — se mide el error cometido únicamente en los datos de ese lote.
            - **Backward pass / Backpropagation** — se calculan los gradientes para saber cómo ajustar los pesos.
            - **Weight update** — el optimizador (Adam, SGD...) modifica los parámetros inmediatamente.
            - **Métricas de entrenamiento** — se acumula el progreso de loss y accuracy.
            """
        )
with epoch_col:
    with st.container(border=True):
        st.markdown("**A nivel de EPOCH** _(al terminar de ver todos los datos)_")
        st.caption("Ocurre solo una vez al final de cada vuelta completa al dataset.")
        st.markdown(
            """
            - **Evaluación en validación** — el modelo se congela y procesa `val_loss`/`val_accuracy`.
            - **EarlyStopping** — revisa si el error de validación mejoró o si debe detenerse.
            - **Checkpoints** — guarda el modelo si alcanzó su mejor rendimiento histórico.
            - **Learning rate scheduler** — reduce la velocidad de aprendizaje si se ha estancado.
            - **Shuffling** — desordena el training set para que los lotes del siguiente epoch cambien.
            """
        )

st.divider()

st.subheader("Presentación y actividad extra")
masterclass_pdf_path = ROOT / "docs" / "Masterclass_ANN.pdf"
extra_cols = st.columns(2)
with extra_cols[0]:
    if masterclass_pdf_path.exists():
        st.download_button(
            "Descargar la presentación (PDF)",
            data=masterclass_pdf_path.read_bytes(),
            file_name="Masterclass_ANN.pdf",
            mime="application/pdf",
            icon=":material/description:",
            width="stretch",
        )
with extra_cols[1]:
    st.link_button(
        "Abrir TensorFlow Playground",
        "https://playground.tensorflow.org/",
        icon=":material/open_in_new:",
        width="stretch",
    )
