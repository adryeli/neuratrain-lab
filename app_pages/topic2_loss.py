"""Tema 2 · Pérdida y Backpropagation — theory, video, and a live gradient demo."""

from __future__ import annotations

import streamlit as st

from app_pages.shared import (
    inject_base_styles,
    render_completion_button,
    render_notebook_downloads,
    render_video_slot,
)

inject_base_styles()

st.title("📉 Tema 2 · Pérdida y Backpropagation")
st.caption("Cómo la red convierte 'me equivoqué' en un número, y cómo sabe qué mover.")

st.markdown(
    """
    De pequeños jugábamos a **frío/caliente**: alguien esconde un objeto y te dice "frío" o
    "caliente" según te acercas o alejas. Una **loss function** es exactamente esa voz: un
    único número que te dice cuánto te has equivocado, cada vez que predices algo.

    Pero saber que hace "frío" no basta — necesitas saber **en qué dirección moverte**. Eso
    es lo que calcula **backpropagation**: recorre la red hacia atrás y reparte la
    responsabilidad del error entre cada peso, usando la regla de la cadena. El resultado es
    el **gradiente**: la brújula que le dice al optimizador hacia dónde mover cada parámetro.
    """
)

render_video_slot("topic2")

st.divider()

st.subheader("Pruébalo tú: mueve el peso, observa el gradiente")
st.caption(
    "Ejemplo mínimo del Notebook 2: `predicción = x · w`, con `x = 2.0` fijo y un objetivo real de `2.0`."
)

w = st.slider("Peso w", -1.0, 3.0, 0.5, 0.05)
x = 2.0
target = 2.0
pred = x * w
loss = (pred - target) ** 2
gradient = 2 * (pred - target) * x

metric_cols = st.columns(3)
metric_cols[0].metric("Predicción (x·w)", f"{pred:.2f}")
metric_cols[1].metric("Loss (error²)", f"{loss:.2f}")
metric_cols[2].metric("Gradiente dL/dw", f"{gradient:.2f}")

if abs(gradient) < 0.05:
    st.success("Gradiente ≈ 0: estás muy cerca del mínimo de la loss.", icon=":material/check_circle:")
elif gradient < 0:
    st.info("Gradiente negativo → **subir** w reduce la loss.", icon=":material/trending_up:")
else:
    st.info("Gradiente positivo → **bajar** w reduce la loss.", icon=":material/trending_down:")

st.caption(
    "Esto es exactamente lo que hace `loss.backward()` en PyTorch o `tape.gradient(...)` en "
    "TensorFlow — solo que para miles de weights a la vez, no uno."
)

st.divider()
st.subheader("Ejercicios reales: el notebook")
st.caption(
    "La teoría y esta mini-demo solo dan la intuición. Los ejercicios de verdad — con "
    "código que tú completas, marcado con ✏️✏️✏️ — están en el notebook."
)
render_notebook_downloads("topic2")

st.divider()
render_completion_button("topic2")
