"""Tema 1 · El Perceptrón — theory, video, and a live decision-boundary demo."""

from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.datasets import make_moons

from neurotrain.visualization import plot_decision_boundary

from app_pages.shared import (
    inject_base_styles,
    render_completion_button,
    render_notebook_downloads,
    render_video_slot,
)

inject_base_styles()

st.title("🧩 Tema 1 · El Perceptrón")
st.caption("Cómo una neurona convierte números en una decisión, y por qué hacen falta varias.")

st.markdown(
    """
    Imagina un **portero de discoteca**. No mira un solo dato: pondera varios — ¿va bien
    vestido?, ¿tiene reserva?, ¿es muy tarde? — y cada criterio le importa un poco distinto.
    Si la suma ponderada supera su umbral de exigencia esa noche, te deja pasar.

    Una **neurona artificial** hace justo eso con números: multiplica cada entrada por un
    **weight** (cuánto importa), suma un **bias** (lo exigente que es por defecto), y aplica
    una **función de activación** (ReLU, Sigmoid...) para decidir la salida. Apilar muchas
    neuronas en capas es lo que llamamos una **red multicapa (MLP)** — y ahí es donde una
    red deja de trazar solo líneas rectas y empieza a curvar fronteras complejas.
    """
)

render_video_slot("topic1")

st.divider()

st.subheader("Pruébalo tú: perceptrón vs. MLP")
st.caption(
    "Con 30 variables reales no se puede dibujar la frontera de decisión. Aquí usamos un "
    "dataset sintético de 2 variables — igual que en el Notebook 1 — para que la veas con tus ojos."
)

noise = st.slider("Ruido del dataset", 0.05, 0.40, 0.20, 0.05, help="Más ruido = clases más mezcladas, más difícil de separar.")
demo_clicked = st.button("Entrenar y comparar", type="primary", icon=":material/play_arrow:")

if demo_clicked:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    X_moons, y_moons = make_moons(n_samples=300, noise=noise, random_state=42)

    with st.spinner("Entrenando un perceptrón y un MLP pequeño..."):
        perceptron = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        perceptron.compile(optimizer="adam", loss="binary_crossentropy")
        perceptron.fit(X_moons, y_moons, epochs=80, verbose=0)

        mlp = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        mlp.compile(optimizer="adam", loss="binary_crossentropy")
        mlp.fit(X_moons, y_moons, epochs=80, verbose=0)

    left, right = st.columns(2)
    with left:
        fig = plot_decision_boundary(
            X_moons, y_moons,
            lambda grid: perceptron.predict(grid, verbose=0).ravel(),
            title="Un perceptrón: solo una recta", lang="es",
        )
        st.pyplot(fig, width="stretch")
    with right:
        fig = plot_decision_boundary(
            X_moons, y_moons,
            lambda grid: mlp.predict(grid, verbose=0).ravel(),
            title="MLP (Dense 8, ReLU): curva la frontera", lang="es",
        )
        st.pyplot(fig, width="stretch")

    perceptron_accuracy = (
        (perceptron.predict(X_moons, verbose=0).ravel() >= 0.5).astype(int) == y_moons
    ).mean()
    mlp_accuracy = ((mlp.predict(X_moons, verbose=0).ravel() >= 0.5).astype(int) == y_moons).mean()
    metric_cols = st.columns(2)
    metric_cols[0].metric("Accuracy del perceptrón", f"{perceptron_accuracy:.1%}")
    metric_cols[1].metric("Accuracy del MLP", f"{mlp_accuracy:.1%}")
else:
    st.markdown(
        '<div class="note">Elige el ruido y pulsa <b>Entrenar y comparar</b> para ver la frontera de cada modelo.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Actividad didáctica: TensorFlow Playground")
st.markdown(
    """
    Selecciona una entrada de datos y juega a cuántas neuronas necesitas para que se
    separen los datos. Después intenta comprender qué learning rate usaste, qué función
    de activación, cuántas neuronas en cada capa, etc.
    """
)
st.link_button(
    "Abrir TensorFlow Playground",
    "https://playground.tensorflow.org/",
    icon=":material/open_in_new:",
)

st.divider()
st.subheader("Ejercicios reales: el notebook")
st.caption(
    "La teoría y esta mini-demo solo dan la intuición. Los ejercicios de verdad — con "
    "código que tú completas, marcado con ✏️✏️✏️ — están en el notebook."
)
render_notebook_downloads("topic1")

st.divider()
render_completion_button("topic1")
