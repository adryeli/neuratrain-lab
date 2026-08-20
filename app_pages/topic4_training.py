"""Tema 4 · Entrenamiento y sobreajuste — theory, video, and a live overfitting demo."""

from __future__ import annotations

from sklearn.datasets import make_moons

from neurotrain.visualization import plot_training_history

import streamlit as st

from app_pages.shared import (
    inject_base_styles,
    render_completion_button,
    render_notebook_downloads,
    render_video_slot,
)

inject_base_styles()

st.title("🎛️ Tema 4 · Entrenamiento y sobreajuste")
st.caption("Epochs, batches, EarlyStopping y Dropout: organizar miles de actualizaciones sin memorizar de más.")

st.markdown(
    """
    Piensa en cómo preparas un examen: **train** son los ejercicios con los que estudias,
    **validation** son los simulacros que te dicen si vas bien encaminado (sin contar para
    la nota final), y **test** es el examen real, que solo abres al final.

    Si estudias demasiados ejercicios de memoria en vez de entender el patrón, sacarás nota
    perfecta en los ejercicios... y mediocre en el examen real. Eso es **sobreajuste**
    (overfitting): la red memoriza el training set en vez de generalizar.

    Dos herramientas lo controlan:
    - **EarlyStopping** — para el entrenamiento en cuanto la validación deja de mejorar.
    - **Dropout** — apaga neuronas al azar durante el entrenamiento, para que la red no
      dependa demasiado de combinaciones concretas.
    """
)

render_video_slot("topic4")

st.divider()

st.subheader("Pruébalo tú: fuerza el sobreajuste")
st.caption(
    "Entrenamos deliberadamente una red grande, sin Dropout ni EarlyStopping, sobre muy "
    "pocos datos — la receta perfecta para ver el sobreajuste en directo."
)

if st.button("Mostrar sobreajuste en acción", type="primary", icon=":material/play_arrow:"):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    X_small, y_small = make_moons(n_samples=60, noise=0.3, random_state=42)

    with st.spinner("Entrenando una red sobredimensionada sin regularización..."):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(
            X_small, y_small,
            validation_split=0.4,
            epochs=250,
            batch_size=8,
            verbose=0,
        )

    history_dict = {key: list(values) for key, values in history.history.items()}
    st.pyplot(plot_training_history(history_dict), width="stretch")

    final_train_loss = history_dict["loss"][-1]
    final_val_loss = history_dict["val_loss"][-1]
    gap = final_val_loss - final_train_loss
    if gap > 0.3:
        st.warning(
            f"Sobreajuste claro: la loss de entrenamiento acaba en {final_train_loss:.3f} "
            f"pero la de validación en {final_val_loss:.3f}. La red memorizó, no generalizó.",
            icon=":material/warning:",
        )
    else:
        st.info(
            f"Con esta semilla la brecha fue moderada (train {final_train_loss:.3f} vs. "
            f"val {final_val_loss:.3f}) — vuelve a pulsar el botón o prueba en el notebook "
            "con más epochs para verlo más claro.",
            icon=":material/info:",
        )
else:
    st.markdown(
        '<div class="note">Pulsa el botón para entrenar una red deliberadamente sobredimensionada y ver cómo se separan train y validation loss.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.subheader("Ejercicios reales: el notebook")
st.caption(
    "La teoría y esta mini-demo solo dan la intuición. Los ejercicios de verdad — con "
    "código que tú completas, marcado con ✏️✏️✏️, y el experimento guiado A/B completo — "
    "están en el notebook."
)
render_notebook_downloads("topic4")

st.divider()
render_completion_button("topic4")

st.divider()
st.subheader("Siguiente parada: el laboratorio")
st.markdown(
    "Cuando termines los 4 temas, ve a **Laboratorio → Modo Experimento** en la barra "
    "lateral: ahí entrenas una red de verdad sobre el dataset clínico real, con todos los "
    "controles (arquitectura, epochs, batch size, Dropout, EarlyStopping) en tus manos."
)
