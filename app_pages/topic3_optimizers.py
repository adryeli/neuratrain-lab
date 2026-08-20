"""Tema 3 · Optimizadores — theory, video, and a live gradient-descent demo."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from app_pages.shared import (
    inject_base_styles,
    render_completion_button,
    render_notebook_downloads,
    render_video_slot,
)

inject_base_styles()

st.title("🧭 Tema 3 · Optimizadores")
st.caption("SGD, Momentum y Adam: distintas formas de bajar la montaña de la loss.")

st.markdown(
    """
    Imagina un **excursionista en medio de la niebla**, buscando el punto más bajo del
    valle. No ve el paisaje completo — solo siente la pendiente bajo sus pies y da un paso
    en esa dirección. Eso es **gradient descent**: en cada paso, mueve los parámetros un
    poco en la dirección que el gradiente dice que reduce la loss.

    - **SGD** da ese paso directo, sin memoria de pasos anteriores — a veces zigzaguea.
    - **Momentum** añade inercia: como una bola rodando, no olvida del todo la velocidad
      que ya llevaba, así que suaviza el zigzag.
    - **Adam** va un paso más allá: adapta el tamaño del paso **por cada parámetro**,
      usando estadísticas acumuladas de sus gradientes. Por eso es el punto de partida más
      frecuente en la industria.
    """
)

render_video_slot("topic3")

st.divider()

st.subheader("Pruébalo tú: elige un learning rate")
st.caption("Loss de juguete: f(w) = (w − 3)² + 1. El mínimo está en w = 3.")

learning_rate = st.select_slider(
    "Learning rate",
    options=[0.01, 0.05, 0.1, 0.3, 0.6, 0.95],
    value=0.1,
)
steps = st.slider("Número de pasos", 5, 40, 15)

w = -3.0
trajectory = [w]
for _ in range(steps):
    gradient = 2 * (w - 3)
    w = w - learning_rate * gradient
    trajectory.append(w)

w_range = np.linspace(-5, 11, 200)
loss_curve = (w_range - 3) ** 2 + 1
trajectory = np.array(trajectory)
trajectory_loss = (trajectory - 3) ** 2 + 1

fig, axis = plt.subplots(figsize=(7, 4))
axis.plot(w_range, loss_curve, color="#94A3B8", label="f(w)")
axis.plot(trajectory, trajectory_loss, marker="o", color="#7C3AED", label="pasos del descenso")
axis.scatter([trajectory[-1]], [trajectory_loss[-1]], color="#F97316", zorder=5, label="posición final")
axis.set(xlabel="w", ylabel="loss")
axis.legend()
axis.grid(alpha=0.2)
fig.tight_layout()
st.pyplot(fig, width="stretch")

final_gap = abs(trajectory[-1] - 3)
if final_gap < 0.1:
    st.success(f"Convergencia suave: terminó a {final_gap:.2f} del mínimo real.", icon=":material/check_circle:")
elif final_gap > 5:
    st.error("Con este learning rate el descenso diverge — se aleja del mínimo en vez de acercarse.", icon=":material/error:")
else:
    st.warning(f"Progreso lento: todavía a {final_gap:.2f} del mínimo tras {steps} pasos.", icon=":material/warning:")

st.divider()
st.subheader("Ejercicios reales: el notebook")
st.caption(
    "La teoría y esta mini-demo solo dan la intuición. Los ejercicios de verdad — con "
    "código que tú completas, marcado con ✏️✏️✏️, incluyendo la demostración de vanishing "
    "gradient — están en el notebook."
)
render_notebook_downloads("topic3")

st.divider()
render_completion_button("topic3")
