"""Shared business logic for the NeuroTrain Lab app: cached loaders, session
state helpers, and small reusable UI pieces used across pages.

Not a page itself — imported by the page scripts in this folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neurotrain.data import load_dataset, prepare_data

TOPIC_KEYS = ["topic1", "topic2", "topic3", "topic4"]

TOPIC_TITLES = {
    "topic1": "1 · El Perceptrón",
    "topic2": "2 · Pérdida y Backpropagation",
    "topic3": "3 · Optimizadores",
    "topic4": "4 · Entrenamiento y sobreajuste",
}

# Real, curated links from the instructor's own playlist — swap these for your
# own recording or a different video whenever you like, this is just a working
# default so the journey isn't empty on day one. Some topics have more than one.
VIDEO_DEFAULTS = {
    "topic1": [
        "https://www.youtube.com/watch?v=MRIv2IwFTPg&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=1",
        "https://www.youtube.com/watch?v=uwbHOpp9xkc&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=2",
    ],
    "topic2": [
        "https://www.youtube.com/watch?v=eNIqz_noix8&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=4",
    ],
    "topic3": [
        "https://www.youtube.com/watch?v=MD2fYip6QsQ&t=337s",
    ],
    "topic4": [
        "https://www.youtube.com/watch?v=7-6X3DTt3R8&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=7",
        "https://www.youtube.com/watch?v=ZmLKqZYlYUI&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=8",
    ],
}

NOTEBOOK_SUMMARIES = {
    "topic1": (
        "Qué es una neurona, las funciones de activación (ReLU, Sigmoid, Softmax), cómo se "
        "apilan en un MLP, forward propagation como multiplicación de matrices, fronteras "
        "de decisión con `make_moons`, y tensores en NumPy, PyTorch y TensorFlow."
    ),
    "topic2": (
        "Qué es una loss function, MSE para regresión, Cross-Entropy para clasificación, la "
        "regla de la cadena sobre un grafo computacional (a mano y con autograd de PyTorch), "
        "y cómo se aplica todo esto al dataset real."
    ),
    "topic3": (
        "Descenso de gradiente, el efecto del learning rate, SGD frente a Momentum frente a "
        "Adam, el problema del vanishing gradient (demostrado con un experimento real), y "
        "una tabla de diagnóstico para cuando una red no aprende."
    ),
    "topic4": (
        "Train/validation/test, escalado sin fuga de información, un baseline honesto, "
        "epochs y batches, EarlyStopping y Dropout, cómo leer curvas de aprendizaje, y un "
        "experimento guiado A/B comparando sobreajuste."
    ),
}

DATASET_CSV_PATH = ROOT / "data" / "breast_cancer_wisconsin.csv"

# The two bonus/solved notebooks in notebooks/material_adicional/, each paired
# with its own explanatory video (not the topic videos above).
BONUS_NOTEBOOKS = {
    "punto_de_partida": {
        "title": "Punto de partida: una red neuronal desde cero",
        "description": (
            "Construye una red neuronal completa a mano, sin frameworks: capas, pesos, "
            "activaciones, forward pass, backpropagation y descenso de gradiente, sobre "
            "el dataset `make_circles`. Ideal para ver, sin ninguna caja negra, el mismo "
            "mecanismo que ya conoces de los Notebooks 1 a 3."
        ),
        "relative_path": "notebooks/material_adicional/punto_de_partida.ipynb",
        "file_name": "punto_de_partida.ipynb",
        "video": "https://www.youtube.com/watch?v=MRIv2IwFTPg&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=1",
    },
    "tres_maneras": {
        "title": "3 Maneras de Programar una Red Neuronal",
        "description": (
            "Material de DOTCSV, adaptado por Elizabeth Sena. El mismo problema resuelto de "
            "tres formas: TensorFlow de bajo nivel, Keras de alto nivel, y scikit-learn como "
            "caja negra — la manera más habitual en la industria hoy en día."
        ),
        "relative_path": "notebooks/material_adicional/3_maneras_de_programar_una_red_neuronal.ipynb",
        "file_name": "3_maneras_de_programar_una_red_neuronal.ipynb",
        "video": "https://www.youtube.com/watch?v=qTNUbPkR2ao",
    },
}

NOTEBOOK_FILES = {
    "topic1": {
        "es": ("01_perceptron.ipynb", "notebooks/es/01_perceptron.ipynb"),
        "en": ("01_perceptron.ipynb", "notebooks/en/01_perceptron.ipynb"),
    },
    "topic2": {
        "es": ("02_perdida_y_backpropagation.ipynb", "notebooks/es/02_perdida_y_backpropagation.ipynb"),
        "en": ("02_loss_and_backpropagation.ipynb", "notebooks/en/02_loss_and_backpropagation.ipynb"),
    },
    "topic3": {
        "es": ("03_optimizadores.ipynb", "notebooks/es/03_optimizadores.ipynb"),
        "en": ("03_optimizers.ipynb", "notebooks/en/03_optimizers.ipynb"),
    },
    "topic4": {
        "es": ("04_entrenamiento_y_sobreajuste.ipynb", "notebooks/es/04_entrenamiento_y_sobreajuste.ipynb"),
        "en": ("04_training_and_overfitting.ipynb", "notebooks/en/04_training_and_overfitting.ipynb"),
    },
}


def inject_base_styles() -> None:
    """Shared CSS: the gradient hero banner and the soft callout box.

    A gradient background isn't achievable with native Streamlit elements,
    so a small amount of custom HTML/CSS is a deliberate exception here —
    it's also the app's one consistent piece of brand identity.
    """

    st.markdown(
        """
        <style>
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 1rem;
            color: white;
            background: linear-gradient(120deg, #312E81 0%, #7C3AED 55%, #2563EB 100%);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2.15rem; }
        .hero p { margin: .45rem 0 0; max-width: 850px; opacity: .92; }
        .note {
            padding: .8rem 1rem;
            border-left: 4px solid #7C3AED;
            background: #F5F3FF;
            border-radius: .35rem;
        }
        .milestone {
            text-align: center;
            opacity: .85;
            font-style: italic;
            margin: .6rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    """The big gradient banner — reserved for the home page only, so it
    keeps its impact instead of becoming repetitive wallpaper."""

    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data
def get_prepared_data():
    return prepare_data(get_dataset())


def ensure_progress_state() -> None:
    st.session_state.setdefault("progress", {key: False for key in TOPIC_KEYS})


def mark_topic_complete(topic_key: str) -> None:
    ensure_progress_state()
    st.session_state["progress"][topic_key] = True


def progress_fraction() -> float:
    ensure_progress_state()
    progress = st.session_state["progress"]
    return sum(progress.values()) / len(progress)


def render_video_slot(topic_key: str, lang_label: str = "Vídeo de la clase") -> None:
    videos = VIDEO_DEFAULTS[topic_key]
    st.subheader(lang_label)
    if len(videos) == 1:
        st.video(videos[0])
    else:
        for index, url in enumerate(videos, start=1):
            st.caption(f"Vídeo {index}")
            st.video(url)


def render_notebook_downloads(topic_key: str) -> None:
    """Lets a student grab either language edition of this topic's notebook
    without the app itself needing to be bilingual."""

    filenames = NOTEBOOK_FILES[topic_key]
    columns = st.columns(2)
    for column, lang in zip(columns, ["es", "en"]):
        display_name, relative_path = filenames[lang]
        notebook_path = ROOT / relative_path
        with column:
            if notebook_path.exists():
                st.download_button(
                    f"Notebook en {'español' if lang == 'es' else 'inglés'}",
                    data=notebook_path.read_bytes(),
                    file_name=display_name,
                    mime="application/x-ipynb+json",
                    icon=":material/download:",
                    width="stretch",
                )


def render_bonus_notebook(key: str) -> None:
    """Renders one bonus/solved notebook block: title, description, download, video."""

    info = BONUS_NOTEBOOKS[key]
    notebook_path = ROOT / info["relative_path"]

    st.markdown(f"**{info['title']}**")
    st.markdown(info["description"])
    if notebook_path.exists():
        st.download_button(
            "Descargar notebook resuelto",
            data=notebook_path.read_bytes(),
            file_name=info["file_name"],
            mime="application/x-ipynb+json",
            icon=":material/download:",
        )
    st.caption("Vídeo explicativo:")
    st.video(info["video"])


def render_dataset_download() -> None:
    if DATASET_CSV_PATH.exists():
        st.download_button(
            "Descargar breast_cancer_wisconsin.csv",
            data=DATASET_CSV_PATH.read_bytes(),
            file_name="breast_cancer_wisconsin.csv",
            mime="text/csv",
            icon=":material/download:",
        )


def render_completion_button(topic_key: str) -> None:
    ensure_progress_state()
    done = st.session_state["progress"][topic_key]
    if done:
        st.success("Tema marcado como completado.", icon=":material/check_circle:")
    else:
        if st.button(
            f"Marcar «{TOPIC_TITLES[topic_key]}» como completado",
            type="primary",
            icon=":material/check_circle:",
        ):
            mark_topic_complete(topic_key)
            st.rerun()
