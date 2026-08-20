"""Modo Experimento — the real, functional training lab.

This is the existing NeuroTrain Lab experience (train an MLP live, tweak
hyperparameters, compare against a baseline), relocated from the old
single-page app.py and extended with live per-epoch progress and an
annotated code-reveal panel that ties the running experiment back to the
concepts covered in Notebooks 1-3.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import streamlit as st

from neurotrain.config import TrainingConfig
from neurotrain.evaluation import classification_metrics
from neurotrain.modeling import (
    fit_logistic_baseline,
    make_progress_callback,
    tensorflow_available,
    train_dense_classifier,
)
from neurotrain.visualization import plot_confusion, plot_roc, plot_training_history

from app_pages.shared import get_dataset, get_prepared_data, inject_base_styles

inject_base_styles()

st.title("🧪 Modo Experimento")
st.caption(
    "El laboratorio real: entrena una red neuronal con datos clínicos reales y compara "
    "cada decisión contra un baseline honesto. Esto es la culminación de los 4 temas."
)

frame = get_dataset()
prepared = get_prepared_data()

with st.sidebar:
    st.header("Configura el experimento")
    architecture_name = st.selectbox(
        "Arquitectura",
        ["Pequeña · 32 → 16", "Mediana · 64 → 32", "Grande · 128 → 64"],
        help="Más neuronas aumentan la capacidad, pero no garantizan mejor generalización.",
    )
    architectures = {
        "Pequeña · 32 → 16": (32, 16),
        "Mediana · 64 → 32": (64, 32),
        "Grande · 128 → 64": (128, 64),
    }
    epochs = st.slider("Épocas máximas", 20, 300, 150, 10)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=32)
    dropout_rate = st.slider("Dropout", 0.0, 0.6, 0.30, 0.05)
    use_early_stopping = st.toggle("Usar EarlyStopping", value=True)
    patience = st.slider("Paciencia", 2, 30, 12, disabled=not use_early_stopping)
    train_clicked = st.button("Entrenar experimento", type="primary", width="stretch")

    st.caption("Semilla fija: 42 · Split estratificado 70/15/15")

config = TrainingConfig(
    hidden_units=architectures[architecture_name],
    dropout_rate=dropout_rate,
    epochs=epochs,
    batch_size=batch_size,
    patience=patience,
    use_early_stopping=use_early_stopping,
)

summary_cols = st.columns(4)
summary_cols[0].metric("Registros reales", f"{len(frame):,}")
summary_cols[1].metric("Variables", frame.shape[1] - 1)
summary_cols[2].metric("Train / Val / Test", f"{len(prepared.y_train)} / {len(prepared.y_val)} / {len(prepared.y_test)}")
summary_cols[3].metric("Clase positiva", "Maligno = 1")

if not tensorflow_available():
    st.error(
        "TensorFlow no está instalado en este entorno. Activa `.venv` y ejecuta "
        "`pip install -r requirements.txt`.",
        icon=":material/error:",
    )

if train_clicked and tensorflow_available():
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def _report_progress(epoch: int, total_epochs: int, logs: dict[str, float]) -> None:
        progress_bar.progress(min(epoch / total_epochs, 1.0))
        loss_value = logs.get("loss")
        loss_text = f" · loss {loss_value:.4f}" if loss_value is not None else ""
        status_text.caption(f"Época {epoch}/{total_epochs}{loss_text}")

    progress_callback = make_progress_callback(_report_progress, config.epochs)

    try:
        baseline = fit_logistic_baseline(prepared)
        baseline_probabilities = baseline.predict_proba(prepared.X_test)[:, 1]
        model, history = train_dense_classifier(
            prepared, config, extra_callbacks=[progress_callback]
        )
        probabilities = model.predict(prepared.X_test, verbose=0).ravel()
        st.session_state["experiment"] = {
            "config": config,
            "history": history,
            "probabilities": probabilities,
            "baseline_probabilities": baseline_probabilities,
            "model": model,
        }
        progress_bar.progress(1.0)
        status_text.empty()
        st.balloons()
    except Exception as exc:  # the UI should show a useful failure, not hide it
        st.exception(exc)

experiment = st.session_state.get("experiment")

tab_train, tab_metrics, tab_cases, tab_code, tab_guide = st.tabs(
    ["Entrenamiento", "Métricas", "Casos del dataset", "Bajo el capó", "Guía rápida"]
)

with tab_train:
    st.subheader("Qué cambia al pulsar Entrenar")
    st.markdown(
        """
        1. La red recibe un **batch** de ejemplos y calcula una predicción.
        2. Compara la predicción con la etiqueta mediante la **loss**.
        3. Backpropagation calcula cómo contribuyó cada peso al error.
        4. Adam actualiza los pesos. Al recorrer todo train termina una **época**.
        5. Validación mide si lo aprendido generaliza a datos no usados para ajustar pesos.
        """
    )
    updates = -(-len(prepared.y_train) // config.batch_size)
    st.info(
        f"Con batch size {config.batch_size}, una época contiene aproximadamente "
        f"{updates} actualizaciones de pesos sobre {len(prepared.y_train)} ejemplos de train.",
        icon=":material/info:",
    )

    if experiment:
        trained_epochs = len(experiment["history"]["loss"])
        st.success(
            f"Experimento terminado: {trained_epochs} de {experiment['config'].epochs} "
            "épocas máximas ejecutadas.",
            icon=":material/check_circle:",
        )
        st.pyplot(plot_training_history(experiment["history"]), width="stretch")
        if trained_epochs < experiment["config"].epochs:
            st.caption("EarlyStopping detuvo el entrenamiento y restauró los mejores pesos.")
    else:
        st.markdown('<div class="note">Configura el experimento en la barra lateral y pulsa <b>Entrenar experimento</b>.</div>', unsafe_allow_html=True)

with tab_metrics:
    threshold = st.slider(
        "Umbral para convertir probabilidad en clase",
        0.20,
        0.80,
        0.50,
        0.01,
        help="Reducirlo suele aumentar sensibilidad y también falsos positivos.",
    )
    if experiment:
        neural_metrics = classification_metrics(
            prepared.y_test, experiment["probabilities"], threshold
        )
        baseline_metrics = classification_metrics(
            prepared.y_test, experiment["baseline_probabilities"], threshold
        )
        metric_order = ["accuracy", "roc_auc", "precision", "sensitivity", "specificity", "f1"]
        comparison = pd.DataFrame(
            {
                "Red neuronal": [neural_metrics[name] for name in metric_order],
                "Regresión logística": [baseline_metrics[name] for name in metric_order],
            },
            index=["Accuracy", "ROC-AUC", "Precisión", "Sensibilidad", "Especificidad", "F1"],
        )
        st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
        st.caption(
            "Que el baseline gane no es un fallo: es evidencia de que este dataset pequeño "
            "puede no necesitar una red neuronal."
        )

        left, right = st.columns(2)
        with left:
            st.pyplot(
                plot_confusion(prepared.y_test, experiment["probabilities"], threshold),
                width="stretch",
            )
        with right:
            st.pyplot(plot_roc(prepared.y_test, experiment["probabilities"]), width="stretch")

        report = {
            "config": asdict(experiment["config"]),
            "threshold": threshold,
            "neural_network": neural_metrics,
            "logistic_regression": baseline_metrics,
            "epochs_executed": len(experiment["history"]["loss"]),
        }
        st.download_button(
            "Descargar resultados JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="neurotrain_experiment.json",
            mime="application/json",
            icon=":material/download:",
        )
    else:
        st.warning("Entrena un experimento para comparar las métricas.", icon=":material/warning:")

with tab_cases:
    st.subheader("Inspecciona un registro reservado para test")
    st.caption("Estos registros no participaron en el ajuste de pesos ni del scaler.")
    if experiment:
        row_position = st.number_input(
            "Posición dentro de test",
            min_value=0,
            max_value=len(prepared.X_test_raw) - 1,
            value=0,
            step=1,
        )
        probability = float(experiment["probabilities"][row_position])
        actual = "Maligno" if prepared.y_test[row_position] == 1 else "Benigno"
        predicted = "Maligno" if probability >= threshold else "Benigno"
        case_cols = st.columns(3)
        case_cols[0].metric("Etiqueta real", actual)
        case_cols[1].metric("Probabilidad estimada", f"{probability:.1%}")
        case_cols[2].metric("Clase con el umbral actual", predicted)
        selected = prepared.X_test_raw.iloc[int(row_position)]
        st.dataframe(
            selected.rename("valor").to_frame().head(10),
            width="stretch",
        )
        st.caption("Se muestran solo las primeras 10 de 30 variables para mantener la vista legible.")
    else:
        st.warning("Entrena un experimento antes de inspeccionar predicciones.", icon=":material/warning:")

with tab_code:
    st.subheader("El código real detrás de cada paso")
    st.caption(
        "Cada bloque de aquí es el mismo mecanismo que ya viste en los notebooks — "
        "esto es esa teoría, corriendo de verdad sobre datos reales."
    )
    with st.expander("1 · Forward pass — visto en el Notebook 1", icon=":material/arrow_forward:"):
        st.code(
            """
            # src/neurotrain/modeling.py — build_dense_classifier()
            layers = [tf.keras.layers.Input(shape=(input_dim,))]
            for units in config.hidden_units:
                layers.append(tf.keras.layers.Dense(units, activation="relu"))
                if config.dropout_rate > 0:
                    layers.append(tf.keras.layers.Dropout(config.dropout_rate))
            layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
            """,
            language="python",
        )
        st.caption("Cada Dense es exactamente `activación(X · W + b)` — la multiplicación de matrices del Notebook 1.")
    with st.expander("2 · Loss — visto en el Notebook 2", icon=":material/rule:"):
        st.code(
            """
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                loss="binary_crossentropy",
                ...
            )
            """,
            language="python",
        )
        st.caption("Binary cross-entropy: la misma loss que calculaste a mano en el Notebook 2, aplicada aquí a 30 variables reales.")
    with st.expander("3 · Backpropagation — visto en el Notebook 2", icon=":material/undo:"):
        st.code(
            """
            # dentro de model.fit(): por cada batch
            # 1. forward -> predicción
            # 2. loss = binary_crossentropy(y, predicción)
            # 3. gradientes = regla de la cadena sobre el grafo computacional
            """,
            language="python",
        )
        st.caption("Keras hace automáticamente lo que hiciste a mano con el ejemplo x=2.0, w=0.5 del Notebook 2.")
    with st.expander("4 · Optimizer step — visto en el Notebook 3", icon=":material/speed:"):
        st.code(
            """
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            # Adam ajusta cada peso usando estadísticas acumuladas de sus gradientes
            """,
            language="python",
        )
        st.caption("El mismo Adam que comparaste contra SGD y Momentum en el Notebook 3.")

with tab_guide:
    st.subheader("Cómo leer las curvas")
    st.markdown(
        """
        - **Train loss y val loss bajan:** el modelo aprende y generaliza mejor.
        - **Train loss baja, pero val loss sube:** aparece sobreajuste.
        - **Ambas pérdidas se mantienen altas:** puede haber infraajuste o una configuración inadecuada.
        - **EarlyStopping:** detiene el proceso cuando `val_loss` deja de mejorar durante la paciencia elegida.
        - **Dropout:** apaga unidades al azar solo durante entrenamiento para reducir dependencia excesiva entre neuronas.
        """
    )
    st.markdown(
        "**Reto recomendado:** entrena primero 128 → 64, dropout 0 y sin EarlyStopping; "
        "después repite con dropout 0.30 y EarlyStopping. Compara `val_loss`, no solo accuracy."
    )

st.caption(
    "Uso exclusivamente educativo. Dataset histórico agregado; no introduce datos personales "
    "ni utilices esta aplicación para diagnóstico, triaje o decisiones clínicas."
)
