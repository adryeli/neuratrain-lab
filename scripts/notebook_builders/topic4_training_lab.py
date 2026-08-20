"""Notebook 4 — Entrenamiento y Sobreajuste / Training and Overfitting.

Covers: dataset audit, train/validation/test split, scaling without leakage,
a Logistic Regression baseline, epoch/batch/iteration arithmetic, building
and compiling an MLP, EarlyStopping + Dropout, ``.fit()``, reading loss/
accuracy curves for overfitting, test evaluation vs. baseline, a guided A/B
overfitting experiment (two independent ``TrainingConfig`` runs), artifact
saving, and the closing self-assessment. Last of the 4 notebooks — assumes
NB1 (perceptron/forward-prop), NB2 (loss/backprop) and NB3 (optimizers) are
already known and does not re-explain them.
"""

from __future__ import annotations

from .assessment import open_question, quiz_question, section_header
from .callouts import callout
from .common import code, markdown
from .exercises import exercise_cell
from .milestones import milestone


def build_es_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        markdown(
            """
            # NeuroTrain Lab — Notebook 4: Entrenamiento y Sobreajuste

            **Tema:** cómo organizar un entrenamiento real de principio a fin —
            split train/validation/test, escalado sin fuga de información, un
            baseline honesto, `Dropout` y `EarlyStopping` — y cómo detectar el
            sobreajuste leyendo las curvas de aprendizaje.

            > Último notebook de 4. Ya conoces el perceptrón (NB1), la pérdida y
            > el backprop (NB2), y los optimizadores (NB3). Aquí juntamos todo
            > para entrenar un modelo real sobre un problema real y aprender a
            > **controlar** su sobreajuste, no solo observarlo.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 Qué aprenderás en este notebook

            Al terminar podrás explicar, sin fórmulas de memoria:

            1. Por qué se separan tres conjuntos (train/validation/test) y no dos.
            2. Por qué el escalado se ajusta solo con train.
            3. Qué hacen `Dropout` y `EarlyStopping`, y por qué se combinan.
            4. Cómo leer `loss` y `val_loss` para diagnosticar sobreajuste.
            5. Por qué siempre comparamos contra un baseline más simple.

            **Mapa mental:** `datos reales → split → escalado → baseline → MLP → EarlyStopping+Dropout → fit() → curvas → test vs baseline → experimento A/B`
            """
        )
    )
    cells.append(
        code(
            """
            from pathlib import Path
            import json
            import math
            import sys

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate
            from neurotrain.config import TrainingConfig
            from neurotrain.data import load_dataset, prepare_data
            from neurotrain.evaluation import classification_metrics
            from neurotrain.modeling import train_dense_classifier
            from neurotrain.visualization import (
                plot_confusion,
                plot_roc,
                plot_training_history,
                plot_training_history_comparison,
            )

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("TensorFlow:", tf.__version__)
            print("Raíz del proyecto:", PROJECT_ROOT)
            """
        )
    )

    cells.append(markdown("## 1. Auditoría rápida del dataset"))
    cells.append(
        markdown(
            """
            Ya viste este CSV de pasada en el Notebook 1. Antes de entrenar
            confirmamos su contrato mínimo con `assert`: forma, ausencia de
            nulos y las dos etiquetas esperadas. Es la última vez que lo hacemos
            "a mano" — el resto del proyecto usa `neurotrain.data.load_dataset()`,
            que hace exactamente estas comprobaciones.
            """
        )
    )
    cells.append(
        code(
            """
            DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
            df = pd.read_csv(DATA_PATH)

            assert df.shape == (569, 31)
            assert not df.isna().any().any()
            assert set(df["diagnosis"].unique()) == {"B", "M"}

            class_counts = df["diagnosis"].value_counts().rename(index={"B": "Benigno", "M": "Maligno"})
            print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
            display(class_counts.to_frame("registros"))
            """
        )
    )

    cells.append(markdown("## 2. Separar X e y"))
    cells.append(
        markdown(
            """
            La clase positiva se define explícitamente como **maligno = 1** —
            así "sensibilidad" significa "proporción de malignos reales que
            detectamos", sin ambigüedad.
            """
        )
    )
    cells.append(
        code(
            """
            X = df.drop(columns="diagnosis")
            y = df["diagnosis"].eq("M").astype("int8")

            print("Forma de X:", X.shape, "| Forma de y:", y.shape)
            """
        )
    )

    cells.append(markdown("## 3. Crear train, validation y test"))
    cells.append(
        callout(
            "concept",
            "es",
            "¿Por qué tres conjuntos y no dos?",
            """
            `train` son los ejercicios con los que la red ajusta sus pesos.
            `validation` son simulacros: la red no aprende de ellos directamente,
            pero **nosotros** los usamos para decidir arquitectura, dropout,
            paciencia o umbral. `test` es el examen final que solo se abre
            **una vez**, al terminar. Si usas test para decidir nada, ya no mide
            lo que dice medir: solo mide qué tan bien te ajustaste al examen.
            """,
        )
    )
    cells.append(
        markdown(
            """
            Usaremos aproximadamente 70% / 15% / 15%. `stratify=y` mantiene una
            proporción parecida de benignos y malignos en cada partición.
            """
        )
    )
    cells.append(
        code(
            """
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE,
            )

            split_summary = pd.DataFrame(
                {
                    "registros": [len(X_train), len(X_val), len(X_test)],
                    "% malignos": [y_train.mean(), y_val.mean(), y_test.mean()],
                },
                index=["train", "validation", "test"],
            )
            display(split_summary.style.format({"% malignos": "{:.1%}"}))
            """
        )
    )

    cells.append(markdown("## 4. Escalar sin fuga de información"))
    cells.append(
        callout(
            "mistake",
            "es",
            "El error más común: escalar antes de dividir",
            """
            Si ajustas `StandardScaler` con **todo** el dataset y luego divides,
            la media y desviación de train ya "vieron" ejemplos de validation y
            test. Es una fuga de información sutil: el modelo no copia
            respuestas, pero su preprocesado sí se benefició del examen final.
            La regla es siempre: `fit_transform` solo en train, `transform` en
            el resto.
            """,
        )
    )
    cells.append(
        code(
            """
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype("float32")
            X_val_scaled = scaler.transform(X_val).astype("float32")
            X_test_scaled = scaler.transform(X_test).astype("float32")

            print("Media aproximada de train:", X_train_scaled.mean(axis=0)[:3].round(5))
            print("Forma que recibirá la red:", X_train_scaled.shape)
            """
        )
    )

    cells.append(milestone("es", 0))

    cells.append(markdown("## 5. Crear un baseline honesto"))
    cells.append(
        markdown(
            """
            Una regresión logística responde a la misma pregunta y es mucho más
            simple. Si rinde igual o mejor que la red, la conclusión profesional
            no es "la ANN falló": es "la complejidad adicional no se justificó
            con estos datos".
            """
        )
    )
    cells.append(
        code(
            """
            baseline = LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)
            baseline.fit(X_train_scaled, y_train)

            baseline_probabilities = baseline.predict_proba(X_test_scaled)[:, 1]
            print("ROC-AUC baseline:", round(roc_auc_score(y_test, baseline_probabilities), 3))
            """
        )
    )

    cells.append(markdown("## 6. Epochs, batches e iteraciones"))
    cells.append(
        markdown(
            """
            Ya usaste el vocabulario "batch"/"step" en el Notebook 3 al hablar de
            optimizadores. Aquí solo lo aterrizamos en números concretos para
            **este** split: con `X_train_scaled` de tamaño fijo y un
            `batch_size` dado, ¿cuántas actualizaciones de pesos ocurren por
            época?
            """
        )
    )
    cells.append(
        code(
            """
            BATCH_SIZE = 32
            EPOCHS = 200
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Completa el cálculo de actualizaciones por época **usando el tamaño
            real de `X_train_scaled`**, no un número fijo. Recuerda: una
            actualización ocurre por cada batch procesado, y el último batch de
            la época puede ser más pequeño (por eso se redondea hacia arriba
            con `math.ceil`).
            """,
            starter_code="""
            updates_per_epoch = math.ceil(✏️✏️✏️)
            max_updates = updates_per_epoch * EPOCHS

            print("Actualizaciones por época:", updates_per_epoch)
            print("Actualizaciones máximas (si se completan todas las épocas):", max_updates)
            """,
            solution_code="""
            updates_per_epoch = math.ceil(len(X_train_scaled) / BATCH_SIZE)
            max_updates = updates_per_epoch * EPOCHS

            print("Actualizaciones por época:", updates_per_epoch)
            print("Actualizaciones máximas (si se completan todas las épocas):", max_updates)
            """,
        )
    )

    cells.append(markdown("## 7. Construir la red"))
    cells.append(
        markdown(
            """
            `30 variables → Dense(32, ReLU) → Dropout(0.30) → Dense(16, ReLU) → Dense(1, Sigmoid)`

            El porqué de 32 y 16 neuronas (hiperparámetros que se validan, no
            fórmulas del número de variables) ya lo viste en el Notebook 1. Lo
            único nuevo aquí es la capa de salida: una neurona con activación
            **Sigmoid** produce la probabilidad que usamos como predicción.
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Completa la activación de la capa de salida. Pista: necesitamos un
            único número entre 0 y 1 interpretable como probabilidad de
            "maligno" — la misma función que usaste en el Notebook 1 para la
            capa de salida binaria.
            """,
            starter_code="""
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dropout(0.30),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="✏️✏️✏️"),
                ],
                name="neurotrain_mlp",
            )
            model.summary()
            """,
            solution_code="""
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dropout(0.30),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ],
                name="neurotrain_mlp",
            )
            model.summary()
            """,
        )
    )

    cells.append(markdown("## 8. Compilar: optimizer, loss y métricas"))
    cells.append(
        markdown(
            """
            `compile()` todavía no entrena, solo configura las reglas. El
            optimizer (Adam) lo estudiaste a fondo en el Notebook 3; la loss
            (binary cross-entropy) la estudiaste a fondo en el Notebook 2. Aquí
            solo los conectamos con métricas que sí observamos pero que no
            sustituyen a la loss.
            """
        )
    )
    cells.append(
        code(
            """
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="roc_auc"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="sensitivity"),
                ],
            )
            """
        )
    )

    cells.append(markdown("## 9. EarlyStopping y Dropout"))
    cells.append(
        callout(
            "concept",
            "es",
            "Dropout: forzar redundancia",
            """
            `Dropout(0.30)` apaga al azar el 30% de las neuronas de esa capa en
            **cada paso de entrenamiento**. Obliga a la red a no depender
            siempre de las mismas rutas internas — el equivalente a estudiar
            sin memorizar el orden exacto de las preguntas. En inferencia
            (predicción real) no se apaga ninguna neurona.
            """,
        )
    )
    cells.append(
        callout(
            "concept",
            "es",
            "EarlyStopping: dejar de entrenar en el momento justo",
            """
            Observa `val_loss` época a época. Si no mejora durante `patience`
            épocas consecutivas, detiene el entrenamiento.
            `restore_best_weights=True` recupera los pesos de la **mejor**
            época observada, no los de la última — así un empeoramiento tardío
            no se queda como resultado final.
            """,
        )
    )
    cells.append(
        callout(
            "doubt",
            "es",
            "¿Validation también entrena a la red?",
            """
            No. `validation_data` se evalúa al final de cada época solo para
            **medir**; sus ejemplos nunca participan en el cálculo del
            gradiente ni actualizan pesos. Por eso puede usarse para decidir
            cuándo parar sin "hacer trampa".
            """,
        )
    )
    cells.append(
        code(
            """
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            )
            """
        )
    )

    cells.append(markdown("## 10. Entrenar con `fit()`"))
    cells.append(
        markdown(
            """
            En cada batch ocurren, en una línea, los cuatro pasos que ya
            diseccionaste en los Notebooks 2 y 3: forward pass → loss →
            backpropagation → paso del optimizer.
            """
        )
    )
    cells.append(
        code(
            """
            history = model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping],
                verbose=0,
            )

            print(f"Épocas ejecutadas: {len(history.history['loss'])} de {EPOCHS}")
            """
        )
    )

    cells.append(markdown("## 11. Leer las curvas de aprendizaje"))
    cells.append(
        markdown(
            """
            - Si `loss` y `val_loss` bajan juntas, la red aprende patrones que
              generalizan.
            - Si `loss` sigue bajando mientras `val_loss` sube, la red está
              **memorizando** train: sobreajuste.
            - Si ambas quedan altas, puede haber infraajuste, pocas épocas o
              una configuración inadecuada.

            No mires solo `accuracy`: con clases desbalanceadas puede parecer
            buena aunque el modelo falle justo en la clase que importa.
            """
        )
    )
    cells.append(
        code(
            """
            history_dict = {key: list(values) for key, values in history.history.items()}
            fig = plot_training_history(history_dict, lang="es")
            plt.show()
            """
        )
    )

    cells.append(markdown("## 12. Evaluar una sola vez en test, frente al baseline"))
    cells.append(
        markdown(
            """
            Umbral inicial 0.50: probabilidad ≥ 0.50 se convierte en "maligno"
            (1). Sensibilidad, especificidad, precisión y ROC-AUC ya las
            calculó `classification_metrics` — la misma lógica que escribirías
            a mano, empaquetada para no repetirla en cada notebook.
            """
        )
    )
    cells.append(
        code(
            """
            THRESHOLD = 0.50
            probabilities = model.predict(X_test_scaled, verbose=0).ravel()

            ann_metrics = classification_metrics(y_test, probabilities, THRESHOLD)
            baseline_metrics = classification_metrics(y_test, baseline_probabilities, THRESHOLD)

            comparison = pd.DataFrame(
                {"Red (MLP)": ann_metrics, "Baseline (LogReg)": baseline_metrics}
            ).loc[["accuracy", "roc_auc", "precision", "sensitivity", "specificity", "f1"]]
            display(comparison.style.format("{:.3f}"))
            """
        )
    )
    cells.append(
        code(
            """
            fig_confusion = plot_confusion(y_test, probabilities, threshold=THRESHOLD, lang="es")
            plt.show()

            fig_roc = plot_roc(y_test, probabilities, lang="es")
            plt.show()
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "es",
            "Ganar al baseline no es opcional para justificar la red",
            """
            Si la red no supera de forma clara a la regresión logística, la
            decisión profesional correcta suele ser **usar el modelo más
            simple**: es más barato de entrenar, más fácil de explicar y menos
            propenso a sobreajustar con pocos datos.
            """,
        )
    )

    cells.append(markdown("## 13. Experimento guiado de sobreajuste (A vs B)"))
    cells.append(
        markdown(
            """
            En vez de reeditar las celdas anteriores (arriesgando perder tu
            referencia), vamos a lanzar **dos configuraciones independientes**
            que coexisten, usando `TrainingConfig`:

            | Variante | Capas | Dropout | EarlyStopping | Hipótesis |
            |---|---:|---:|---:|---|
            | A | 128 → 64 | 0.0 | No | Train mejorará; validation puede empeorar |
            | B | 32 → 16 | 0.30 | Sí | Menos capacidad de memorizar, parada más temprana |

            La variante B es, de hecho, la misma arquitectura que acabas de
            entrenar a mano en las Secciones 7-10 — aquí la reproducimos vía
            `TrainingConfig` para que quede lado a lado con A.

            Antes de ejecutar, **escribe tu predicción** en una celda de texto
            propia: ¿cuál crees que tendrá menor `val_loss` mínima?
            """
        )
    )
    cells.append(
        markdown(
            """
            `load_dataset()` y `prepare_data()` son la **misma lógica** que
            escribiste a mano en las Secciones 1, 3 y 4 (auditoría, split
            estratificado, `StandardScaler` ajustado solo con train) —
            empaquetada para que un proyecto real no la repita en cada
            experimento.
            """
        )
    )
    cells.append(
        code(
            """
            frame = load_dataset()
            data = prepare_data(frame, random_state=RANDOM_STATE)
            print("Train:", data.X_train.shape, "| Val:", data.X_val.shape, "| Test:", data.X_test.shape)
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Completa `config_a` siguiendo la hipótesis de la tabla: sin
            Dropout y sin EarlyStopping, para que la red pueda memorizar train
            libremente durante las 200 épocas.
            """,
            starter_code="""
            config_a = TrainingConfig(
                hidden_units=(128, 64),
                dropout_rate=✏️✏️✏️,
                use_early_stopping=✏️✏️✏️,
                epochs=200,
                batch_size=32,
                random_state=RANDOM_STATE,
            )
            model_a, history_a = train_dense_classifier(data, config_a)
            print("Épocas ejecutadas (A):", len(history_a["loss"]))
            """,
            solution_code="""
            config_a = TrainingConfig(
                hidden_units=(128, 64),
                dropout_rate=0.0,
                use_early_stopping=False,
                epochs=200,
                batch_size=32,
                random_state=RANDOM_STATE,
            )
            model_a, history_a = train_dense_classifier(data, config_a)
            print("Épocas ejecutadas (A):", len(history_a["loss"]))
            """,
        )
    )
    cells.append(
        code(
            """
            config_b = TrainingConfig(
                hidden_units=(32, 16),
                dropout_rate=0.30,
                use_early_stopping=True,
                patience=12,
                epochs=200,
                batch_size=32,
                random_state=RANDOM_STATE,
            )
            model_b, history_b = train_dense_classifier(data, config_b)
            print("Épocas ejecutadas (B):", len(history_b["loss"]))
            """
        )
    )

    cells.append(milestone("es", 3))

    cells.append(
        code(
            """
            fig_comparison = plot_training_history_comparison(
                history_a,
                history_b,
                "A: 128→64, sin regularización",
                "B: 32→16, con Dropout+EarlyStopping",
                lang="es",
            )
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            Responde con la gráfica delante:

            1. ¿En qué época fue mínima `val_loss` en cada variante?
            2. ¿Cuánto se separaron `loss` y `val_loss` en A? ¿Y en B?
            3. ¿La red más grande (A) mejoró el resultado en test, o solo en train?
            4. ¿Alguna de las dos variantes justificó ser más compleja que el baseline de la Sección 5?
            """
        )
    )

    cells.append(markdown("## 14. Guardar el modelo y el preprocesado"))
    cells.append(
        markdown(
            """
            Un modelo sin su scaler no reproduce el mismo flujo: guardamos
            ambos y metadatos mínimos del experimento de referencia (la
            variante B, que fue la que evaluamos en test en la Sección 12).
            """
        )
    )
    cells.append(
        code(
            """
            ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
            ARTIFACTS_DIR.mkdir(exist_ok=True)

            model.save(ARTIFACTS_DIR / "neurotrain_model.keras")
            joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

            metadata = {
                "dataset": "UCI Breast Cancer Wisconsin Diagnostic",
                "positive_class": "M = 1",
                "feature_names": X.columns.tolist(),
                "threshold": THRESHOLD,
                "epochs_executed": len(history.history["loss"]),
                "test_metrics": ann_metrics,
                "intended_use": "educational demonstration only",
            }
            (ARTIFACTS_DIR / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print("Artefactos guardados en:", ARTIFACTS_DIR)
            """
        )
    )

    cells.append(markdown("## 15. Del notebook al producto"))
    cells.append(
        markdown(
            """
            Todo lo que hiciste en estos 4 notebooks vive también en una app
            Streamlit con un **recorrido guiado**: una página "Inicio" y una
            página por tema (Perceptrón, Pérdida y backprop, Optimizadores,
            Entrenamiento — este mismo notebook, resumido de forma visual).

            Además tiene un **"Modo Experimento"**: una página de laboratorio
            donde puedes reentrenar de forma interactiva — cambiando
            arquitectura, dropout, épocas, paciencia y umbral — con progreso
            en vivo por época y un panel que revela el código real detrás de
            cada botón.

            ```powershell
            streamlit run app.py
            ```

            **Siguiente paso:** abre la app, reproduce las variantes A y B
            desde el Modo Experimento, y explica en voz alta qué cambió. Si
            puedes justificar el resultado sin releer este notebook, ya
            dominas el núcleo de la masterclass.
            """
        )
    )

    cells.append(section_header("es"))
    cells.append(
        quiz_question(
            "es", 1,
            "¿Por qué se separa un conjunto de validation además de train y test?",
            [
                "Porque el modelo necesita más datos para aprender",
                "Para decidir arquitectura, dropout, paciencia o umbral sin tocar test",
                "Porque test siempre debe ser más grande que train",
                "Validation y test son el mismo conjunto con otro nombre",
            ],
            1,
            "Validation guía decisiones humanas de configuración durante el desarrollo; test se abre una sola vez, al final, para no contaminar la medición.",
        )
    )
    cells.append(
        quiz_question(
            "es", 2,
            "¿Con qué datos debe ajustarse (`fit`) el `StandardScaler`?",
            ["Con todo el dataset, antes de dividir", "Solo con train", "Solo con test",
             "Con train y validation juntos, pero nunca con test"],
            1,
            "Ajustar el scaler con datos fuera de train filtra información del examen final al preprocesado, aunque el modelo nunca 'vea' esas etiquetas directamente.",
        )
    )
    cells.append(
        quiz_question(
            "es", 3,
            "¿Qué hace `restore_best_weights=True` en `EarlyStopping`?",
            [
                "Reinicia los pesos a valores aleatorios al terminar",
                "Guarda los pesos de la última época, sea buena o mala",
                "Recupera los pesos de la época con mejor `val_loss` observada",
                "Congela los pesos de la primera época como referencia",
            ],
            2,
            "Sin esta opción, el modelo se quedaría con los pesos de la última época entrenada, que puede ser peor que una anterior si ya venía empeorando.",
        )
    )
    cells.append(
        quiz_question(
            "es", 4,
            "En el experimento A/B, la variante A (128→64, sin Dropout, sin EarlyStopping) muestra `loss` de train bajando mucho mientras `val_loss` sube tras cierta época. ¿Qué está pasando?",
            ["Infraajuste", "Sobreajuste: la red memoriza train y deja de generalizar",
             "Un error en el código, esa combinación no debería ocurrir", "El learning rate es demasiado bajo"],
            1,
            "Es la firma clásica del sobreajuste: la red sigue reduciendo el error en los datos que ve, pero empeora en datos nuevos porque memorizó detalles de train.",
        )
    )
    cells.append(
        quiz_question(
            "es", 5,
            "¿Qué optimizer y qué loss se usaron para compilar la red en este notebook, y por qué (según lo visto en Notebooks 2 y 3)?",
            [
                "SGD puro y MSE, porque son los más simples",
                "Adam y binary cross-entropy, porque Adam adapta el learning rate por parámetro y BCE penaliza probabilidades mal calibradas en clasificación binaria",
                "Momentum y categorical cross-entropy, porque hay más de dos clases",
                "Adam y accuracy, porque accuracy es la métrica que de verdad se minimiza",
            ],
            1,
            "Adam (NB3) combina momentum y tasas de aprendizaje adaptativas por parámetro; binary cross-entropy (NB2) es la loss correcta para clasificación binaria con salida Sigmoid. Accuracy es una métrica de observación, no la función que se minimiza.",
        )
    )
    cells.append(
        open_question(
            "es", 6,
            "Un modelo alcanza 99% de accuracy en train y 71% en validation. ¿Qué está pasando y qué probarías primero?",
            [
                "Nombra el fenómeno: sobreajuste (la red memoriza train, no generaliza).",
                "Propone reducir capacidad (menos neuronas/capas) o subir Dropout.",
                "Propone activar o endurecer EarlyStopping (menor patience) para no seguir entrenando tras el punto de mejor val_loss.",
                "Menciona que también podría deberse a muy pocos datos de train para la complejidad del modelo.",
            ],
        )
    )
    cells.append(
        open_question(
            "es", 7,
            "Explica a alguien no técnico por qué comparamos la red neuronal contra una regresión logística simple en vez de confiar directamente en la red.",
            [
                "Deja claro que un modelo complejo no es automáticamente mejor; hay que demostrarlo con datos.",
                "Menciona que el baseline es más barato, más rápido de entrenar y más fácil de explicar a terceros.",
                "Explica que si el baseline empata o gana, la conclusión profesional es usar el modelo simple, no forzar la red.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 ¡Enhorabuena! Completaste los 4 notebooks de NeuroTrain Lab 🎉",
                "Del perceptrón al entrenamiento con control de sobreajuste: ya conoces "
                "todo el camino. Ahora abre la app Streamlit y pon a prueba tus variantes "
                "A y B en el Modo Experimento.",
            )
            """
        )
    )

    return cells


def build_en_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        markdown(
            """
            # NeuroTrain Lab — Notebook 4: Training and Overfitting

            **Topic:** how to organize a real training run end to end — a
            train/validation/test split, scaling without information leakage,
            an honest baseline, `Dropout` and `EarlyStopping` — and how to spot
            overfitting by reading the learning curves.

            > Last of 4 notebooks. You already know the perceptron (NB1), loss
            > and backprop (NB2), and optimizers (NB3). Here we put it all
            > together to train a real model on a real problem and learn to
            > **control** overfitting, not just observe it.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 What you'll learn in this notebook

            By the end you should be able to explain, without memorized formulas:

            1. Why we split into three sets (train/validation/test), not two.
            2. Why scaling is fit only on train.
            3. What `Dropout` and `EarlyStopping` do, and why they're combined.
            4. How to read `loss` and `val_loss` to diagnose overfitting.
            5. Why we always compare against a simpler baseline.

            **Mental map:** `real data → split → scaling → baseline → MLP → EarlyStopping+Dropout → fit() → curves → test vs baseline → A/B experiment`
            """
        )
    )
    cells.append(
        code(
            """
            from pathlib import Path
            import json
            import math
            import sys

            import joblib
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate
            from neurotrain.config import TrainingConfig
            from neurotrain.data import load_dataset, prepare_data
            from neurotrain.evaluation import classification_metrics
            from neurotrain.modeling import train_dense_classifier
            from neurotrain.visualization import (
                plot_confusion,
                plot_roc,
                plot_training_history,
                plot_training_history_comparison,
            )

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("TensorFlow:", tf.__version__)
            print("Project root:", PROJECT_ROOT)
            """
        )
    )

    cells.append(markdown("## 1. Quick dataset audit"))
    cells.append(
        markdown(
            """
            You already glanced at this CSV in Notebook 1. Before training we
            confirm its minimal contract with `assert`: shape, no missing
            values, and the two expected labels. This is the last time we do it
            "by hand" — the rest of the project uses
            `neurotrain.data.load_dataset()`, which runs exactly these checks.
            """
        )
    )
    cells.append(
        code(
            """
            DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
            df = pd.read_csv(DATA_PATH)

            assert df.shape == (569, 31)
            assert not df.isna().any().any()
            assert set(df["diagnosis"].unique()) == {"B", "M"}

            class_counts = df["diagnosis"].value_counts().rename(index={"B": "Benign", "M": "Malignant"})
            print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            display(class_counts.to_frame("records"))
            """
        )
    )

    cells.append(markdown("## 2. Split X and y"))
    cells.append(
        markdown(
            """
            The positive class is explicitly defined as **malignant = 1** — so
            "sensitivity" unambiguously means "proportion of true malignant
            cases we detect."
            """
        )
    )
    cells.append(
        code(
            """
            X = df.drop(columns="diagnosis")
            y = df["diagnosis"].eq("M").astype("int8")

            print("X shape:", X.shape, "| y shape:", y.shape)
            """
        )
    )

    cells.append(markdown("## 3. Creating train, validation, and test"))
    cells.append(
        callout(
            "concept",
            "en",
            "Why three sets, not two?",
            """
            `train` is the practice set the network adjusts its weights on.
            `validation` is a mock exam: the network never learns from it
            directly, but **we** use it to decide architecture, dropout,
            patience, or threshold. `test` is the final exam, opened only
            **once**, at the very end. If you use test to decide anything, it
            stops measuring what it's supposed to measure — it just measures
            how well you overfit to the exam.
            """,
        )
    )
    cells.append(
        markdown(
            """
            We'll use roughly 70% / 15% / 15%. `stratify=y` keeps a similar
            proportion of benign and malignant cases in every split.
            """
        )
    )
    cells.append(
        code(
            """
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE,
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE,
            )

            split_summary = pd.DataFrame(
                {
                    "records": [len(X_train), len(X_val), len(X_test)],
                    "% malignant": [y_train.mean(), y_val.mean(), y_test.mean()],
                },
                index=["train", "validation", "test"],
            )
            display(split_summary.style.format({"% malignant": "{:.1%}"}))
            """
        )
    )

    cells.append(markdown("## 4. Scaling without information leakage"))
    cells.append(
        callout(
            "mistake",
            "en",
            "The most common mistake: scaling before splitting",
            """
            If you fit `StandardScaler` on the **whole** dataset and split
            afterwards, train's mean and standard deviation already "saw"
            validation and test examples. It's a subtle leak: the model
            doesn't copy answers, but its preprocessing already benefited from
            the final exam. The rule is always: `fit_transform` only on train,
            `transform` on the rest.
            """,
        )
    )
    cells.append(
        code(
            """
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train).astype("float32")
            X_val_scaled = scaler.transform(X_val).astype("float32")
            X_test_scaled = scaler.transform(X_test).astype("float32")

            print("Approx. train mean:", X_train_scaled.mean(axis=0)[:3].round(5))
            print("Shape the network will receive:", X_train_scaled.shape)
            """
        )
    )

    cells.append(milestone("en", 0))

    cells.append(markdown("## 5. Creating an honest baseline"))
    cells.append(
        markdown(
            """
            A logistic regression answers the same question and is much
            simpler. If it performs as well as or better than the network, the
            professional conclusion isn't "the ANN failed": it's "the extra
            complexity wasn't justified by this data."
            """
        )
    )
    cells.append(
        code(
            """
            baseline = LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)
            baseline.fit(X_train_scaled, y_train)

            baseline_probabilities = baseline.predict_proba(X_test_scaled)[:, 1]
            print("Baseline ROC-AUC:", round(roc_auc_score(y_test, baseline_probabilities), 3))
            """
        )
    )

    cells.append(markdown("## 6. Epochs, batches, and iterations"))
    cells.append(
        markdown(
            """
            You already used the "batch"/"step" vocabulary in Notebook 3 when
            discussing optimizers. Here we just ground it in concrete numbers
            for **this** split: given a fixed-size `X_train_scaled` and a
            `batch_size`, how many weight updates happen per epoch?
            """
        )
    )
    cells.append(
        code(
            """
            BATCH_SIZE = 32
            EPOCHS = 200
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Complete the updates-per-epoch calculation **using the actual size
            of `X_train_scaled`**, not a fixed number. Remember: one update
            happens per batch processed, and the epoch's last batch can be
            smaller (that's why we round up with `math.ceil`).
            """,
            starter_code="""
            updates_per_epoch = math.ceil(✏️✏️✏️)
            max_updates = updates_per_epoch * EPOCHS

            print("Updates per epoch:", updates_per_epoch)
            print("Max updates (if all epochs run):", max_updates)
            """,
            solution_code="""
            updates_per_epoch = math.ceil(len(X_train_scaled) / BATCH_SIZE)
            max_updates = updates_per_epoch * EPOCHS

            print("Updates per epoch:", updates_per_epoch)
            print("Max updates (if all epochs run):", max_updates)
            """,
        )
    )

    cells.append(markdown("## 7. Building the network"))
    cells.append(
        markdown(
            """
            `30 features → Dense(32, ReLU) → Dropout(0.30) → Dense(16, ReLU) → Dense(1, Sigmoid)`

            Why 32 and 16 neurons (hyperparameters to validate, not formulas
            derived from the feature count) was already covered in Notebook 1.
            The only new part here is the output layer: a single neuron with a
            **Sigmoid** activation produces the probability we use as a
            prediction.
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Complete the output layer's activation. Hint: we need a single
            number between 0 and 1, interpretable as the probability of
            "malignant" — the same function you used in Notebook 1 for the
            binary output layer.
            """,
            starter_code="""
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dropout(0.30),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="✏️✏️✏️"),
                ],
                name="neurotrain_mlp",
            )
            model.summary()
            """,
            solution_code="""
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
                    tf.keras.layers.Dense(32, activation="relu"),
                    tf.keras.layers.Dropout(0.30),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ],
                name="neurotrain_mlp",
            )
            model.summary()
            """,
        )
    )

    cells.append(markdown("## 8. Compiling: optimizer, loss, and metrics"))
    cells.append(
        markdown(
            """
            `compile()` doesn't train yet, it only configures the rules. You
            studied the optimizer (Adam) in depth in Notebook 3; you studied
            the loss (binary cross-entropy) in depth in Notebook 2. Here we
            just wire them together with metrics we observe but that never
            replace the loss.
            """
        )
    )
    cells.append(
        code(
            """
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
                metrics=[
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="roc_auc"),
                    tf.keras.metrics.Precision(name="precision"),
                    tf.keras.metrics.Recall(name="sensitivity"),
                ],
            )
            """
        )
    )

    cells.append(markdown("## 9. EarlyStopping and Dropout"))
    cells.append(
        callout(
            "concept",
            "en",
            "Dropout: forcing redundancy",
            """
            `Dropout(0.30)` randomly switches off 30% of that layer's neurons
            on **every training step**. It forces the network to not always
            rely on the same internal paths — like studying without
            memorizing the exact order of the questions. At inference
            (real prediction) no neuron is switched off.
            """,
        )
    )
    cells.append(
        callout(
            "concept",
            "en",
            "EarlyStopping: stopping at just the right moment",
            """
            Watches `val_loss` epoch by epoch. If it doesn't improve for
            `patience` consecutive epochs, training stops.
            `restore_best_weights=True` recovers the weights from the
            **best** observed epoch, not the last one — so a late-training
            regression doesn't become the final result.
            """,
        )
    )
    cells.append(
        callout(
            "doubt",
            "en",
            "Does validation train the network too?",
            """
            No. `validation_data` is evaluated at the end of every epoch only
            to **measure**; its examples never participate in the gradient
            computation or update weights. That's why it can be used to decide
            when to stop without "cheating."
            """,
        )
    )
    cells.append(
        code(
            """
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
                verbose=1,
            )
            """
        )
    )

    cells.append(markdown("## 10. Training with `fit()`"))
    cells.append(
        markdown(
            """
            In every batch, the four steps you already dissected in Notebooks
            2 and 3 happen in one line: forward pass → loss → backpropagation
            → optimizer step.
            """
        )
    )
    cells.append(
        code(
            """
            history = model.fit(
                X_train_scaled,
                y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=[early_stopping],
                verbose=0,
            )

            print(f"Epochs run: {len(history.history['loss'])} of {EPOCHS}")
            """
        )
    )

    cells.append(markdown("## 11. Reading the learning curves"))
    cells.append(
        markdown(
            """
            - If `loss` and `val_loss` go down together, the network is
              learning patterns that generalize.
            - If `loss` keeps dropping while `val_loss` rises, the network is
              **memorizing** train: overfitting.
            - If both stay high, there may be underfitting, too few epochs, or
              an unsuitable configuration.

            Don't look only at `accuracy`: with imbalanced classes it can look
            good even while the model fails exactly the class that matters.
            """
        )
    )
    cells.append(
        code(
            """
            history_dict = {key: list(values) for key, values in history.history.items()}
            fig = plot_training_history(history_dict, lang="en")
            plt.show()
            """
        )
    )

    cells.append(markdown("## 12. Evaluating once on test, against the baseline"))
    cells.append(
        markdown(
            """
            Initial threshold of 0.50: probability ≥ 0.50 becomes "malignant"
            (1). Sensitivity, specificity, precision, and ROC-AUC are computed
            by `classification_metrics` — the same logic you'd write by hand,
            packaged so it isn't repeated in every notebook.
            """
        )
    )
    cells.append(
        code(
            """
            THRESHOLD = 0.50
            probabilities = model.predict(X_test_scaled, verbose=0).ravel()

            ann_metrics = classification_metrics(y_test, probabilities, THRESHOLD)
            baseline_metrics = classification_metrics(y_test, baseline_probabilities, THRESHOLD)

            comparison = pd.DataFrame(
                {"Network (MLP)": ann_metrics, "Baseline (LogReg)": baseline_metrics}
            ).loc[["accuracy", "roc_auc", "precision", "sensitivity", "specificity", "f1"]]
            display(comparison.style.format("{:.3f}"))
            """
        )
    )
    cells.append(
        code(
            """
            fig_confusion = plot_confusion(y_test, probabilities, threshold=THRESHOLD, lang="en")
            plt.show()

            fig_roc = plot_roc(y_test, probabilities, lang="en")
            plt.show()
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "en",
            "Beating the baseline isn't optional to justify the network",
            """
            If the network doesn't clearly beat the logistic regression, the
            correct professional call is usually to **use the simpler model**:
            it's cheaper to train, easier to explain, and less prone to
            overfitting with limited data.
            """,
        )
    )

    cells.append(markdown("## 13. Guided overfitting experiment (A vs B)"))
    cells.append(
        markdown(
            """
            Instead of re-editing the cells above (risking losing your
            reference run), we'll launch **two independent configurations**
            that coexist, using `TrainingConfig`:

            | Variant | Layers | Dropout | EarlyStopping | Hypothesis |
            |---|---:|---:|---:|---|
            | A | 128 → 64 | 0.0 | No | Train will improve; validation might get worse |
            | B | 32 → 16 | 0.30 | Yes | Less capacity to memorize, earlier stopping |

            Variant B is actually the same architecture you just trained by
            hand in Sections 7-10 — here we reproduce it via `TrainingConfig`
            so it sits side by side with A.

            Before running, **write your prediction** in your own text cell:
            which one do you think will have the lower minimum `val_loss`?
            """
        )
    )
    cells.append(
        markdown(
            """
            `load_dataset()` and `prepare_data()` are the **same logic** you
            wrote by hand in Sections 1, 3, and 4 (audit, stratified split,
            `StandardScaler` fit only on train) — packaged so a real project
            doesn't repeat it in every experiment.
            """
        )
    )
    cells.append(
        code(
            """
            frame = load_dataset()
            data = prepare_data(frame, random_state=RANDOM_STATE)
            print("Train:", data.X_train.shape, "| Val:", data.X_val.shape, "| Test:", data.X_test.shape)
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Complete `config_a` following the table's hypothesis: no Dropout
            and no EarlyStopping, so the network can memorize train freely
            over the 200 epochs.
            """,
            starter_code="""
            config_a = TrainingConfig(
                hidden_units=(128, 64),
                dropout_rate=✏️✏️✏️,
                use_early_stopping=✏️✏️✏️,
                epochs=200,
                batch_size=32,
                random_state=RANDOM_STATE,
            )
            model_a, history_a = train_dense_classifier(data, config_a)
            print("Epochs run (A):", len(history_a["loss"]))
            """,
            solution_code="""
            config_a = TrainingConfig(
                hidden_units=(128, 64),
                dropout_rate=0.0,
                use_early_stopping=False,
                epochs=200,
                batch_size=32,
                random_state=RANDOM_STATE,
            )
            model_a, history_a = train_dense_classifier(data, config_a)
            print("Epochs run (A):", len(history_a["loss"]))
            """,
        )
    )
    cells.append(
        code(
            """
            config_b = TrainingConfig(
                hidden_units=(32, 16),
                dropout_rate=0.30,
                use_early_stopping=True,
                patience=12,
                epochs=200,
                batch_size=32,
                random_state=RANDOM_STATE,
            )
            model_b, history_b = train_dense_classifier(data, config_b)
            print("Epochs run (B):", len(history_b["loss"]))
            """
        )
    )

    cells.append(milestone("en", 3))

    cells.append(
        code(
            """
            fig_comparison = plot_training_history_comparison(
                history_a,
                history_b,
                "A: 128->64, no regularization",
                "B: 32->16, with Dropout+EarlyStopping",
                lang="en",
            )
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            Answer with the plot in front of you:

            1. At which epoch was `val_loss` minimal for each variant?
            2. How far apart were `loss` and `val_loss` in A? And in B?
            3. Did the bigger network (A) improve the test result, or only train?
            4. Did either variant justify being more complex than the Section 5 baseline?
            """
        )
    )

    cells.append(markdown("## 14. Saving the model and preprocessing"))
    cells.append(
        markdown(
            """
            A model without its scaler doesn't reproduce the same pipeline: we
            save both, plus minimal metadata for the reference experiment
            (variant B, which we evaluated on test in Section 12).
            """
        )
    )
    cells.append(
        code(
            """
            ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
            ARTIFACTS_DIR.mkdir(exist_ok=True)

            model.save(ARTIFACTS_DIR / "neurotrain_model.keras")
            joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")

            metadata = {
                "dataset": "UCI Breast Cancer Wisconsin Diagnostic",
                "positive_class": "M = 1",
                "feature_names": X.columns.tolist(),
                "threshold": THRESHOLD,
                "epochs_executed": len(history.history["loss"]),
                "test_metrics": ann_metrics,
                "intended_use": "educational demonstration only",
            }
            (ARTIFACTS_DIR / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            print("Artifacts saved to:", ARTIFACTS_DIR)
            """
        )
    )

    cells.append(markdown("## 15. From notebook to product"))
    cells.append(
        markdown(
            """
            Everything you did across these 4 notebooks also lives in a
            Streamlit app with a **guided journey**: a "Home" page and one
            page per topic (Perceptron, Loss & backprop, Optimizers, Training —
            this same notebook, summarized visually).

            It also has an **"Experiment Mode"**: a lab page where you can
            retrain interactively — changing architecture, dropout, epochs,
            patience, and threshold — with live per-epoch progress and a panel
            that reveals the real code behind each button.

            ```powershell
            streamlit run app.py
            ```

            **Next step:** open the app, reproduce variants A and B from
            Experiment Mode, and explain out loud what changed. If you can
            justify the result without rereading this notebook, you've
            mastered the core of the masterclass.
            """
        )
    )

    cells.append(section_header("en"))
    cells.append(
        quiz_question(
            "en", 1,
            "Why do we split off a validation set in addition to train and test?",
            [
                "Because the model needs more data to learn",
                "To decide architecture, dropout, patience, or threshold without touching test",
                "Because test must always be larger than train",
                "Validation and test are the same set under a different name",
            ],
            1,
            "Validation guides human configuration decisions during development; test is opened only once, at the end, so the measurement isn't contaminated.",
        )
    )
    cells.append(
        quiz_question(
            "en", 2,
            "What data should `StandardScaler` be fit on?",
            ["The whole dataset, before splitting", "Only train", "Only test",
             "Train and validation together, but never test"],
            1,
            "Fitting the scaler on data outside train leaks final-exam information into preprocessing, even though the model never directly 'sees' those labels.",
        )
    )
    cells.append(
        quiz_question(
            "en", 3,
            "What does `restore_best_weights=True` do in `EarlyStopping`?",
            [
                "Resets the weights to random values at the end",
                "Keeps the weights from the last epoch, good or bad",
                "Recovers the weights from the epoch with the best observed `val_loss`",
                "Freezes the first epoch's weights as a reference",
            ],
            2,
            "Without this option, the model would keep the last trained epoch's weights, which can be worse than an earlier epoch if it was already regressing.",
        )
    )
    cells.append(
        quiz_question(
            "en", 4,
            "In the A/B experiment, variant A (128->64, no Dropout, no EarlyStopping) shows train `loss` dropping a lot while `val_loss` rises past a certain epoch. What's happening?",
            ["Underfitting", "Overfitting: the network memorizes train and stops generalizing",
             "A code bug, that combination shouldn't happen", "The learning rate is too low"],
            1,
            "This is the classic overfitting signature: the network keeps reducing error on data it sees, but gets worse on new data because it memorized train details.",
        )
    )
    cells.append(
        quiz_question(
            "en", 5,
            "Which optimizer and loss did we use to compile the network in this notebook, and why (per Notebooks 2 and 3)?",
            [
                "Plain SGD and MSE, because they're the simplest",
                "Adam and binary cross-entropy, because Adam adapts the learning rate per-parameter and BCE penalizes miscalibrated probabilities in binary classification",
                "Momentum and categorical cross-entropy, because there are more than two classes",
                "Adam and accuracy, because accuracy is what's actually minimized",
            ],
            1,
            "Adam (NB3) combines momentum with per-parameter adaptive learning rates; binary cross-entropy (NB2) is the correct loss for binary classification with a Sigmoid output. Accuracy is a monitoring metric, not the minimized function.",
        )
    )
    cells.append(
        open_question(
            "en", 6,
            "A model reaches 99% train accuracy and 71% validation accuracy. What's happening, and what would you try first?",
            [
                "Names the phenomenon: overfitting (the network memorizes train, doesn't generalize).",
                "Proposes reducing capacity (fewer neurons/layers) or raising Dropout.",
                "Proposes enabling or tightening EarlyStopping (lower patience) to stop training past the best-val_loss point.",
                "Mentions this could also stem from too little train data for the model's complexity.",
            ],
        )
    )
    cells.append(
        open_question(
            "en", 7,
            "Explain to a non-technical stakeholder why we compare the neural network against a plain logistic regression instead of trusting the network outright.",
            [
                "Makes clear a complex model isn't automatically better; it has to be proven with data.",
                "Mentions the baseline is cheaper, faster to train, and easier to explain to others.",
                "Explains that if the baseline ties or wins, the professional conclusion is to use the simple model, not force the network.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 Congratulations! You finished all 4 NeuroTrain Lab notebooks 🎉",
                "From the perceptron to training with overfitting under control: you now know "
                "the whole path. Now open the Streamlit app and put your A and B variants to "
                "the test in Experiment Mode.",
            )
            """
        )
    )

    return cells
