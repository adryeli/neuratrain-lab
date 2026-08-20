"""Notebook 2 — Pérdida y Backpropagation / Loss and Backpropagation.

Covers: what a loss function is (and how it differs from accuracy), MSE for
regression, Binary Cross-Entropy for classification, the chain rule as a
small computational graph worked by hand, and a tiny PyTorch autograd demo
that reproduces the hand-worked gradient exactly. Closes by computing one
real Binary Cross-Entropy value on a row of the breast-cancer dataset with
a fixed (untrained) weight vector.
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
            # NeuroTrain Lab — Notebook 2: Pérdida y Backpropagation

            **Tema:** qué es una función de pérdida, cómo se mide el error con MSE
            (regresión) y Cross-Entropy (clasificación), y cómo la regla de la cadena
            permite calcular exactamente cuánto debe cambiar cada weight para reducir
            ese error — lo que las librerías llaman *backpropagation*.

            > Segundo notebook de 4. En el Notebook 1 aprendiste a **predecir** (forward
            > propagation). Aquí aprendes a **medir el error** y a calcular la dirección en
            > la que hay que mover cada weight para reducirlo. Moverlos de verdad —
            > entrenar— es el Notebook 3.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 Qué aprenderás en este notebook

            Al terminar podrás explicar, sin fórmulas de memoria:

            1. Qué es una función de pérdida y por qué no es lo mismo que la exactitud (accuracy).
            2. Cómo funciona MSE (Mean Squared Error) para problemas de regresión.
            3. Cómo funciona Binary Cross-Entropy para problemas de clasificación, y por qué
               "equivocarse con seguridad" se penaliza mucho más que "no estar seguro".
            4. Qué es la regla de la cadena y cómo se aplica, paso a paso, sobre un grafo
               computacional pequeño para obtener un gradiente.
            5. Que `loss.backward()` en PyTorch no es magia: es exactamente la misma
               aritmética que acabas de hacer a mano.

            **Mapa mental:** `predicción → pérdida → regla de la cadena → gradiente → (Notebook 3: ajustar weights)`
            """
        )
    )
    cells.append(
        code(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            import torch

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("NumPy:", np.__version__, "| PyTorch:", torch.__version__, "| TensorFlow:", tf.__version__)
            print("Raíz del proyecto:", PROJECT_ROOT)
            """
        )
    )

    cells.append(markdown("## 1. Qué es una función de pérdida"))
    cells.append(
        callout(
            "concept",
            "es",
            "El juego de 'frío / caliente'",
            """
            De niños jugábamos a esconder un objeto y guiar a alguien con "frío" (lejos) o
            "caliente" (cerca). Una función de pérdida hace justo eso: convierte "qué tan
            equivocada está la predicción" en **un solo número**. Cuanto más alto, más
            "frío" está el modelo; cuanto más bajo (idealmente 0), más "caliente".

            No nos dice **en qué dirección** moverse — eso lo resuelve la regla de la
            cadena, en la Sección 4 — pero sí nos dice si vamos mejorando.
            """,
        )
    )
    cells.append(
        markdown(
            """
            ### Pérdida vs. exactitud (accuracy)

            No son lo mismo, y confundirlas es un error típico:

            - La **pérdida** (loss) es lo que el optimizador **minimiza directamente**. Es
              continua y sensible: distingue entre "acerté con 0.51 de probabilidad" y
              "acerté con 0.99 de probabilidad", aunque ambas cuenten como acierto.
            - La **exactitud** (accuracy) es lo que a los humanos nos resulta fácil de
              interpretar ("acertó el 90% de las veces"), pero es una cuenta más tosca —
              solo mira si cruzaste el umbral de 0.5, no por cuánto.

            Por eso pueden **divergir a corto plazo**: la pérdida puede bajar (el modelo
            está cada vez más seguro) sin que la exactitud cambie todavía, o viceversa.
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "es",
            "Entonces, ¿por qué no entrenar directamente para maximizar la exactitud?",
            """
            Porque la exactitud no es **derivable**: es un escalón (acierto/fallo), no una
            curva suave. La regla de la cadena (Sección 4) necesita una función suave para
            calcular en qué dirección mover cada weight. Por eso entrenamos minimizando una
            pérdida continua (MSE, Cross-Entropy...) y usamos la exactitud solo para
            **interpretar** el resultado, no para optimizarlo.
            """,
        )
    )

    cells.append(markdown("## 2. MSE (Mean Squared Error): la pérdida para regresión"))
    cells.append(
        markdown(
            r"""
            Para un problema de **regresión** (predecir un número continuo, no una clase),
            la pérdida más común es el **Error Cuadrático Medio**:

            $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

            Elevar al cuadrado hace dos cosas: (1) los errores negativos y positivos no se
            cancelan entre sí, y (2) penaliza mucho más los errores grandes que los pequeños.
            """
        )
    )
    cells.append(
        code(
            """
            # Juguete: predecir la temperatura (°C) real a partir de un termómetro barato
            y_real = np.array([20.0, 25.0, 30.0, 22.0])
            y_pred = np.array([18.0, 27.0, 29.0, 25.0])

            errores_cuadrados = (y_real - y_pred) ** 2
            mse = errores_cuadrados.mean()

            print("Errores al cuadrado:", errores_cuadrados)
            print("MSE:", mse)

            plt.figure(figsize=(5.5, 4))
            x_pos = np.arange(len(y_real))
            plt.scatter(x_pos, y_real, color="#2563EB", label="Real", zorder=3, s=60)
            plt.scatter(x_pos, y_pred, color="#F97316", label="Predicción", zorder=3, s=60)
            for xi, real, pred in zip(x_pos, y_real, y_pred):
                plt.plot([xi, xi], [real, pred], color="#94A3B8", linestyle="--", zorder=1)
            plt.title(f"Brecha al cuadrado entre predicción y realidad (MSE = {mse:.2f})")
            plt.xticks(x_pos, [f"medición {i+1}" for i in x_pos])
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Implementa `mse(y_true, y_pred)` manualmente (sin usar `tf.keras.losses`) y
            comprueba que coincide con `tf.keras.losses.MeanSquaredError()` sobre los mismos
            `y_real`/`y_pred` de arriba (debería dar `4.5`).
            """,
            starter_code="""
            def mse(y_true, y_pred):
                return np.mean((y_true - y_pred) ✏️✏️✏️)


            mi_mse = mse(y_real, y_pred)
            mse_keras = tf.keras.losses.MeanSquaredError()(y_real, y_pred).numpy()

            print("Manual:", mi_mse, "| Keras:", mse_keras)
            assert np.isclose(mi_mse, mse_keras)
            print("¡Coinciden!")
            """,
            solution_code="""
            def mse(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)


            mi_mse = mse(y_real, y_pred)
            mse_keras = tf.keras.losses.MeanSquaredError()(y_real, y_pred).numpy()

            print("Manual:", mi_mse, "| Keras:", mse_keras)
            assert np.isclose(mi_mse, mse_keras)
            print("¡Coinciden!")
            """,
        )
    )

    cells.append(markdown("## 3. Cross-Entropy: la pérdida para clasificación"))
    cells.append(
        callout(
            "concept",
            "es",
            "MSE no es el ajuste natural para nuestro problema",
            """
            Nuestro hilo conductor —¿tumor benigno o maligno?— es una **clasificación
            binaria**: la salida es una probabilidad entre 0 y 1, no un número continuo sin
            límites. MSE trataría por igual un error de "predije 0.5 cuando era 1" que uno
            de "predije 0.99 cuando era 0" en términos relativos — no captura bien que
            **estar seguro y equivocado es mucho peor que estar inseguro**. Para eso existe
            la **Binary Cross-Entropy (BCE)**.
            """,
        )
    )
    cells.append(
        markdown(
            r"""
            Para una etiqueta verdadera $y \in \{0, 1\}$ y una probabilidad predicha $p$:

            $$\text{BCE} = -\big[y \log(p) + (1-y)\log(1-p)\big]$$

            Si la etiqueta real es $y=1$, esto se reduce a $-\log(p)$: cuanto más lejos esté
            $p$ de 1, más grande (y más rápido crece) la pérdida.
            """
        )
    )
    cells.append(
        code(
            """
            # Tabla numérica: y_real = 1 (maligno), tres niveles de confianza del modelo
            bce_keras = tf.keras.losses.BinaryCrossentropy()
            y_real_bce = tf.constant([[1.0]])

            probabilidades = [0.9, 0.5, 0.1]
            perdidas = [bce_keras(y_real_bce, tf.constant([[p]])).numpy() for p in probabilidades]

            for p, l in zip(probabilidades, perdidas):
                print(f"p={p:>3} (confianza {'correcta' if p > 0.5 else 'incorrecta' if p < 0.5 else 'nula'})"
                      f"  ->  BCE = {l:.4f}")

            plt.figure(figsize=(5, 3.8))
            plt.bar([str(p) for p in probabilidades], perdidas, color=["#22C55E", "#F97316", "#F43F5E"])
            plt.title("BCE para y=1 según la probabilidad predicha")
            plt.xlabel("p predicho")
            plt.ylabel("Binary Cross-Entropy")
            plt.grid(alpha=0.2, axis="y")
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            Con `y=1`: `p=0.9` (correcto y seguro) da BCE ≈ **0.105**; `p=0.5` (inseguro) da
            BCE ≈ **0.693**; `p=0.1` (seguro y **equivocado**) da BCE ≈ **2.303** — más de 20
            veces la pérdida de `p=0.9`. Esa asimetría es intencional: el modelo aprende a
            **no estar confiadamente equivocado**.
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Implementa `binary_cross_entropy(y_true, y_pred)` manualmente usando
            `-[y·log(p) + (1-y)·log(1-p)]`, y comprueba que coincide con
            `tf.keras.losses.BinaryCrossentropy()` para `y=1, p=0.9`.
            """,
            starter_code="""
            def binary_cross_entropy(y_true, y_pred):
                return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(✏️✏️✏️))


            mi_bce = binary_cross_entropy(np.array([1.0]), np.array([0.9]))
            bce_keras_val = bce_keras(tf.constant([[1.0]]), tf.constant([[0.9]])).numpy()

            print("Manual:", mi_bce, "| Keras:", bce_keras_val)
            assert np.isclose(mi_bce, bce_keras_val, atol=1e-5)
            print("¡Coinciden!")
            """,
            solution_code="""
            def binary_cross_entropy(y_true, y_pred):
                return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


            mi_bce = binary_cross_entropy(np.array([1.0]), np.array([0.9]))
            bce_keras_val = bce_keras(tf.constant([[1.0]]), tf.constant([[0.9]])).numpy()

            print("Manual:", mi_bce, "| Keras:", bce_keras_val)
            assert np.isclose(mi_bce, bce_keras_val, atol=1e-5)
            print("¡Coinciden!")
            """,
        )
    )
    cells.append(
        callout(
            "mistake",
            "es",
            "np.log(0) rompe todo",
            """
            Si `p` llega a ser exactamente 0 o 1, `np.log(0)` da `-inf` y la pérdida explota.
            Keras evita esto internamente recortando `p` a un rango seguro (ej.
            `[1e-7, 1-1e-7]`). No lo necesitamos en estos ejemplos porque elegimos `p`
            estrictamente entre 0 y 1, pero es la razón por la que nunca deberías usar
            Sigmoid + BCE manual sin ese recorte en código de producción.
            """,
        )
    )

    cells.append(milestone("es", 1))

    cells.append(markdown("## 4. La regla de la cadena: cómo se calcula un gradiente"))
    cells.append(
        callout(
            "concept",
            "es",
            "El corazón de este notebook",
            """
            Ya sabemos **medir** el error (Secciones 2-3). Ahora necesitamos saber **en qué
            dirección y cuánto** mover cada weight para reducirlo. Eso es un **gradiente**:
            la derivada de la pérdida respecto a un weight, `dL/dw`. La regla de la cadena
            nos deja calcularlo paso a paso, multiplicando derivadas locales a lo largo del
            camino desde la pérdida hasta el weight — eso es *backpropagation*.
            """,
        )
    )
    cells.append(
        markdown(
            r"""
            ### Ejemplo mínimo: una sola entrada, un solo weight

            $$x = 2.0 \qquad w = 0.5 \qquad \text{pred} = x \cdot w \qquad \text{target} = 2.0
            \qquad L = (\text{pred} - \text{target})^2$$

            Grafo computacional (hacia adelante):

            ```
            w --(× x)--> pred --(vs target)--> L
            ```

            Y hacia atrás (lo que queremos: `dL/dw`):

            ```
            L --> pred --> w
            ```

            Paso a paso, con regla de la cadena `dL/dw = (dL/dpred) · (dpred/dw)`:

            1. `pred = x · w = 2.0 · 0.5 = 1.0`
            2. `L = (pred - target)² = (1.0 - 2.0)² = 1.0`
            3. `dL/dpred = 2·(pred - target) = 2·(-1.0) = -2.0`
            4. `dpred/dw = x = 2.0`
            5. `dL/dw = dL/dpred · dpred/dw = -2.0 · 2.0 = -4.0`

            `dL/dw = -4.0` significa: si aumentamos `w` un poquito, la pérdida **baja**
            (gradiente negativo) — así que el optimizador (Notebook 3) moverá `w` en
            dirección **contraria** al gradiente para reducir `L`.
            """
        )
    )
    cells.append(
        code(
            """
            # Verificamos el ejemplo anterior con código, sin autograd todavía — puro cálculo manual
            x, w, target = 2.0, 0.5, 2.0

            pred = x * w
            loss = (pred - target) ** 2

            dL_dpred = 2 * (pred - target)
            dpred_dw = x
            dL_dw = dL_dpred * dpred_dw

            print(f"pred={pred}  loss={loss}")
            print(f"dL/dpred={dL_dpred}  dpred/dw={dpred_dw}  dL/dw={dL_dw}")
            """
        )
    )
    cells.append(
        markdown(
            r"""
            ### Un grafo un poco más grande: entrada → lineal → activación → pérdida

            Añadimos un bias y una activación ReLU antes de calcular la pérdida —
            exactamente la estructura de una neurona real seguida de su pérdida:

            $$x=3.0 \quad w=0.4 \quad b=-0.5 \quad z = x\cdot w + b \quad \text{pred} = \text{ReLU}(z)
            \quad \text{target}=1.5 \quad L=(\text{pred}-\text{target})^2$$

            Forward, paso a paso:

            1. `z = x·w + b = 3.0·0.4 + (-0.5) = 0.7`
            2. `pred = ReLU(z) = ReLU(0.7) = 0.7` (z es positivo, ReLU no cambia nada)
            3. `L = (pred - target)² = (0.7 - 1.5)² = 0.64`

            Backward, paso a paso (regla de la cadena en cada nodo, de atrás hacia adelante):

            4. `dL/dpred = 2·(pred - target) = 2·(-0.8) = -1.6`
            5. `dpred/dz = 1` si `z > 0`, `0` si `z < 0` (la derivada de ReLU) → aquí `z=0.7>0`, así que `dpred/dz = 1`
            6. `dL/dz = dL/dpred · dpred/dz = -1.6 · 1 = -1.6`
            7. `dz/dw = x = 3.0` → `dL/dw = dL/dz · dz/dw = -1.6 · 3.0 = -4.8`
            8. `dz/db = 1` → `dL/db = dL/dz · dz/db = -1.6`
            """
        )
    )
    cells.append(
        code(
            """
            # Mismo grafo (lineal + ReLU + pérdida), verificado en código
            def relu(z):
                return max(0.0, z)


            x2, w2, b2, target2 = 3.0, 0.4, -0.5, 1.5

            z2 = x2 * w2 + b2
            pred2 = relu(z2)
            loss2 = (pred2 - target2) ** 2

            dL_dpred2 = 2 * (pred2 - target2)
            dpred_dz2 = 1.0 if z2 > 0 else 0.0
            dL_dz2 = dL_dpred2 * dpred_dz2
            dL_dw2 = dL_dz2 * x2
            dL_db2 = dL_dz2 * 1.0

            print(f"z={z2}  pred={pred2}  loss={loss2}")
            print(f"dL/dpred={dL_dpred2}  dpred/dz={dpred_dz2}  dL/dz={dL_dz2}")
            print(f"dL/dw={dL_dw2}  dL/db={dL_db2}")
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Repite el mismo grafo (`z = x·w + b`, `pred = ReLU(z)`, `L = (pred - target)²`)
            con nuevos números: `x=1.5, w=0.3, b=0.2, target=0.5`. Completa el forward pass
            (`z3` y `pred3`) — el resto (backward) ya está escrito para que verifiques tu
            resultado.
            """,
            starter_code="""
            x3, w3, b3, target3 = 1.5, 0.3, 0.2, 0.5

            z3 = ✏️✏️✏️
            pred3 = relu(z3)
            loss3 = (pred3 - target3) ** 2

            dL_dpred3 = 2 * (pred3 - target3)
            dpred_dz3 = 1.0 if z3 > 0 else 0.0
            dL_dz3 = dL_dpred3 * dpred_dz3
            dL_dw3 = dL_dz3 * x3

            print(f"z={z3}  pred={pred3}  loss={loss3}  dL/dw={dL_dw3}")
            assert np.isclose(z3, 0.65) and np.isclose(loss3, 0.0225) and np.isclose(dL_dw3, 0.45)
            print("¡Correcto!")
            """,
            solution_code="""
            x3, w3, b3, target3 = 1.5, 0.3, 0.2, 0.5

            z3 = x3 * w3 + b3
            pred3 = relu(z3)
            loss3 = (pred3 - target3) ** 2

            dL_dpred3 = 2 * (pred3 - target3)
            dpred_dz3 = 1.0 if z3 > 0 else 0.0
            dL_dz3 = dL_dpred3 * dpred_dz3
            dL_dw3 = dL_dz3 * x3

            print(f"z={z3}  pred={pred3}  loss={loss3}  dL/dw={dL_dw3}")
            assert np.isclose(z3, 0.65) and np.isclose(loss3, 0.0225) and np.isclose(dL_dw3, 0.45)
            print("¡Correcto!")
            """,
        )
    )

    cells.append(markdown("## 5. Autograd: la misma cuenta, calculada por PyTorch"))
    cells.append(
        callout(
            "remember",
            "es",
            "El momento 'ajá'",
            """
            `loss.backward()` **no es magia**: PyTorch recuerda cada operación aplicada a
            un tensor con `requires_grad=True` y aplica exactamente la regla de la cadena
            que hicimos a mano en la Sección 4. Vamos a reconstruir el ejemplo mínimo
            (`x=2.0, w=0.5, target=2.0`) con `torch` y comprobar que `w.grad` da **el mismo
            -4.0** que calculamos a mano.
            """,
        )
    )
    cells.append(
        code(
            """
            xw = torch.tensor(2.0)
            w_t = torch.tensor(0.5, requires_grad=True)  # solo w es "entrenable"
            target_t = torch.tensor(2.0)

            pred_t = xw * w_t
            loss_t = (pred_t - target_t) ** 2

            loss_t.backward()  # aplica la regla de la cadena automáticamente

            print("pred:", pred_t.item(), "| loss:", loss_t.item())
            print("w.grad (calculado por autograd):", w_t.grad.item())
            print("dL/dw calculado a mano en la Sección 4:", -4.0)
            assert np.isclose(w_t.grad.item(), -4.0)
            print("\\n¡Coinciden exactamente! autograd = regla de la cadena aplicada por software.")
            """
        )
    )

    cells.append(milestone("es", 2))

    cells.append(markdown("## 6. Aplicándolo al dataset real: una pérdida, un ejemplo"))
    cells.append(
        markdown(
            """
            Cerramos con el hilo conductor del curso: tomamos **una fila real** del dataset
            *Breast Cancer Wisconsin* (30 variables), un vector de weights **fijo** (no
            entrenado — eso es el Notebook 3) y un bias que elegimos nosotros, calculamos la
            probabilidad predicha con Sigmoid y comparamos la Binary Cross-Entropy de
            nuestra implementación manual contra `tf.keras.losses.BinaryCrossentropy()`.
            """
        )
    )
    cells.append(
        code(
            """
            DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
            df = pd.read_csv(DATA_PATH)

            feature_cols = [c for c in df.columns if c not in ("id", "diagnosis")]
            x_row = df.loc[0, feature_cols].to_numpy(dtype="float64")
            # diagnosis: 'M' (maligno) -> 1, 'B' (benigno) -> 0
            y_row = 1.0 if df.loc[0, "diagnosis"] == "M" else 0.0

            # Weights fijos y arbitrarios (sin entrenar): normalizamos x para que z no explote
            x_row_norm = (x_row - x_row.mean()) / x_row.std()
            rng = np.random.default_rng(RANDOM_STATE)
            w_fijo = rng.normal(0, 0.05, size=x_row_norm.shape)
            b_fijo = 0.0

            def sigmoid(z):
                return 1 / (1 + np.exp(-z))

            z_row = np.dot(x_row_norm, w_fijo) + b_fijo
            p_row = sigmoid(z_row)

            bce_manual = binary_cross_entropy(np.array([y_row]), np.array([p_row]))[0]
            bce_tf = bce_keras(tf.constant([[y_row]]), tf.constant([[p_row]])).numpy()

            print("Etiqueta real (1=maligno, 0=benigno):", y_row)
            print("Probabilidad predicha (weights sin entrenar):", p_row)
            print("BCE manual:", bce_manual, "| BCE Keras:", bce_tf)
            assert np.allclose(bce_manual, bce_tf, atol=1e-5)
            print("\\n¡Coinciden! Con weights sin entrenar, esta pérdida es solo el punto de partida.")
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "es",
            "¿Por qué la pérdida no es baja si el modelo 'nunca vio' este dato?",
            """
            Porque los weights son **aleatorios**, no entrenados — la predicción es
            esencialmente una apuesta a ciegas. El punto de esta sección no es obtener una
            pérdida baja, sino confirmar que sabemos **calcularla correctamente** para un
            caso real. Reducir esa pérdida de verdad, moviendo los weights con el gradiente
            que aprendiste a calcular en la Sección 4, es exactamente lo que hace el
            optimizador del Notebook 3.
            """,
        )
    )

    cells.append(section_header("es"))
    cells.append(
        quiz_question(
            "es", 1,
            "¿Cuál es la pérdida más adecuada para un problema de clasificación binaria (¿maligno o benigno?)?",
            ["MSE, porque siempre funciona igual de bien",
             "Binary Cross-Entropy, porque refleja qué tan correcta es una probabilidad",
             "No hace falta ninguna pérdida si ya usamos Sigmoid",
             "Accuracy, porque es la métrica que de verdad importa"],
            1,
            "MSE trata los errores como distancias numéricas; BCE está diseñada para probabilidades y penaliza con fuerza estar confiadamente equivocado.",
        )
    )
    cells.append(
        quiz_question(
            "es", 2,
            "Con y=1, ¿qué predicción produce mayor Binary Cross-Entropy?",
            ["p = 0.9", "p = 0.5", "p = 0.1", "Todas producen la misma pérdida"],
            2,
            "Cuanto más lejos está p de la etiqueta real (y=1), mayor es -log(p). p=0.1 (confiadamente equivocado) da la pérdida más alta, ≈2.303.",
        )
    )
    cells.append(
        quiz_question(
            "es", 3,
            "¿Qué calcula exactamente `loss.backward()` en PyTorch?",
            ["Entrena el modelo y actualiza los weights directamente",
             "Aplica la regla de la cadena para calcular el gradiente de la pérdida respecto a cada tensor con requires_grad=True",
             "Calcula únicamente la pérdida, sin gradientes",
             "Reinicia los weights a valores aleatorios"],
            1,
            "backward() recorre el grafo computacional hacia atrás aplicando la regla de la cadena en cada nodo, guardando el resultado en .grad — no mueve los weights por sí solo (eso es el optimizador, Notebook 3).",
        )
    )
    cells.append(
        quiz_question(
            "es", 4,
            "Si `dL/dw` da exactamente 0 para un weight, ¿qué implica eso para el entrenamiento en ese punto?",
            ["Que el modelo ya es perfecto siempre",
             "Que, en ese punto exacto, mover ese weight un poquito no cambiaría la pérdida (puede ser un mínimo, un máximo o un punto de silla)",
             "Que hay un error de código y hay que revisar los datos",
             "Que ese weight debe eliminarse de la red"],
            1,
            "Un gradiente cero solo dice que la pendiente local es plana para ese weight en ese punto — no garantiza que sea el mejor punto posible ni que el resto de la red también esté optimizada.",
        )
    )
    cells.append(
        quiz_question(
            "es", 5,
            "En el grafo `w -> (×x) -> pred -> loss`, ¿qué regla te permite obtener `dL/dw` a partir de `dL/dpred` y `dpred/dw`?",
            ["La regla de L'Hôpital", "La regla de la cadena: dL/dw = dL/dpred · dpred/dw",
             "La regla de tres simple", "No se puede calcular sin conocer w"],
            1,
            "La regla de la cadena multiplica las derivadas locales a lo largo del camino desde la pérdida hasta el weight — es la base matemática de backpropagation.",
        )
    )
    cells.append(
        open_question(
            "es", 6,
            "¿El hecho de que la pérdida baje en un ejemplo concreto garantiza que el modelo, en general, sea mejor?",
            [
                "Distingue entre la pérdida de un ejemplo individual y la pérdida promedio (o de validación) sobre muchos ejemplos.",
                "Menciona que mejorar en un ejemplo puede empeorar en otros (sobreajuste a ese punto).",
                "Concluye que solo la tendencia de la pérdida sobre un conjunto amplio, no un único ejemplo, es una señal fiable.",
            ],
        )
    )
    cells.append(
        open_question(
            "es", 7,
            "Explícale a un compañero la regla de la cadena usando el ejemplo x=2, w=0.5, sin usar fórmulas.",
            [
                "Describe el camino: cambiar w cambia pred, y cambiar pred cambia la pérdida — el efecto se 'encadena'.",
                "Explica que el gradiente final es el producto de cuánto cambia cada paso por el cambio del paso anterior.",
                "Usa el ejemplo concreto (dL/dw = -4.0) para mostrar que el resultado es un número calculable, no una intuición vaga.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 ¡Enhorabuena! Completaste el Notebook 2: Pérdida y Backpropagation 🎉",
                "Ya sabes medir el error con MSE y Cross-Entropy, y calcular exactamente cómo "
                "moverías cada weight con la regla de la cadena. En el Notebook 3 usarás ese "
                "gradiente de verdad: el optimizador entrena la red paso a paso.",
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
            # NeuroTrain Lab — Notebook 2: Loss and Backpropagation

            **Topic:** what a loss function is, how MSE (regression) and Cross-Entropy
            (classification) measure error, and how the chain rule lets us calculate
            exactly how much each weight should change to reduce that error — what
            libraries call *backpropagation*.

            > Second of 4 notebooks. In Notebook 1 you learned to **predict** (forward
            > propagation). Here you learn to **measure the error** and compute the
            > direction each weight should move to reduce it. Actually moving them —
            > training — is Notebook 3.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 What you'll learn in this notebook

            By the end you should be able to explain, without memorized formulas:

            1. What a loss function is and why it isn't the same thing as accuracy.
            2. How MSE (Mean Squared Error) works for regression problems.
            3. How Binary Cross-Entropy works for classification problems, and why "being
               confidently wrong" is penalized far more than "being unsure."
            4. What the chain rule is and how it's applied, step by step, over a small
               computational graph to get a gradient.
            5. That `loss.backward()` in PyTorch isn't magic — it's exactly the same
               arithmetic you just did by hand.

            **Mental map:** `prediction → loss → chain rule → gradient → (Notebook 3: update weights)`
            """
        )
    )
    cells.append(
        code(
            """
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import tensorflow as tf
            import torch

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("NumPy:", np.__version__, "| PyTorch:", torch.__version__, "| TensorFlow:", tf.__version__)
            print("Project root:", PROJECT_ROOT)
            """
        )
    )

    cells.append(markdown("## 1. What a loss function is"))
    cells.append(
        callout(
            "concept",
            "en",
            "The 'hot / cold' game",
            """
            As kids we'd hide an object and guide someone with "cold" (far) or "hot"
            (close). A loss function does exactly that: it turns "how wrong is this
            prediction" into **a single number**. The higher it is, the "colder" the model
            is; the lower (ideally 0), the "hotter."

            It doesn't tell us **which direction** to move — that's the chain rule, in
            Section 4 — but it does tell us whether we're improving.
            """,
        )
    )
    cells.append(
        markdown(
            """
            ### Loss vs. accuracy

            These are not the same thing, and confusing them is a typical mistake:

            - The **loss** is what the optimizer **directly minimizes**. It's continuous
              and sensitive: it distinguishes "I got it right with 0.51 probability" from
              "I got it right with 0.99 probability," even though both count as correct.
            - **Accuracy** is what's easy for humans to interpret ("it got 90% right"), but
              it's a cruder count — it only looks at whether you crossed the 0.5 threshold,
              not by how much.

            That's why they can **diverge short-term**: loss can go down (the model is
            getting more confident) without accuracy changing yet, or vice versa.
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "en",
            "So why not just train directly to maximize accuracy?",
            """
            Because accuracy isn't **differentiable**: it's a step function (right/wrong),
            not a smooth curve. The chain rule (Section 4) needs a smooth function to
            calculate which direction to move each weight. That's why we train by
            minimizing a continuous loss (MSE, Cross-Entropy...) and use accuracy only to
            **interpret** the result, not to optimize it.
            """,
        )
    )

    cells.append(markdown("## 2. MSE (Mean Squared Error): the loss for regression"))
    cells.append(
        markdown(
            r"""
            For a **regression** problem (predicting a continuous number, not a class), the
            most common loss is the **Mean Squared Error**:

            $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

            Squaring does two things: (1) negative and positive errors don't cancel each
            other out, and (2) it penalizes large errors far more than small ones.
            """
        )
    )
    cells.append(
        code(
            """
            # Toy example: predicting the real temperature (°C) from a cheap thermometer
            y_true = np.array([20.0, 25.0, 30.0, 22.0])
            y_pred = np.array([18.0, 27.0, 29.0, 25.0])

            squared_errors = (y_true - y_pred) ** 2
            mse = squared_errors.mean()

            print("Squared errors:", squared_errors)
            print("MSE:", mse)

            plt.figure(figsize=(5.5, 4))
            x_pos = np.arange(len(y_true))
            plt.scatter(x_pos, y_true, color="#2563EB", label="Real", zorder=3, s=60)
            plt.scatter(x_pos, y_pred, color="#F97316", label="Prediction", zorder=3, s=60)
            for xi, real, pred in zip(x_pos, y_true, y_pred):
                plt.plot([xi, xi], [real, pred], color="#94A3B8", linestyle="--", zorder=1)
            plt.title(f"Squared gap between prediction and reality (MSE = {mse:.2f})")
            plt.xticks(x_pos, [f"reading {i+1}" for i in x_pos])
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Implement `mse(y_true, y_pred)` manually (without using `tf.keras.losses`) and
            check it matches `tf.keras.losses.MeanSquaredError()` on the same
            `y_true`/`y_pred` above (should give `4.5`).
            """,
            starter_code="""
            def mse(y_true, y_pred):
                return np.mean((y_true - y_pred) ✏️✏️✏️)


            my_mse = mse(y_true, y_pred)
            mse_keras = tf.keras.losses.MeanSquaredError()(y_true, y_pred).numpy()

            print("Manual:", my_mse, "| Keras:", mse_keras)
            assert np.isclose(my_mse, mse_keras)
            print("They match!")
            """,
            solution_code="""
            def mse(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)


            my_mse = mse(y_true, y_pred)
            mse_keras = tf.keras.losses.MeanSquaredError()(y_true, y_pred).numpy()

            print("Manual:", my_mse, "| Keras:", mse_keras)
            assert np.isclose(my_mse, mse_keras)
            print("They match!")
            """,
        )
    )

    cells.append(markdown("## 3. Cross-Entropy: the loss for classification"))
    cells.append(
        callout(
            "concept",
            "en",
            "MSE isn't the natural fit for our problem",
            """
            Our running problem — benign or malignant tumor? — is **binary
            classification**: the output is a probability between 0 and 1, not an
            unbounded continuous number. MSE would treat "I predicted 0.5 when it was 1"
            and "I predicted 0.99 when it was 0" too similarly in relative terms — it
            doesn't capture that **being confident and wrong is far worse than being
            unsure**. That's what **Binary Cross-Entropy (BCE)** is for.
            """,
        )
    )
    cells.append(
        markdown(
            r"""
            For a true label $y \in \{0, 1\}$ and a predicted probability $p$:

            $$\text{BCE} = -\big[y \log(p) + (1-y)\log(1-p)\big]$$

            If the true label is $y=1$, this reduces to $-\log(p)$: the further $p$ is from
            1, the larger (and faster-growing) the loss.
            """
        )
    )
    cells.append(
        code(
            """
            # Numeric table: y_true = 1 (malignant), three confidence levels
            bce_keras = tf.keras.losses.BinaryCrossentropy()
            y_true_bce = tf.constant([[1.0]])

            probabilities = [0.9, 0.5, 0.1]
            losses = [bce_keras(y_true_bce, tf.constant([[p]])).numpy() for p in probabilities]

            for p, l in zip(probabilities, losses):
                print(f"p={p:>3} ({'correct' if p > 0.5 else 'incorrect' if p < 0.5 else 'no'} confidence)"
                      f"  ->  BCE = {l:.4f}")

            plt.figure(figsize=(5, 3.8))
            plt.bar([str(p) for p in probabilities], losses, color=["#22C55E", "#F97316", "#F43F5E"])
            plt.title("BCE for y=1 as a function of the predicted probability")
            plt.xlabel("predicted p")
            plt.ylabel("Binary Cross-Entropy")
            plt.grid(alpha=0.2, axis="y")
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            With `y=1`: `p=0.9` (correct and confident) gives BCE ≈ **0.105**; `p=0.5`
            (unsure) gives BCE ≈ **0.693**; `p=0.1` (confident and **wrong**) gives BCE ≈
            **2.303** — more than 20 times the loss at `p=0.9`. That asymmetry is
            intentional: the model learns **not to be confidently wrong**.
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Implement `binary_cross_entropy(y_true, y_pred)` manually using
            `-[y·log(p) + (1-y)·log(1-p)]`, and check it matches
            `tf.keras.losses.BinaryCrossentropy()` for `y=1, p=0.9`.
            """,
            starter_code="""
            def binary_cross_entropy(y_true, y_pred):
                return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(✏️✏️✏️))


            my_bce = binary_cross_entropy(np.array([1.0]), np.array([0.9]))
            bce_keras_val = bce_keras(tf.constant([[1.0]]), tf.constant([[0.9]])).numpy()

            print("Manual:", my_bce, "| Keras:", bce_keras_val)
            assert np.isclose(my_bce, bce_keras_val, atol=1e-5)
            print("They match!")
            """,
            solution_code="""
            def binary_cross_entropy(y_true, y_pred):
                return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


            my_bce = binary_cross_entropy(np.array([1.0]), np.array([0.9]))
            bce_keras_val = bce_keras(tf.constant([[1.0]]), tf.constant([[0.9]])).numpy()

            print("Manual:", my_bce, "| Keras:", bce_keras_val)
            assert np.isclose(my_bce, bce_keras_val, atol=1e-5)
            print("They match!")
            """,
        )
    )
    cells.append(
        callout(
            "mistake",
            "en",
            "np.log(0) breaks everything",
            """
            If `p` ever hits exactly 0 or 1, `np.log(0)` gives `-inf` and the loss explodes.
            Keras avoids this internally by clipping `p` to a safe range (e.g.
            `[1e-7, 1-1e-7]`). We don't need it in these examples because we chose `p`
            strictly between 0 and 1, but it's why you should never use Sigmoid + manual
            BCE without that clipping in production code.
            """,
        )
    )

    cells.append(milestone("en", 1))

    cells.append(markdown("## 4. The chain rule: how a gradient is computed"))
    cells.append(
        callout(
            "concept",
            "en",
            "The heart of this notebook",
            """
            We already know how to **measure** error (Sections 2-3). Now we need to know
            **which direction and by how much** to move each weight to reduce it. That's a
            **gradient**: the derivative of the loss with respect to a weight, `dL/dw`. The
            chain rule lets us compute it step by step, multiplying local derivatives along
            the path from the loss back to the weight — that's *backpropagation*.
            """,
        )
    )
    cells.append(
        markdown(
            r"""
            ### Minimal example: one input, one weight

            $$x = 2.0 \qquad w = 0.5 \qquad \text{pred} = x \cdot w \qquad \text{target} = 2.0
            \qquad L = (\text{pred} - \text{target})^2$$

            Computational graph (forward):

            ```
            w --(× x)--> pred --(vs target)--> L
            ```

            And backward (what we want: `dL/dw`):

            ```
            L --> pred --> w
            ```

            Step by step, with the chain rule `dL/dw = (dL/dpred) · (dpred/dw)`:

            1. `pred = x · w = 2.0 · 0.5 = 1.0`
            2. `L = (pred - target)² = (1.0 - 2.0)² = 1.0`
            3. `dL/dpred = 2·(pred - target) = 2·(-1.0) = -2.0`
            4. `dpred/dw = x = 2.0`
            5. `dL/dw = dL/dpred · dpred/dw = -2.0 · 2.0 = -4.0`

            `dL/dw = -4.0` means: if we increase `w` a little, the loss **decreases**
            (negative gradient) — so the optimizer (Notebook 3) will move `w` in the
            **opposite** direction of the gradient to reduce `L`.
            """
        )
    )
    cells.append(
        code(
            """
            # Verify the example above in code, no autograd yet — pure manual computation
            x, w, target = 2.0, 0.5, 2.0

            pred = x * w
            loss = (pred - target) ** 2

            dL_dpred = 2 * (pred - target)
            dpred_dw = x
            dL_dw = dL_dpred * dpred_dw

            print(f"pred={pred}  loss={loss}")
            print(f"dL/dpred={dL_dpred}  dpred/dw={dpred_dw}  dL/dw={dL_dw}")
            """
        )
    )
    cells.append(
        markdown(
            r"""
            ### A slightly bigger graph: input → linear → activation → loss

            We add a bias and a ReLU activation before computing the loss — exactly the
            structure of a real neuron followed by its loss:

            $$x=3.0 \quad w=0.4 \quad b=-0.5 \quad z = x\cdot w + b \quad \text{pred} = \text{ReLU}(z)
            \quad \text{target}=1.5 \quad L=(\text{pred}-\text{target})^2$$

            Forward, step by step:

            1. `z = x·w + b = 3.0·0.4 + (-0.5) = 0.7`
            2. `pred = ReLU(z) = ReLU(0.7) = 0.7` (z is positive, ReLU leaves it unchanged)
            3. `L = (pred - target)² = (0.7 - 1.5)² = 0.64`

            Backward, step by step (chain rule at each node, back to front):

            4. `dL/dpred = 2·(pred - target) = 2·(-0.8) = -1.6`
            5. `dpred/dz = 1` if `z > 0`, `0` if `z < 0` (ReLU's derivative) → here `z=0.7>0`, so `dpred/dz = 1`
            6. `dL/dz = dL/dpred · dpred/dz = -1.6 · 1 = -1.6`
            7. `dz/dw = x = 3.0` → `dL/dw = dL/dz · dz/dw = -1.6 · 3.0 = -4.8`
            8. `dz/db = 1` → `dL/db = dL/dz · dz/db = -1.6`
            """
        )
    )
    cells.append(
        code(
            """
            # Same graph (linear + ReLU + loss), verified in code
            def relu(z):
                return max(0.0, z)


            x2, w2, b2, target2 = 3.0, 0.4, -0.5, 1.5

            z2 = x2 * w2 + b2
            pred2 = relu(z2)
            loss2 = (pred2 - target2) ** 2

            dL_dpred2 = 2 * (pred2 - target2)
            dpred_dz2 = 1.0 if z2 > 0 else 0.0
            dL_dz2 = dL_dpred2 * dpred_dz2
            dL_dw2 = dL_dz2 * x2
            dL_db2 = dL_dz2 * 1.0

            print(f"z={z2}  pred={pred2}  loss={loss2}")
            print(f"dL/dpred={dL_dpred2}  dpred/dz={dpred_dz2}  dL/dz={dL_dz2}")
            print(f"dL/dw={dL_dw2}  dL/db={dL_db2}")
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Repeat the same graph (`z = x·w + b`, `pred = ReLU(z)`, `L = (pred - target)²`)
            with new numbers: `x=1.5, w=0.3, b=0.2, target=0.5`. Fill in the forward pass
            (`z3` and `pred3`) — the rest (backward) is already written so you can check
            your result.
            """,
            starter_code="""
            x3, w3, b3, target3 = 1.5, 0.3, 0.2, 0.5

            z3 = ✏️✏️✏️
            pred3 = relu(z3)
            loss3 = (pred3 - target3) ** 2

            dL_dpred3 = 2 * (pred3 - target3)
            dpred_dz3 = 1.0 if z3 > 0 else 0.0
            dL_dz3 = dL_dpred3 * dpred_dz3
            dL_dw3 = dL_dz3 * x3

            print(f"z={z3}  pred={pred3}  loss={loss3}  dL/dw={dL_dw3}")
            assert np.isclose(z3, 0.65) and np.isclose(loss3, 0.0225) and np.isclose(dL_dw3, 0.45)
            print("Correct!")
            """,
            solution_code="""
            x3, w3, b3, target3 = 1.5, 0.3, 0.2, 0.5

            z3 = x3 * w3 + b3
            pred3 = relu(z3)
            loss3 = (pred3 - target3) ** 2

            dL_dpred3 = 2 * (pred3 - target3)
            dpred_dz3 = 1.0 if z3 > 0 else 0.0
            dL_dz3 = dL_dpred3 * dpred_dz3
            dL_dw3 = dL_dz3 * x3

            print(f"z={z3}  pred={pred3}  loss={loss3}  dL/dw={dL_dw3}")
            assert np.isclose(z3, 0.65) and np.isclose(loss3, 0.0225) and np.isclose(dL_dw3, 0.45)
            print("Correct!")
            """,
        )
    )

    cells.append(markdown("## 5. Autograd: the same computation, done by PyTorch"))
    cells.append(
        callout(
            "remember",
            "en",
            "The 'aha' moment",
            """
            `loss.backward()` is **not magic**: PyTorch remembers every operation applied
            to a tensor with `requires_grad=True` and applies exactly the chain rule we did
            by hand in Section 4. Let's rebuild the minimal example (`x=2.0, w=0.5,
            target=2.0`) with `torch` and confirm `w.grad` gives the **same -4.0** we
            computed by hand.
            """,
        )
    )
    cells.append(
        code(
            """
            xw = torch.tensor(2.0)
            w_t = torch.tensor(0.5, requires_grad=True)  # only w is "trainable"
            target_t = torch.tensor(2.0)

            pred_t = xw * w_t
            loss_t = (pred_t - target_t) ** 2

            loss_t.backward()  # applies the chain rule automatically

            print("pred:", pred_t.item(), "| loss:", loss_t.item())
            print("w.grad (computed by autograd):", w_t.grad.item())
            print("dL/dw computed by hand in Section 4:", -4.0)
            assert np.isclose(w_t.grad.item(), -4.0)
            print("\\nThey match exactly! autograd = the chain rule applied by software.")
            """
        )
    )

    cells.append(milestone("en", 2))

    cells.append(markdown("## 6. Applying it to the real dataset: one loss, one example"))
    cells.append(
        markdown(
            """
            We close with the course's running thread: we take **one real row** of the
            *Breast Cancer Wisconsin* dataset (30 features), a **fixed** (not trained —
            that's Notebook 3) weight vector and a bias we choose, compute the predicted
            probability with Sigmoid, and compare our manual Binary Cross-Entropy against
            `tf.keras.losses.BinaryCrossentropy()`.
            """
        )
    )
    cells.append(
        code(
            """
            DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
            df = pd.read_csv(DATA_PATH)

            feature_cols = [c for c in df.columns if c not in ("id", "diagnosis")]
            x_row = df.loc[0, feature_cols].to_numpy(dtype="float64")
            # diagnosis: 'M' (malignant) -> 1, 'B' (benign) -> 0
            y_row = 1.0 if df.loc[0, "diagnosis"] == "M" else 0.0

            # Fixed, arbitrary weights (untrained): normalize x so z doesn't explode
            x_row_norm = (x_row - x_row.mean()) / x_row.std()
            rng = np.random.default_rng(RANDOM_STATE)
            w_fixed = rng.normal(0, 0.05, size=x_row_norm.shape)
            b_fixed = 0.0

            def sigmoid(z):
                return 1 / (1 + np.exp(-z))

            z_row = np.dot(x_row_norm, w_fixed) + b_fixed
            p_row = sigmoid(z_row)

            bce_manual = binary_cross_entropy(np.array([y_row]), np.array([p_row]))[0]
            bce_tf = bce_keras(tf.constant([[y_row]]), tf.constant([[p_row]])).numpy()

            print("True label (1=malignant, 0=benign):", y_row)
            print("Predicted probability (untrained weights):", p_row)
            print("Manual BCE:", bce_manual, "| Keras BCE:", bce_tf)
            assert np.allclose(bce_manual, bce_tf, atol=1e-5)
            print("\\nThey match! With untrained weights, this loss is just the starting point.")
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "en",
            "Why isn't the loss low if the model 'never saw' this data before?",
            """
            Because the weights are **random**, not trained — the prediction is essentially
            a blind guess. The point of this section isn't to get a low loss, but to
            confirm we can **compute it correctly** on a real case. Actually reducing that
            loss, by moving the weights along the gradient you learned to compute in
            Section 4, is exactly what Notebook 3's optimizer does.
            """,
        )
    )

    cells.append(section_header("en"))
    cells.append(
        quiz_question(
            "en", 1,
            "Which loss best fits a binary classification problem (malignant or benign?)?",
            ["MSE, because it always works equally well",
             "Binary Cross-Entropy, because it reflects how correct a probability is",
             "No loss is needed once we use Sigmoid",
             "Accuracy, because it's the metric that actually matters"],
            1,
            "MSE treats errors as numeric distances; BCE is designed for probabilities and strongly penalizes being confidently wrong.",
        )
    )
    cells.append(
        quiz_question(
            "en", 2,
            "With y=1, which prediction produces the highest Binary Cross-Entropy?",
            ["p = 0.9", "p = 0.5", "p = 0.1", "All produce the same loss"],
            2,
            "The further p is from the true label (y=1), the larger -log(p) is. p=0.1 (confidently wrong) gives the highest loss, ≈2.303.",
        )
    )
    cells.append(
        quiz_question(
            "en", 3,
            "What exactly does `loss.backward()` compute in PyTorch?",
            ["It trains the model and updates the weights directly",
             "It applies the chain rule to compute the loss's gradient with respect to every tensor with requires_grad=True",
             "It only computes the loss, without gradients",
             "It resets the weights to random values"],
            1,
            "backward() walks the computational graph backward applying the chain rule at each node, storing the result in .grad — it doesn't move the weights itself (that's the optimizer, Notebook 3).",
        )
    )
    cells.append(
        quiz_question(
            "en", 4,
            "If `dL/dw` is exactly 0 for a weight, what does that imply for training at that point?",
            ["That the model is now perfect, always",
             "That, at that exact point, nudging that weight wouldn't change the loss (it could be a minimum, a maximum, or a saddle point)",
             "That there's a code bug and the data must be checked",
             "That this weight must be removed from the network"],
            1,
            "A zero gradient only says the local slope is flat for that weight at that point — it doesn't guarantee it's the best possible point, or that the rest of the network is optimized too.",
        )
    )
    cells.append(
        quiz_question(
            "en", 5,
            "In the graph `w -> (×x) -> pred -> loss`, which rule lets you get `dL/dw` from `dL/dpred` and `dpred/dw`?",
            ["L'Hôpital's rule", "The chain rule: dL/dw = dL/dpred · dpred/dw",
             "The rule of three", "It can't be computed without knowing w"],
            1,
            "The chain rule multiplies local derivatives along the path from the loss back to the weight — it's the mathematical basis of backpropagation.",
        )
    )
    cells.append(
        open_question(
            "en", 6,
            "Does the loss going down on one specific example guarantee the model is better overall?",
            [
                "Distinguishes between the loss on a single example and the average (or validation) loss over many examples.",
                "Mentions that improving on one example can make others worse (overfitting to that point).",
                "Concludes that only the loss trend over a broad set, not a single example, is a reliable signal.",
            ],
        )
    )
    cells.append(
        open_question(
            "en", 7,
            "Explain the chain rule to a classmate using the x=2, w=0.5 example, without formulas.",
            [
                "Describes the path: changing w changes pred, and changing pred changes the loss — the effect 'chains' through.",
                "Explains the final gradient is the product of how much each step changes times the previous step's change.",
                "Uses the concrete example (dL/dw = -4.0) to show the result is a computable number, not a vague intuition.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 Congratulations! You finished Notebook 2: Loss and Backpropagation 🎉",
                "You now know how to measure error with MSE and Cross-Entropy, and how to "
                "compute exactly how you'd move each weight using the chain rule. In Notebook 3 "
                "you'll put that gradient to real use: the optimizer trains the network step by step.",
            )
            """
        )
    )

    return cells
