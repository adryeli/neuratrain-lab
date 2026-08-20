"""Notebook 1 — El Perceptrón / The Perceptron.

Covers: the perceptron, activation functions (ReLU/Sigmoid/Softmax), the
MLP as stacked layers, forward propagation as matrix transformations, a
decision-boundary demo on a synthetic 2D dataset, and tensors in NumPy /
PyTorch / TensorFlow. No PyTorch training happens here or anywhere else in
the project — only tensor creation, ops, and one forward pass.
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
            # NeuroTrain Lab — Notebook 1: El Perceptrón

            **Tema:** qué es una neurona artificial, para qué sirven las funciones de
            activación (ReLU, Sigmoid, Softmax), cómo se apilan en una red multicapa (MLP)
            y cómo se ve todo esto como tensores en PyTorch y TensorFlow.

            > Primer notebook de 4. Aquí no entrenamos nada todavía: solo entendemos cómo
            > una red **predice**. Aprender a que se equivoque menos es el Notebook 2.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 Qué aprenderás en este notebook

            Al terminar podrás explicar, sin fórmulas de memoria:

            1. Qué calcula matemáticamente una neurona (perceptrón).
            2. Qué hacen ReLU, Sigmoid y Softmax, y cuándo se usa cada una.
            3. Por qué una sola neurona no basta y para qué apilamos capas (MLP).
            4. Por qué "forward propagation" es, en el fondo, multiplicar matrices.
            5. Qué es un tensor y cómo se ve el mismo cálculo en NumPy, PyTorch y TensorFlow.

            **Mapa mental:** `neurona → activación → capa → MLP → forward propagation → tensores`
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
            from sklearn.datasets import make_moons

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate
            from neurotrain.visualization import plot_decision_boundary

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("NumPy:", np.__version__, "| PyTorch:", torch.__version__, "| TensorFlow:", tf.__version__)
            print("Raíz del proyecto:", PROJECT_ROOT)
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 0. El hilo conductor de los 4 notebooks

            En los 4 notebooks vamos a usar el mismo problema real de fondo: **predecir si un
            tumor es benigno o maligno** a partir de 30 variables numéricas (dataset *Breast
            Cancer Wisconsin*). Aquí solo lo miramos por encima — la auditoría completa y el
            entrenamiento real llegan en el Notebook 4.
            """
        )
    )
    cells.append(
        code(
            """
            DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
            df = pd.read_csv(DATA_PATH)
            print(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
            df.head(3)
            """
        )
    )

    cells.append(markdown("## 1. Qué es una neurona artificial"))
    cells.append(
        callout(
            "concept",
            "es",
            "La neurona como un portero de discoteca",
            """
            Imagina un portero decidiendo si dejar entrar a alguien. No mira un solo dato:
            pondera varios (¿va bien vestido? ¿tiene reserva? ¿es tarde?) y cada criterio
            "pesa" distinto para él. Si la suma ponderada supera su umbral de exigencia
            (su "actitud" ese día), te deja pasar.

            Una neurona artificial hace exactamente eso con números: multiplica cada entrada
            `x` por un **weight** que dice cuánto importa, suma un **bias** (lo exigente que
            es por defecto) y aplica una función de activación para decidir la salida.
            """,
        )
    )
    cells.append(
        markdown(
            """
            Matemáticamente, para una neurona con entradas $x_1, x_2, \\dots, x_n$:

            $$z = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b$$

            $z$ es solo un número. Para convertirlo en una decisión aplicamos una **función de
            activación** — eso es la Sección 2. Antes, construyamos $z$ a mano.
            """
        )
    )
    cells.append(
        code(
            """
            def weighted_sum(x, w, b):
                \"\"\"z = x·w + b — el cálculo que hace una neurona antes de activar.\"\"\"
                return np.dot(x, w) + b


            # Ejemplo: "¿llevo paraguas?" con 2 entradas: prob. de lluvia, viento (0-1)
            x_ejemplo = np.array([0.8, 0.3])
            w_ejemplo = np.array([0.9, 0.2])
            b_ejemplo = -0.4

            z = weighted_sum(x_ejemplo, w_ejemplo, b_ejemplo)
            print("z (suma ponderada):", z)
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Completa `weighted_sum_manual` calculando `z` **sin usar `np.dot`**, con un bucle
            `for` que recorra `x` y `w` a la vez. Debe dar el mismo resultado que la celda
            anterior (`z ≈ 0.98`).
            """,
            starter_code="""
            def weighted_sum_manual(x, w, b):
                total = 0.0
                for xi, wi in zip(x, w):
                    total += ✏️✏️✏️
                return total + b

            print(weighted_sum_manual(x_ejemplo, w_ejemplo, b_ejemplo))
            """,
            solution_code="""
            def weighted_sum_manual(x, w, b):
                total = 0.0
                for xi, wi in zip(x, w):
                    total += xi * wi
                return total + b

            print(weighted_sum_manual(x_ejemplo, w_ejemplo, b_ejemplo))
            """,
        )
    )
    cells.append(
        callout(
            "doubt",
            "es",
            "¿Con qué weight se queda la neurona?",
            """
            Con todos. Si una neurona recibe 30 valores (como en nuestro dataset real),
            tiene **30 weights** — uno por cada entrada — más un bias. No existe "el weight
            de la neurona" en singular; cada conexión de entrada tiene el suyo propio.
            """,
        )
    )

    cells.append(markdown("## 2. Funciones de activación: ReLU, Sigmoid y Softmax"))
    cells.append(
        callout(
            "concept",
            "es",
            "¿Por qué no dejar z tal cual?",
            """
            Si apilamos neuronas sin ninguna activación no lineal entre medias, toda la red
            —por muchas capas que tenga— sigue siendo matemáticamente equivalente a **una
            sola** transformación lineal. La no linealidad es lo que permite aprender formas
            curvas, no solo líneas rectas. Lo comprobamos en la Sección 2.3.
            """,
        )
    )
    cells.append(
        code(
            """
            def relu(z):
                return np.maximum(0, z)


            def sigmoid(z):
                return 1 / (1 + np.exp(-z))


            z_valores = np.linspace(-6, 6, 200)

            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].plot(z_valores, relu(z_valores), color="#7C3AED")
            axes[0].set_title("ReLU(z) = max(0, z)")
            axes[0].axhline(0, color="#94A3B8", linewidth=0.8)
            axes[1].plot(z_valores, sigmoid(z_valores), color="#2563EB")
            axes[1].set_title("Sigmoid(z) = 1 / (1 + e⁻ᶻ)")
            axes[1].axhline(0.5, color="#94A3B8", linewidth=0.8, linestyle="--")
            for axis in axes:
                axis.grid(alpha=0.2)
            fig.tight_layout()
            plt.show()

            print("ReLU(-3) =", relu(-3), "| ReLU(2) =", relu(2))
            print("Sigmoid(0) =", sigmoid(0), "| Sigmoid(6) ≈", round(sigmoid(6), 3))
            """
        )
    )
    cells.append(
        markdown(
            """
            - **ReLU** dice "pasa tal cual si eres positivo, si eres negativo eres cero".
              Se usa casi siempre en las **capas ocultas**: es barata de calcular y ayuda a
              que el gradiente fluya bien (más sobre esto en el Notebook 3).
            - **Sigmoid** aplasta cualquier número a un rango (0, 1) — se lee como una
              **probabilidad**. Se usa en la **capa de salida** de clasificación binaria
              (¿maligno o benigno? un único número entre 0 y 1).
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="Implementa `sigmoid` y `relu` tú mismo y comprueba que coinciden con Keras.",
            starter_code="""
            def mi_relu(z):
                return np.maximum(✏️✏️✏️, z)


            def mi_sigmoid(z):
                return 1 / (1 + np.exp(✏️✏️✏️))


            keras_relu = tf.keras.activations.relu(tf.constant([-2.0, 0.0, 3.0])).numpy()
            keras_sigmoid = tf.keras.activations.sigmoid(tf.constant([-2.0, 0.0, 3.0])).numpy()

            assert np.allclose(mi_relu(np.array([-2.0, 0.0, 3.0])), keras_relu)
            assert np.allclose(mi_sigmoid(np.array([-2.0, 0.0, 3.0])), keras_sigmoid)
            print("¡Coinciden con Keras!")
            """,
            solution_code="""
            def mi_relu(z):
                return np.maximum(0, z)


            def mi_sigmoid(z):
                return 1 / (1 + np.exp(-z))
            """,
        )
    )
    cells.append(
        callout(
            "mistake",
            "es",
            "La no linealidad importa de verdad",
            """
            Comprueba qué pasa si encadenas dos transformaciones **lineales** sin activación
            entre medias: sigue siendo una única transformación lineal, sin curvas nuevas.
            """,
        )
    )
    cells.append(
        code(
            """
            # Dos capas "lineales" (sin activación) compuestas siguen siendo una sola línea
            A = np.array([[2.0, 0.0], [0.0, 2.0]])
            B = np.array([[1.0, 1.0], [0.0, 1.0]])
            compuesta = A @ B
            print("Aplicar A y luego B equivale a UNA sola matriz:")
            print(compuesta)
            print("Por eso metemos ReLU/Sigmoid entre capas: rompen esa equivalencia.")
            """
        )
    )
    cells.append(
        markdown(
            """
            ### Softmax: cuando hay más de 2 clases

            Sigmoid da una probabilidad para una sola clase. Cuando hay **varias clases que
            se excluyen entre sí** (ej. "manzana / plátano / naranja"), usamos **Softmax**:
            convierte una lista de números (*logits*) en probabilidades que **siempre suman 1**.
            """
        )
    )
    cells.append(
        code(
            """
            def softmax(logits):
                exponentes = np.exp(logits - np.max(logits))  # -max: estabilidad numérica
                return exponentes / exponentes.sum()


            logits_fruta = np.array([2.0, 1.0, 0.1])  # puntuación cruda para manzana/plátano/naranja
            probabilidades = softmax(logits_fruta)
            print("Logits:        ", logits_fruta)
            print("Probabilidades:", probabilidades.round(3))
            print("Suma:          ", probabilidades.sum())
            """
        )
    )

    cells.append(milestone("es", 0))

    cells.append(markdown("## 3. De una neurona a un MLP"))
    cells.append(
        markdown(
            """
            Una neurona sola solo puede trazar una **línea recta** de separación. Para
            aprender formas más complejas, apilamos neuronas en **capas**, y capas en una
            **red multicapa (MLP, Multi-Layer Perceptron)**:

            `entradas → capa oculta 1 → capa oculta 2 → ... → capa de salida`

            El número de neuronas por capa (32, 16, lo que sea) y el número de capas son
            **hiperparámetros**: decisiones que tomamos y validamos, no fórmulas que se
            deriven del número de entradas.
            """
        )
    )

    cells.append(markdown("## 4. Forward propagation como multiplicación de matrices"))
    cells.append(
        callout(
            "concept",
            "es",
            "Una capa Dense es solo esto",
            r"""
            `Dense(n, activation)` calcula, para **todos los ejemplos del batch a la vez**:

            $$H = \text{activación}(X \cdot W + b)$$

            $X$ es la matriz de entradas, $W$ la matriz de weights de la capa (una columna
            por neurona) y $b$ el vector de biases. Es exactamente la misma cuenta que
            hicimos a mano en la Sección 1, aplicada a todas las neuronas de la capa de golpe.
            """,
        )
    )
    cells.append(
        code(
            """
            # Un MLP de juguete: 2 entradas -> capa oculta de 3 -> salida de 1
            x = np.array([[1.0, 2.0]])  # 1 ejemplo, 2 variables

            W1 = np.array([[0.5, -0.9, 0.3],
                           [0.2,  0.1, -0.1]])   # (2 entradas, 3 neuronas ocultas)
            b1 = np.array([0.1, -0.2, 0.05])

            z1 = x @ W1 + b1
            h1 = relu(z1)
            print("z1 (antes de activar):", z1)
            print("h1 (tras ReLU):       ", h1, "  <- el -0.9 se convirtió en 0")

            W2 = np.array([[0.7], [-0.5], [0.9]])  # (3 entradas, 1 neurona de salida)
            b2 = np.array([-0.1])

            z2 = h1 @ W2 + b2
            y_hat = sigmoid(z2)
            print("z2 (antes de activar):", z2)
            print("y_hat (tras Sigmoid): ", y_hat, " <- probabilidad final")
            """
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Repite el forward pass anterior para un segundo ejemplo `x2 = [[-1.0, 0.5]]`,
            reutilizando los mismos `W1`, `b1`, `W2`, `b2`. Completa la multiplicación de
            matrices de la primera capa.
            """,
            starter_code="""
            x2 = np.array([[-1.0, 0.5]])

            z1_b = ✏️✏️✏️ + b1
            h1_b = relu(z1_b)
            z2_b = h1_b @ W2 + b2
            y_hat_b = sigmoid(z2_b)
            print("y_hat para x2:", y_hat_b)
            """,
            solution_code="""
            x2 = np.array([[-1.0, 0.5]])

            z1_b = x2 @ W1 + b1
            h1_b = relu(z1_b)
            z2_b = h1_b @ W2 + b2
            y_hat_b = sigmoid(z2_b)
            print("y_hat para x2:", y_hat_b)
            """,
        )
    )

    cells.append(markdown("## 5. Visualizando por qué hacen falta capas: fronteras de decisión"))
    cells.append(
        markdown(
            """
            Con 30 variables reales no podemos "dibujar" la frontera que separa maligno de
            benigno. Así que usamos un dataset sintético de 2 variables — `make_moons` —
            donde sí podemos verla.
            """
        )
    )
    cells.append(
        code(
            """
            X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=RANDOM_STATE)

            plt.figure(figsize=(4.5, 4))
            plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="RdBu_r", edgecolor="white")
            plt.title("make_moons: 2 clases, frontera curva")
            plt.show()
            """
        )
    )
    cells.append(
        code(
            """
            # Un único perceptrón (equivalente a Dense(1, sigmoid) sin capas ocultas)
            perceptron = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            perceptron.compile(optimizer="adam", loss="binary_crossentropy")
            perceptron.fit(X_moons, y_moons, epochs=80, verbose=0)

            # Un MLP pequeño con una capa oculta no lineal
            mlp = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            mlp.compile(optimizer="adam", loss="binary_crossentropy")
            mlp.fit(X_moons, y_moons, epochs=80, verbose=0)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            plot_decision_boundary(
                X_moons, y_moons,
                lambda grid: perceptron.predict(grid, verbose=0).ravel(),
                title="Un perceptrón: solo puede trazar una recta", lang="es", ax=axes[0],
            )
            plot_decision_boundary(
                X_moons, y_moons,
                lambda grid: mlp.predict(grid, verbose=0).ravel(),
                title="MLP (Dense 8, ReLU): curva la frontera", lang="es", ax=axes[1],
            )
            fig.tight_layout()
            plt.show()
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "es",
            "¿Y con el dataset real de 30 variables?",
            """
            El mismo principio aplica, solo que la frontera vive en un espacio de 30
            dimensiones que no podemos dibujar en papel. `make_moons` existe únicamente para
            que **veas** con tus ojos por qué un MLP con activación no lineal separa formas
            que un perceptrón solo no puede.

            **Reto:** repite esta celda cambiando `make_moons` por
            `sklearn.datasets.make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=RANDOM_STATE)`.
            """,
        )
    )

    cells.append(milestone("es", 2))

    cells.append(markdown("## 6. Tensores: NumPy, PyTorch y TensorFlow"))
    cells.append(
        callout(
            "concept",
            "es",
            "Un tensor es solo un array con superpoderes",
            """
            Un **tensor** es la misma idea que un array de NumPy (números organizados en
            filas/columnas/dimensiones), pero con dos añadidos que nos importarán más
            adelante: puede vivir en GPU, y puede **recordar las operaciones que se le
            aplicaron** para calcular gradientes automáticamente (lo veremos en el Notebook 2).
            Para operaciones básicas de hoy, se comportan igual que un array.
            """,
        )
    )
    cells.append(
        code(
            """
            # Crear el mismo tensor 2x2 en los tres "idiomas"
            datos = [[1.0, 2.0], [3.0, 4.0]]

            array_np = np.array(datos)
            tensor_torch = torch.tensor(datos)
            tensor_tf = tf.constant(datos)

            print("NumPy     ->", array_np.shape, array_np.dtype)
            print("PyTorch   ->", tensor_torch.shape, tensor_torch.dtype)
            print("TensorFlow->", tensor_tf.shape, tensor_tf.dtype)
            """
        )
    )
    cells.append(
        code(
            """
            # Mismas operaciones básicas en los tres frameworks
            print("Suma +10:")
            print(" numpy :", array_np + 10)
            print(" torch :", (tensor_torch + 10).numpy())
            print(" tf    :", (tensor_tf + 10).numpy())

            print("\\nReshape a (4,):")
            print(" numpy :", array_np.reshape(4))
            print(" torch :", tensor_torch.reshape(4).numpy())
            print(" tf    :", tf.reshape(tensor_tf, (4,)).numpy())
            """
        )
    )
    cells.append(
        markdown(
            "Ahora repetimos **exactamente** el forward pass de la Sección 4, pero calculado "
            "en los tres frameworks a la vez — deben dar el mismo número."
        )
    )
    cells.append(
        code(
            """
            x_np = np.array([[1.0, 2.0]], dtype="float32")

            # --- NumPy (lo que ya hicimos) ---
            salida_numpy = sigmoid(relu(x_np @ W1 + b1) @ W2 + b2)

            # --- PyTorch ---
            x_t = torch.tensor(x_np)
            W1_t, b1_t = torch.tensor(W1, dtype=torch.float32), torch.tensor(b1, dtype=torch.float32)
            W2_t, b2_t = torch.tensor(W2, dtype=torch.float32), torch.tensor(b2, dtype=torch.float32)
            salida_torch = torch.sigmoid(torch.relu(x_t @ W1_t + b1_t) @ W2_t + b2_t)

            # --- TensorFlow ---
            x_tf = tf.constant(x_np)
            salida_tf = tf.sigmoid(tf.nn.relu(x_tf @ W1 + b1) @ W2 + b2)

            print("NumPy      ->", salida_numpy)
            print("PyTorch    ->", salida_torch.numpy())
            print("TensorFlow ->", salida_tf.numpy())
            print("\\n¿Coinciden los tres?", np.allclose(salida_numpy, salida_torch.numpy())
                  and np.allclose(salida_numpy, salida_tf.numpy()))
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "es",
            "No memorices sintaxis de PyTorch",
            """
            El resto del proyecto (Notebooks 2-4, la app) usa **TensorFlow/Keras** para
            entrenar de verdad. PyTorch aparece aquí — y una vez más en el Notebook 2 — solo
            para que reconozcas los mismos conceptos detrás de una sintaxis distinta. No
            necesitas dominar PyTorch para seguir el resto del curso.
            """,
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="Crea un tensor de PyTorch con los valores `[10, 20, 30, 40, 50, 60]` y dale forma `(2, 3)`.",
            starter_code="""
            valores = torch.tensor([10, 20, 30, 40, 50, 60])
            matriz = valores.reshape(✏️✏️✏️)
            print(matriz)
            print(matriz.shape)
            """,
            solution_code="""
            valores = torch.tensor([10, 20, 30, 40, 50, 60])
            matriz = valores.reshape(2, 3)
            print(matriz)
            print(matriz.shape)
            """,
        )
    )

    cells.append(section_header("es"))
    cells.append(
        quiz_question(
            "es", 1,
            "Una neurona recibe 64 valores de entrada. ¿Cuántos weights tiene?",
            ["1, compartido para todas las entradas", "64, uno por entrada, más el bias",
             "64, y además otro por cada neurona de la red", "0, los weights los tiene la capa, no la neurona"],
            1,
            "Cada conexión de entrada tiene su propio weight. 64 entradas → 64 weights + 1 bias.",
        )
    )
    cells.append(
        quiz_question(
            "es", 2,
            "¿Qué produce ReLU cuando la entrada es negativa, por ejemplo -3?",
            ["-3 sin cambios", "0", "3 (el valor absoluto)", "Un error, ReLU no admite negativos"],
            1,
            "ReLU(z) = max(0, z). Cualquier valor negativo se convierte en 0; los positivos pasan igual.",
        )
    )
    cells.append(
        quiz_question(
            "es", 3,
            "¿Por qué el MLP separa `make_moons` y un único perceptrón no puede?",
            ["Porque el MLP tiene más datos de entrenamiento",
             "Porque el MLP combina varias neuronas con una activación no lineal, permitiendo fronteras curvas",
             "Porque el perceptrón usa Sigmoid y el MLP no",
             "No hay diferencia real, es cuestión de suerte con la semilla aleatoria"],
            1,
            "Un perceptrón solo traza una recta. Apilar neuronas con una no linealidad entre medias permite curvar esa frontera.",
        )
    )
    cells.append(
        quiz_question(
            "es", 4,
            "Softmax convierte 3 logits en 3 probabilidades. ¿Qué es siempre cierto del resultado?",
            ["Todas valen exactamente 0.33", "Suman exactamente 1", "La mayor es siempre mayor que 0.9",
             "Pueden ser negativas si el logit lo es"],
            1,
            "Softmax normaliza para que las probabilidades de todas las clases sumen 1, sin importar los valores de entrada.",
        )
    )
    cells.append(
        quiz_question(
            "es", 5,
            "¿Qué diferencia hay, para una operación básica como sumar 10, entre un tensor de PyTorch y un array de NumPy?",
            ["El resultado numérico es distinto", "Ninguna en el resultado; el tensor además puede ir a GPU y calcular gradientes",
             "Los tensores no admiten reshape", "Los tensores solo aceptan números enteros"],
            1,
            "Para operaciones básicas, el resultado numérico es idéntico. Los tensores añaden capacidades (GPU, autograd) que aprovecharemos en el Notebook 2.",
        )
    )
    cells.append(
        open_question(
            "es", 6,
            "Explica con tus propias palabras qué hace un forward pass, sin usar la palabra 'magia'.",
            [
                "Menciona que cada capa hace una multiplicación de matrices más un vector de biases.",
                "Explica que después de cada capa (menos casi siempre la última) hay una activación no lineal.",
                "Deja claro que el resultado final es una predicción, no todavía un aprendizaje.",
            ],
        )
    )
    cells.append(
        open_question(
            "es", 7,
            "Un compañero dice: 'con más neuronas en la capa oculta, la red siempre predice mejor'. ¿Estás de acuerdo?",
            [
                "Distingue entre capacidad (más parámetros) y generalización (predecir bien en datos nuevos).",
                "Menciona que más neuronas también significa más riesgo de sobreajuste (idea que se retoma en el Notebook 4).",
                "Concluye que el número de neuronas es un hiperparámetro que se valida, no una ley fija.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 ¡Enhorabuena! Completaste el Notebook 1: El Perceptrón 🎉",
                "Ya sabes cómo una neurona transforma números en decisiones y cómo se ve eso "
                "como tensores. En el Notebook 2 descubrirás cómo la red aprende de sus errores.",
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
            # NeuroTrain Lab — Notebook 1: The Perceptron

            **Topic:** what an artificial neuron is, what activation functions (ReLU,
            Sigmoid, Softmax) are for, how they stack into a multi-layer network (MLP), and
            how all of this looks as tensors in PyTorch and TensorFlow.

            > First of 4 notebooks. We don't train anything yet — we're only understanding
            > how a network **predicts**. Teaching it to make fewer mistakes is Notebook 2.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 What you'll learn in this notebook

            By the end you should be able to explain, without memorized formulas:

            1. What a neuron (perceptron) computes mathematically.
            2. What ReLU, Sigmoid, and Softmax do, and when to use each.
            3. Why a single neuron isn't enough, and why we stack layers (MLP).
            4. Why "forward propagation" is, underneath, just matrix multiplication.
            5. What a tensor is, and how the same computation looks in NumPy, PyTorch, and TensorFlow.

            **Mental map:** `neuron → activation → layer → MLP → forward propagation → tensors`
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
            from sklearn.datasets import make_moons

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate
            from neurotrain.visualization import plot_decision_boundary

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            torch.manual_seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("NumPy:", np.__version__, "| PyTorch:", torch.__version__, "| TensorFlow:", tf.__version__)
            print("Project root:", PROJECT_ROOT)
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 0. The thread running through all 4 notebooks

            All 4 notebooks share the same real underlying problem: **predicting whether a
            tumor is benign or malignant** from 30 numeric features (the *Breast Cancer
            Wisconsin* dataset). Here we just glance at it — the full audit and real training
            happen in Notebook 4.
            """
        )
    )
    cells.append(
        code(
            """
            DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv"
            df = pd.read_csv(DATA_PATH)
            print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            df.head(3)
            """
        )
    )

    cells.append(markdown("## 1. What an artificial neuron is"))
    cells.append(
        callout(
            "concept",
            "en",
            "A neuron as a nightclub bouncer",
            """
            Picture a bouncer deciding whether to let someone in. They don't look at just
            one thing: they weigh several signals (well dressed? on the guest list? how late
            is it?), and each signal matters a different amount to them. If the weighted sum
            clears their threshold for the night (how strict they're feeling), you're in.

            An artificial neuron does exactly that with numbers: it multiplies each input
            `x` by a **weight** saying how much it matters, adds a **bias** (how strict the
            neuron is by default), and applies an activation function to decide the output.
            """,
        )
    )
    cells.append(
        markdown(
            """
            Mathematically, for a neuron with inputs $x_1, x_2, \\dots, x_n$:

            $$z = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b$$

            $z$ is just a number. Turning it into a decision needs an **activation
            function** — that's Section 2. First, let's compute $z$ by hand.
            """
        )
    )
    cells.append(
        code(
            """
            def weighted_sum(x, w, b):
                \"\"\"z = x·w + b — what a neuron computes before activating.\"\"\"
                return np.dot(x, w) + b


            # Example: "should I bring an umbrella?" — 2 inputs: rain probability, wind (0-1)
            x_example = np.array([0.8, 0.3])
            w_example = np.array([0.9, 0.2])
            b_example = -0.4

            z = weighted_sum(x_example, w_example, b_example)
            print("z (weighted sum):", z)
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Complete `weighted_sum_manual`, computing `z` **without using `np.dot`**, using a
            `for` loop over `x` and `w` together. It should match the previous cell
            (`z ≈ 0.98`).
            """,
            starter_code="""
            def weighted_sum_manual(x, w, b):
                total = 0.0
                for xi, wi in zip(x, w):
                    total += ✏️✏️✏️
                return total + b

            print(weighted_sum_manual(x_example, w_example, b_example))
            """,
            solution_code="""
            def weighted_sum_manual(x, w, b):
                total = 0.0
                for xi, wi in zip(x, w):
                    total += xi * wi
                return total + b

            print(weighted_sum_manual(x_example, w_example, b_example))
            """,
        )
    )
    cells.append(
        callout(
            "doubt",
            "en",
            "Which weight does the neuron keep?",
            """
            All of them. If a neuron receives 30 values (like in our real dataset), it has
            **30 weights** — one per input — plus a bias. There's no such thing as "the
            neuron's weight" in the singular; every incoming connection has its own.
            """,
        )
    )

    cells.append(markdown("## 2. Activation functions: ReLU, Sigmoid, and Softmax"))
    cells.append(
        callout(
            "concept",
            "en",
            "Why not just leave z as it is?",
            """
            If we stack neurons with no non-linear activation in between, the whole
            network — no matter how many layers — stays mathematically equivalent to a
            **single** linear transformation. Non-linearity is what lets a network learn
            curved shapes, not just straight lines. We'll prove it in Section 2.3.
            """,
        )
    )
    cells.append(
        code(
            """
            def relu(z):
                return np.maximum(0, z)


            def sigmoid(z):
                return 1 / (1 + np.exp(-z))


            z_values = np.linspace(-6, 6, 200)

            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            axes[0].plot(z_values, relu(z_values), color="#7C3AED")
            axes[0].set_title("ReLU(z) = max(0, z)")
            axes[0].axhline(0, color="#94A3B8", linewidth=0.8)
            axes[1].plot(z_values, sigmoid(z_values), color="#2563EB")
            axes[1].set_title("Sigmoid(z) = 1 / (1 + e⁻ᶻ)")
            axes[1].axhline(0.5, color="#94A3B8", linewidth=0.8, linestyle="--")
            for axis in axes:
                axis.grid(alpha=0.2)
            fig.tight_layout()
            plt.show()

            print("ReLU(-3) =", relu(-3), "| ReLU(2) =", relu(2))
            print("Sigmoid(0) =", sigmoid(0), "| Sigmoid(6) ≈", round(sigmoid(6), 3))
            """
        )
    )
    cells.append(
        markdown(
            """
            - **ReLU** says "pass through unchanged if you're positive, zero out if you're
              negative." Used almost everywhere in **hidden layers**: cheap to compute and
              helps gradients flow well (more on that in Notebook 3).
            - **Sigmoid** squashes any number into the range (0, 1) — read as a
              **probability**. Used in the **output layer** for binary classification
              (malignant or benign? a single number between 0 and 1).
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="Implement `sigmoid` and `relu` yourself and check they match Keras.",
            starter_code="""
            def my_relu(z):
                return np.maximum(✏️✏️✏️, z)


            def my_sigmoid(z):
                return 1 / (1 + np.exp(✏️✏️✏️))


            keras_relu = tf.keras.activations.relu(tf.constant([-2.0, 0.0, 3.0])).numpy()
            keras_sigmoid = tf.keras.activations.sigmoid(tf.constant([-2.0, 0.0, 3.0])).numpy()

            assert np.allclose(my_relu(np.array([-2.0, 0.0, 3.0])), keras_relu)
            assert np.allclose(my_sigmoid(np.array([-2.0, 0.0, 3.0])), keras_sigmoid)
            print("Matches Keras!")
            """,
            solution_code="""
            def my_relu(z):
                return np.maximum(0, z)


            def my_sigmoid(z):
                return 1 / (1 + np.exp(-z))
            """,
        )
    )
    cells.append(
        callout(
            "mistake",
            "en",
            "Non-linearity actually matters",
            """
            Let's check what happens if you chain two **linear** transformations with no
            activation in between: it's still a single linear transformation, no new curves.
            """,
        )
    )
    cells.append(
        code(
            """
            # Two "linear" layers (no activation) composed are still just one line
            A = np.array([[2.0, 0.0], [0.0, 2.0]])
            B = np.array([[1.0, 1.0], [0.0, 1.0]])
            composed = A @ B
            print("Applying A then B is equivalent to ONE single matrix:")
            print(composed)
            print("That's why we insert ReLU/Sigmoid between layers: they break that equivalence.")
            """
        )
    )
    cells.append(
        markdown(
            """
            ### Softmax: when there are more than 2 classes

            Sigmoid gives one probability for one class. When there are **several mutually
            exclusive classes** (e.g. "apple / banana / orange"), we use **Softmax**: it
            turns a list of numbers (*logits*) into probabilities that **always sum to 1**.
            """
        )
    )
    cells.append(
        code(
            """
            def softmax(logits):
                exponents = np.exp(logits - np.max(logits))  # -max: numerical stability
                return exponents / exponents.sum()


            fruit_logits = np.array([2.0, 1.0, 0.1])  # raw scores for apple/banana/orange
            probabilities = softmax(fruit_logits)
            print("Logits:       ", fruit_logits)
            print("Probabilities:", probabilities.round(3))
            print("Sum:          ", probabilities.sum())
            """
        )
    )

    cells.append(milestone("en", 0))

    cells.append(markdown("## 3. From one neuron to an MLP"))
    cells.append(
        markdown(
            """
            A single neuron can only draw a **straight line** boundary. To learn more
            complex shapes, we stack neurons into **layers**, and layers into a
            **Multi-Layer Perceptron (MLP)**:

            `inputs → hidden layer 1 → hidden layer 2 → ... → output layer`

            The number of neurons per layer (32, 16, whatever) and the number of layers are
            **hyperparameters**: decisions we make and validate, not formulas derived from
            the number of inputs.
            """
        )
    )

    cells.append(markdown("## 4. Forward propagation as matrix multiplication"))
    cells.append(
        callout(
            "concept",
            "en",
            "A Dense layer is just this",
            r"""
            `Dense(n, activation)` computes, for **every example in the batch at once**:

            $$H = \text{activation}(X \cdot W + b)$$

            $X$ is the input matrix, $W$ is the layer's weight matrix (one column per
            neuron), and $b$ is the bias vector. It's the exact same computation we did by
            hand in Section 1, applied to every neuron in the layer at once.
            """,
        )
    )
    cells.append(
        code(
            """
            # A toy MLP: 2 inputs -> hidden layer of 3 -> output of 1
            x = np.array([[1.0, 2.0]])  # 1 example, 2 features

            W1 = np.array([[0.5, -0.9, 0.3],
                           [0.2,  0.1, -0.1]])   # (2 inputs, 3 hidden neurons)
            b1 = np.array([0.1, -0.2, 0.05])

            z1 = x @ W1 + b1
            h1 = relu(z1)
            print("z1 (before activation):", z1)
            print("h1 (after ReLU):       ", h1, "  <- the -0.9 became 0")

            W2 = np.array([[0.7], [-0.5], [0.9]])  # (3 inputs, 1 output neuron)
            b2 = np.array([-0.1])

            z2 = h1 @ W2 + b2
            y_hat = sigmoid(z2)
            print("z2 (before activation):", z2)
            print("y_hat (after Sigmoid): ", y_hat, " <- final probability")
            """
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Repeat the forward pass above for a second example `x2 = [[-1.0, 0.5]]`, reusing
            the same `W1`, `b1`, `W2`, `b2`. Fill in the first layer's matrix multiplication.
            """,
            starter_code="""
            x2 = np.array([[-1.0, 0.5]])

            z1_b = ✏️✏️✏️ + b1
            h1_b = relu(z1_b)
            z2_b = h1_b @ W2 + b2
            y_hat_b = sigmoid(z2_b)
            print("y_hat for x2:", y_hat_b)
            """,
            solution_code="""
            x2 = np.array([[-1.0, 0.5]])

            z1_b = x2 @ W1 + b1
            h1_b = relu(z1_b)
            z2_b = h1_b @ W2 + b2
            y_hat_b = sigmoid(z2_b)
            print("y_hat for x2:", y_hat_b)
            """,
        )
    )

    cells.append(markdown("## 5. Seeing why layers matter: decision boundaries"))
    cells.append(
        markdown(
            """
            With 30 real features we can't "draw" the boundary that separates malignant from
            benign. So we use a synthetic 2-feature dataset — `make_moons` — where we can.
            """
        )
    )
    cells.append(
        code(
            """
            X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=RANDOM_STATE)

            plt.figure(figsize=(4.5, 4))
            plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="RdBu_r", edgecolor="white")
            plt.title("make_moons: 2 classes, curved boundary")
            plt.show()
            """
        )
    )
    cells.append(
        code(
            """
            # A single perceptron (equivalent to Dense(1, sigmoid) with no hidden layers)
            perceptron = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            perceptron.compile(optimizer="adam", loss="binary_crossentropy")
            perceptron.fit(X_moons, y_moons, epochs=80, verbose=0)

            # A small MLP with one non-linear hidden layer
            mlp = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(2,)),
                tf.keras.layers.Dense(8, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ])
            mlp.compile(optimizer="adam", loss="binary_crossentropy")
            mlp.fit(X_moons, y_moons, epochs=80, verbose=0)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            plot_decision_boundary(
                X_moons, y_moons,
                lambda grid: perceptron.predict(grid, verbose=0).ravel(),
                title="A perceptron: can only draw a line", lang="en", ax=axes[0],
            )
            plot_decision_boundary(
                X_moons, y_moons,
                lambda grid: mlp.predict(grid, verbose=0).ravel(),
                title="MLP (Dense 8, ReLU): curves the boundary", lang="en", ax=axes[1],
            )
            fig.tight_layout()
            plt.show()
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "en",
            "What about the real 30-feature dataset?",
            """
            Same principle, just living in a 30-dimensional space we can't draw on paper.
            `make_moons` exists purely so you can **see with your own eyes** why an MLP with
            a non-linear activation separates shapes a lone perceptron can't.

            **Challenge:** rerun this cell swapping `make_moons` for
            `sklearn.datasets.make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=RANDOM_STATE)`.
            """,
        )
    )

    cells.append(milestone("en", 2))

    cells.append(markdown("## 6. Tensors: NumPy, PyTorch, and TensorFlow"))
    cells.append(
        callout(
            "concept",
            "en",
            "A tensor is just an array with superpowers",
            """
            A **tensor** is the same idea as a NumPy array (numbers organized in
            rows/columns/dimensions), plus two extras that will matter later: it can live on
            a GPU, and it can **remember the operations applied to it** to compute gradients
            automatically (Notebook 2). For today's basic operations, they behave exactly
            like an array.
            """,
        )
    )
    cells.append(
        code(
            """
            # Create the same 2x2 tensor in all three "languages"
            data = [[1.0, 2.0], [3.0, 4.0]]

            array_np = np.array(data)
            tensor_torch = torch.tensor(data)
            tensor_tf = tf.constant(data)

            print("NumPy     ->", array_np.shape, array_np.dtype)
            print("PyTorch   ->", tensor_torch.shape, tensor_torch.dtype)
            print("TensorFlow->", tensor_tf.shape, tensor_tf.dtype)
            """
        )
    )
    cells.append(
        code(
            """
            # Same basic operations across all three frameworks
            print("Add +10:")
            print(" numpy :", array_np + 10)
            print(" torch :", (tensor_torch + 10).numpy())
            print(" tf    :", (tensor_tf + 10).numpy())

            print("\\nReshape to (4,):")
            print(" numpy :", array_np.reshape(4))
            print(" torch :", tensor_torch.reshape(4).numpy())
            print(" tf    :", tf.reshape(tensor_tf, (4,)).numpy())
            """
        )
    )
    cells.append(
        markdown(
            "Now let's repeat **exactly** the forward pass from Section 4, computed in all "
            "three frameworks at once — they should give the same number."
        )
    )
    cells.append(
        code(
            """
            x_np = np.array([[1.0, 2.0]], dtype="float32")

            # --- NumPy (what we already did) ---
            numpy_output = sigmoid(relu(x_np @ W1 + b1) @ W2 + b2)

            # --- PyTorch ---
            x_t = torch.tensor(x_np)
            W1_t, b1_t = torch.tensor(W1, dtype=torch.float32), torch.tensor(b1, dtype=torch.float32)
            W2_t, b2_t = torch.tensor(W2, dtype=torch.float32), torch.tensor(b2, dtype=torch.float32)
            torch_output = torch.sigmoid(torch.relu(x_t @ W1_t + b1_t) @ W2_t + b2_t)

            # --- TensorFlow ---
            x_tf = tf.constant(x_np)
            tf_output = tf.sigmoid(tf.nn.relu(x_tf @ W1 + b1) @ W2 + b2)

            print("NumPy      ->", numpy_output)
            print("PyTorch    ->", torch_output.numpy())
            print("TensorFlow ->", tf_output.numpy())
            print("\\nDo all three match?", np.allclose(numpy_output, torch_output.numpy())
                  and np.allclose(numpy_output, tf_output.numpy()))
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "en",
            "Don't memorize PyTorch syntax",
            """
            The rest of the project (Notebooks 2-4, the app) uses **TensorFlow/Keras** for
            actual training. PyTorch shows up here — and once more in Notebook 2 — only so
            you recognize the same concepts behind different syntax. You don't need to
            master PyTorch to follow the rest of the course.
            """,
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="Create a PyTorch tensor with values `[10, 20, 30, 40, 50, 60]` and reshape it to `(2, 3)`.",
            starter_code="""
            values = torch.tensor([10, 20, 30, 40, 50, 60])
            matrix = values.reshape(✏️✏️✏️)
            print(matrix)
            print(matrix.shape)
            """,
            solution_code="""
            values = torch.tensor([10, 20, 30, 40, 50, 60])
            matrix = values.reshape(2, 3)
            print(matrix)
            print(matrix.shape)
            """,
        )
    )

    cells.append(section_header("en"))
    cells.append(
        quiz_question(
            "en", 1,
            "A neuron receives 64 input values. How many weights does it have?",
            ["1, shared across all inputs", "64, one per input, plus the bias",
             "64, plus one more per neuron in the network", "0, weights belong to the layer, not the neuron"],
            1,
            "Every incoming connection has its own weight. 64 inputs → 64 weights + 1 bias.",
        )
    )
    cells.append(
        quiz_question(
            "en", 2,
            "What does ReLU output for a negative input, e.g. -3?",
            ["-3 unchanged", "0", "3 (the absolute value)", "An error, ReLU doesn't accept negatives"],
            1,
            "ReLU(z) = max(0, z). Any negative value becomes 0; positive values pass through unchanged.",
        )
    )
    cells.append(
        quiz_question(
            "en", 3,
            "Why does an MLP separate `make_moons` when a single perceptron can't?",
            ["Because the MLP sees more training data",
             "Because the MLP combines several neurons with a non-linear activation, enabling curved boundaries",
             "Because the perceptron uses Sigmoid and the MLP doesn't",
             "There's no real difference, it's down to the random seed"],
            1,
            "A perceptron can only draw a straight line. Stacking neurons with a non-linearity in between lets that boundary curve.",
        )
    )
    cells.append(
        quiz_question(
            "en", 4,
            "Softmax turns 3 logits into 3 probabilities. What's always true about the result?",
            ["They're all exactly 0.33", "They sum to exactly 1", "The largest is always above 0.9",
             "They can be negative if the logit is negative"],
            1,
            "Softmax normalizes so every class's probability sums to 1, regardless of the input values.",
        )
    )
    cells.append(
        quiz_question(
            "en", 5,
            "For a basic operation like adding 10, what's different between a PyTorch tensor and a NumPy array?",
            ["The numeric result is different", "Nothing in the result; the tensor can additionally run on GPU and track gradients",
             "Tensors don't support reshape", "Tensors only accept integers"],
            1,
            "For basic operations the numeric result is identical. Tensors add capabilities (GPU, autograd) we'll use starting in Notebook 2.",
        )
    )
    cells.append(
        open_question(
            "en", 6,
            "Explain in your own words what a forward pass is doing, without using the word 'magic'.",
            [
                "Mentions that each layer does a matrix multiplication plus a bias vector.",
                "Explains that after each layer (almost always except the last) there's a non-linear activation.",
                "Makes clear the final result is a prediction, not yet any learning.",
            ],
        )
    )
    cells.append(
        open_question(
            "en", 7,
            "A classmate says: 'with more neurons in the hidden layer, the network always predicts better.' Do you agree?",
            [
                "Distinguishes between capacity (more parameters) and generalization (predicting well on new data).",
                "Mentions that more neurons also means more risk of overfitting (picked back up in Notebook 4).",
                "Concludes that the number of neurons is a hyperparameter to validate, not a fixed rule.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 Congratulations! You finished Notebook 1: The Perceptron 🎉",
                "You now know how a neuron turns numbers into decisions, and how that looks "
                "as tensors. In Notebook 2 you'll discover how the network learns from its mistakes.",
            )
            """
        )
    )

    return cells
