"""Notebook 3 — Optimizadores / Optimizers.

Covers: gradient descent as a step-by-step hiker-in-the-fog process, the
effect of the learning rate, SGD vs Momentum vs Adam trained on the same
small MLP, an empirical vanishing-gradient demonstration (sigmoid vs ReLU
across a deep stack of Dense layers), and a "why isn't my network learning?"
diagnostic table. Builds directly on Notebook 2 (loss functions, gradients,
chain rule, backprop) and bridges to Notebook 4 (training loops, epochs,
overfitting).
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
            # NeuroTrain Lab — Notebook 3: Optimizadores

            **Tema:** cómo un optimizador usa el gradiente (Notebook 2) para de verdad mover
            los weights, qué papel juega la tasa de aprendizaje (*learning rate*), en qué se
            diferencian SGD, Momentum y Adam, y por qué a veces los gradientes "desaparecen"
            en redes profundas.

            > Notebook 3 de 4. Ya sabes calcular el gradiente de la pérdida. Ahora usamos ese
            > gradiente para **dar pasos** que reduzcan el error — y veremos que no todos los
            > pasos son iguales.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 Qué aprenderás en este notebook

            1. Qué hace el descenso de gradiente, paso a paso, sobre una función de pérdida.
            2. Por qué la tasa de aprendizaje es la decisión más delicada de todo el entrenamiento.
            3. Qué añade Momentum sobre SGD puro, y qué hace Adam distinto (a nivel conceptual).
            4. Cómo reconocer un gradiente que se desvanece en una red profunda, y por qué pasa.
            5. Un diagnóstico rápido para "mi red no aprende" antes de tocar el código a ciegas.

            **Mapa mental:** `gradiente → paso de descenso → learning rate → SGD → Momentum → Adam → gradientes que desaparecen`
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
            from sklearn.datasets import make_moons

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate
            from neurotrain.visualization import plot_gradient_magnitude_by_layer

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("NumPy:", np.__version__, "| TensorFlow:", tf.__version__)
            print("Raíz del proyecto:", PROJECT_ROOT)
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 0. Dónde estamos

            En el Notebook 2 aprendiste a calcular $\\partial L / \\partial w$ para cada weight
            de la red — el gradiente te dice **en qué dirección** crece la pérdida. Lo único
            que falta es la regla que decide, con ese gradiente en la mano, **cuánto y cómo**
            mover cada weight. Esa regla es el **optimizador**.
            """
        )
    )

    cells.append(markdown("## 1. Descenso de gradiente: el excursionista en la niebla"))
    cells.append(
        callout(
            "concept",
            "es",
            "Un excursionista que no ve el valle",
            """
            Imagina a alguien bajando una montaña envuelta en niebla espesa. No puede ver el
            valle ni el mapa completo — solo siente, bajo sus pies, hacia dónde baja el
            terreno **en ese punto exacto**. Da un paso en esa dirección, vuelve a sentir la
            pendiente, da otro paso, y así sucesivamente.

            Eso es exactamente el descenso de gradiente: en cada punto solo conocemos el
            gradiente **local** (la pendiente bajo nuestros pies), no la forma completa de la
            función de pérdida. Actualizamos así:

            $$w \\leftarrow w - \\eta \\cdot \\frac{\\partial L}{\\partial w}$$

            donde $\\eta$ (eta) es la **tasa de aprendizaje** — el tamaño del paso del excursionista.
            """,
        )
    )
    cells.append(
        code(
            """
            def f(w):
                \"\"\"Una 'pérdida' de juguete en 1D: un cuenco con mínimo en w=3.\"\"\"
                return (w - 3) ** 2 + 1


            def gradiente(w):
                return 2 * (w - 3)


            def descenso_gradiente(w_inicial, lr, pasos):
                trayectoria = [w_inicial]
                w = w_inicial
                for _ in range(pasos):
                    w = w - lr * gradiente(w)
                    trayectoria.append(w)
                return np.array(trayectoria)


            trayectoria = descenso_gradiente(w_inicial=-2.0, lr=0.2, pasos=15)
            print("Posiciones de w:", trayectoria.round(3))
            print("Pérdida final:", round(f(trayectoria[-1]), 4), "(el mínimo real vale 1.0 en w=3)")
            """
        )
    )
    cells.append(
        code(
            """
            w_curva = np.linspace(-3, 8, 200)
            plt.figure(figsize=(7, 4.5))
            plt.plot(w_curva, f(w_curva), color="#94A3B8", label="f(w) = (w-3)² + 1")
            plt.plot(trayectoria, f(trayectoria), "o-", color="#7C3AED", label="pasos del descenso")
            for i in range(len(trayectoria) - 1):
                plt.annotate(
                    "", xy=(trayectoria[i + 1], f(trayectoria[i + 1])),
                    xytext=(trayectoria[i], f(trayectoria[i])),
                    arrowprops=dict(arrowstyle="->", color="#F97316", alpha=0.6),
                )
            plt.scatter([3], [1], color="#22C55E", zorder=5, label="mínimo real (w=3)")
            plt.title("El excursionista bajando el cuenco de pérdida, paso a paso")
            plt.xlabel("w")
            plt.ylabel("f(w)")
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
            Completa `descenso_gradiente_manual`: en cada paso, calcula el gradiente en el `w`
            actual y actualízalo restando `lr * gradiente`. Pruébalo con `w_inicial=6.0, lr=0.3, pasos=10`
            y confirma que `w` se acerca a 3.
            """,
            starter_code="""
            def descenso_gradiente_manual(w_inicial, lr, pasos):
                w = w_inicial
                for _ in range(pasos):
                    g = gradiente(w)
                    w = w - ✏️✏️✏️
                return w


            w_final = descenso_gradiente_manual(w_inicial=6.0, lr=0.3, pasos=10)
            print("w final:", round(w_final, 4))
            """,
            solution_code="""
            def descenso_gradiente_manual(w_inicial, lr, pasos):
                w = w_inicial
                for _ in range(pasos):
                    g = gradiente(w)
                    w = w - lr * g
                return w


            w_final = descenso_gradiente_manual(w_inicial=6.0, lr=0.3, pasos=10)
            print("w final:", round(w_final, 4))
            """,
        )
    )

    cells.append(markdown("## 2. El efecto de la tasa de aprendizaje"))
    cells.append(
        callout(
            "doubt",
            "es",
            "¿Por qué no usar siempre un learning rate gigante para llegar rápido?",
            """
            Porque el excursionista, si da pasos demasiado grandes, puede saltar por encima
            del valle y aterrizar en la ladera de enfrente — o incluso más lejos de donde
            empezó. Un paso pequeño es seguro pero lento; un paso grande es rápido pero puede
            hacerte rebotar sin parar, o divergir. Vamos a verlo con números.
            """,
        )
    )
    cells.append(
        code(
            """
            escenarios = {
                "lr demasiado pequeño (0.01)": descenso_gradiente(-2.0, lr=0.01, pasos=40),
                "lr adecuado (0.3)": descenso_gradiente(-2.0, lr=0.3, pasos=40),
                "lr demasiado grande (1.05)": descenso_gradiente(-2.0, lr=1.05, pasos=40),
            }

            plt.figure(figsize=(8, 4.5))
            colores = ["#2563EB", "#22C55E", "#F97316"]
            for (nombre, trayectoria_e), color in zip(escenarios.items(), colores):
                plt.plot(trayectoria_e, marker=".", label=nombre, color=color)
            plt.axhline(3, color="#94A3B8", linestyle="--", linewidth=0.8, label="mínimo (w=3)")
            plt.title("Mismo punto de partida, tres learning rates distintos")
            plt.xlabel("paso")
            plt.ylabel("w")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()

            for nombre, trayectoria_e in escenarios.items():
                print(f"{nombre}: w tras 40 pasos = {trayectoria_e[-1]:.3f}")
            """
        )
    )
    cells.append(
        markdown(
            """
            Con `lr=0.01` el excursionista apenas se ha movido tras 40 pasos: sigue lejos del
            mínimo. Con `lr=0.3` converge de forma suave. Con `lr=1.05` cada paso lo manda
            **más lejos** que el anterior — está divergiendo, no convergiendo.
            """
        )
    )
    cells.append(
        callout(
            "mistake",
            "es",
            "Un learning rate demasiado alto no siempre se nota a simple vista al principio",
            """
            En una red real, un learning rate demasiado alto a veces produce una pérdida que
            **baja un poco al inicio** y luego empieza a oscilar o se dispara a `NaN`. No
            asumas que "si arrancó bajando, el learning rate está bien": sigue vigilando la
            curva varias épocas más.
            """,
        )
    )
    cells.extend(
        exercise_cell(
            "es",
            prompt="""
            Elige un cuarto learning rate para probar (cualquier número positivo que no sea
            0.01, 0.3 o 1.05). **Antes de ejecutar la celda**, escribe en un comentario qué
            crees que va a pasar (converge suave / lento / diverge). Luego ejecuta y comprueba
            si acertaste.
            """,
            starter_code="""
            # Mi predicción: ✏️✏️✏️ (escribe aquí converge suave / demasiado lento / diverge)
            mi_lr = 0.6
            trayectoria_mia = descenso_gradiente(-2.0, lr=mi_lr, pasos=40)
            print("w tras 40 pasos:", round(trayectoria_mia[-1], 3))
            plt.plot(trayectoria_mia, marker=".", color="#7C3AED")
            plt.axhline(3, color="#94A3B8", linestyle="--")
            plt.title(f"Mi learning rate = {mi_lr}")
            plt.show()
            """,
            solution_code="""
            # Mi predicción: con lr=0.6 el paso es mayor que el óptimo pero sigue dentro del
            # rango que converge (0 < lr < 1 para esta parábola); esperaría oscilación
            # amortiguada que sí llega cerca del mínimo.
            mi_lr = 0.6
            trayectoria_mia = descenso_gradiente(-2.0, lr=mi_lr, pasos=40)
            print("w tras 40 pasos:", round(trayectoria_mia[-1], 3))
            plt.plot(trayectoria_mia, marker=".", color="#7C3AED")
            plt.axhline(3, color="#94A3B8", linestyle="--")
            plt.title(f"Mi learning rate = {mi_lr}")
            plt.show()
            """,
        )
    )

    cells.append(milestone("es", 1))

    cells.append(markdown("## 3. SGD, Momentum y Adam: el mismo MLP, tres optimizadores"))
    cells.append(
        callout(
            "concept",
            "es",
            "Momentum: una bola que conserva algo de su velocidad",
            """
            El excursionista de la Sección 1 decide su paso mirando **solo** la pendiente
            actual, como si empezara de cero en cada paso. **Momentum** es distinto: imagina
            en vez de un excursionista, una bola rodando cuesta abajo — conserva parte de la
            velocidad que ya traía. Eso le permite atravesar pequeños baches sin frenarse del
            todo, y acelerar en tramos donde la pendiente apunta siempre en la misma dirección.

            **Adam** va un paso más allá: adapta el tamaño del paso **por cada parámetro por
            separado**, usando estadísticas acumuladas del gradiente reciente. No necesitas
            memorizar su fórmula — solo saber que combina la idea de Momentum con pasos
            adaptativos, y por eso suele converger rápido "de fábrica" sin apenas ajustar el
            learning rate a mano.
            """,
        )
    )
    cells.append(
        code(
            """
            X_moons, y_moons = make_moons(n_samples=300, noise=0.25, random_state=RANDOM_STATE)

            plt.figure(figsize=(4.5, 4))
            plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="RdBu_r", edgecolor="white")
            plt.title("make_moons (más ruido que en el Notebook 1): el reto para los 3 optimizadores")
            plt.show()
            """
        )
    )
    cells.append(
        code(
            """
            def construir_mlp():
                tf.keras.utils.set_random_seed(RANDOM_STATE)
                return tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(2,)),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ])


            optimizadores = {
                "SGD": tf.keras.optimizers.SGD(learning_rate=0.05),
                "SGD + Momentum": tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
                "Adam": tf.keras.optimizers.Adam(learning_rate=0.05),
            }

            historiales = {}
            for nombre, optimizador in optimizadores.items():
                modelo = construir_mlp()
                modelo.compile(optimizer=optimizador, loss="binary_crossentropy")
                historial = modelo.fit(X_moons, y_moons, epochs=50, verbose=0)
                historiales[nombre] = historial.history["loss"]
                print(f"{nombre:16s} -> pérdida inicial {historial.history['loss'][0]:.3f}, "
                      f"pérdida final {historial.history['loss'][-1]:.3f}")
            """
        )
    )
    cells.append(
        code(
            """
            plt.figure(figsize=(7.5, 4.5))
            colores_opt = {"SGD": "#F97316", "SGD + Momentum": "#2563EB", "Adam": "#22C55E"}
            for nombre, perdidas in historiales.items():
                plt.plot(perdidas, label=nombre, color=colores_opt[nombre])
            plt.title("Misma red, mismos datos, mismo learning rate: solo cambia el optimizador")
            plt.xlabel("época")
            plt.ylabel("pérdida (binary cross-entropy)")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            Con el mismo `learning_rate=0.05` para los tres, verás que SGD puro es el que más
            tarda en bajar la pérdida, Momentum acelera esa bajada, y Adam suele converger más
            rápido y de forma más estable desde las primeras épocas — sin que hayamos tocado
            ningún otro hiperparámetro.
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "es",
            "Adam no es magia, es la opción por defecto razonable",
            """
            Adam suele funcionar bien "tal cual" en muchos problemas, por eso es tan popular
            como punto de partida. Pero "por defecto razonable" no es "siempre óptimo": para
            problemas concretos, SGD + Momentum bien ajustado a veces generaliza mejor. Elegir
            optimizador sigue siendo una decisión que se valida, no una ley fija.
            """,
        )
    )

    cells.append(markdown("## 4. Gradientes que desaparecen: por qué la profundidad no es gratis"))
    cells.append(
        callout(
            "concept",
            "es",
            "Multiplicar muchos números pequeños, muchas veces",
            """
            En el Notebook 2 viste que backpropagation usa la **regla de la cadena**: el
            gradiente de una capa profunda se calcula multiplicando, capa a capa, las
            derivadas locales de todas las capas que hay entre esa capa y la salida. La
            derivada de Sigmoid nunca supera **0.25** (su pico está en $z=0$). Si encadenas 8-10
            capas con Sigmoid, estás multiplicando 8-10 números que valen como mucho 0.25 —
            el resultado se hace minúsculo muy rápido. ReLU, en cambio, tiene derivada 1 (para
            las neuronas activas) o 0, así que no aplasta el gradiente de la misma forma al
            encadenarse.
            """,
        )
    )
    cells.append(
        code(
            """
            def construir_red_profunda(activacion, profundidad=9):
                capas = [tf.keras.layers.Input(shape=(2,))]
                for _ in range(profundidad):
                    capas.append(tf.keras.layers.Dense(16, activation=activacion))
                capas.append(tf.keras.layers.Dense(1, activation="sigmoid"))
                return tf.keras.Sequential(capas)


            def magnitudes_de_gradiente(modelo, X, y):
                \"\"\"Media de |gradiente| del kernel de cada capa Dense, de salida a entrada.\"\"\"
                X_t = tf.convert_to_tensor(X, dtype=tf.float32)
                y_t = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)
                with tf.GradientTape() as tape:
                    prediccion = modelo(X_t, training=True)
                    perdida = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_t, prediccion))
                gradientes = tape.gradient(perdida, modelo.trainable_weights)
                kernels = [g for w, g in zip(modelo.trainable_weights, gradientes) if "kernel" in w.name]
                magnitudes = [float(tf.reduce_mean(tf.abs(g))) for g in kernels]
                magnitudes.reverse()  # de la capa más cercana a la salida hacia la entrada
                return magnitudes


            magnitudes_por_activacion = {}
            for activacion in ["sigmoid", "relu"]:
                tf.keras.utils.set_random_seed(RANDOM_STATE)
                red = construir_red_profunda(activacion)
                magnitudes_por_activacion[activacion] = magnitudes_de_gradiente(red, X_moons, y_moons)
                print(activacion, [f"{m:.2e}" for m in magnitudes_por_activacion[activacion]])
            """
        )
    )
    cells.append(
        code(
            """
            plot_gradient_magnitude_by_layer(magnitudes_por_activacion, lang="es")
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            Con esta arquitectura exacta (9 capas ocultas de 16 neuronas, semilla 42), la capa
            Sigmoid más cercana a la salida tiene un gradiente medio de **~4.0 × 10⁻²**, y la
            capa más cercana a la entrada cae hasta **~1.6 × 10⁻⁸** — una caída de **más de 6
            órdenes de magnitud**. Con ReLU, todas las capas se mantienen en un rango parecido,
            entre **~1.6 × 10⁻⁴ y ~9.4 × 10⁻⁴** sin tendencia a desaparecer. Esa capa más
            cercana a la entrada en la red Sigmoid recibe un gradiente tan diminuto que,
            prácticamente, **deja de aprender**: sus weights apenas cambian entrenamiento tras
            entrenamiento.
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "es",
            "¿Entonces nunca debo usar Sigmoid en capas ocultas?",
            """
            Como regla práctica: usa ReLU (o variantes como Leaky ReLU) en las **capas
            ocultas** de redes profundas, y reserva Sigmoid para la **capa de salida** en
            clasificación binaria, donde solo hay una capa y el problema no aparece. Es
            justo lo que ya venías haciendo desde el Notebook 1 — ahora sabes el porqué.
            """,
        )
    )

    cells.append(milestone("es", 2))

    cells.append(markdown("## 5. '¿Por qué mi red no aprende?' — tabla de diagnóstico rápido"))
    cells.append(
        markdown(
            """
            | Síntoma | Causa probable | Qué revisar |
            |---|---|---|
            | La pérdida está plana desde la época 1 | Learning rate demasiado pequeño / gradiente desvanecido / neuronas ReLU "muertas" | Sube el learning rate; revisa la magnitud del gradiente por capa (Sección 4); comprueba cuántas activaciones ReLU dan siempre 0 |
            | La pérdida explota o se vuelve `NaN` | Learning rate demasiado alto / inestabilidad numérica | Baja el learning rate; revisa si hay valores de entrada sin normalizar o `logits` extremos |
            | La pérdida de entrenamiento mejora pero la de validación empeora | Sobreajuste (el modelo memoriza en vez de generalizar) | No se cubre aquí — es el tema central del Notebook 4 |
            | La pérdida baja muy despacio, de forma constante | Learning rate algo bajo, u optimizador sin momentum en un terreno con curvatura difícil | Prueba Momentum o Adam (Sección 3) antes de tocar la arquitectura |
            | La pérdida oscila sin bajar de forma clara | Learning rate demasiado alto para ese optimizador | Reduce el learning rate; compara con la Sección 2 |
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "es",
            "Antes de cambiar la arquitectura, revisa el optimizador",
            """
            Es tentador, cuando una red no aprende, añadir capas o neuronas de inmediato. Pero
            muchos de los síntomas más comunes (pérdida plana, pérdida que explota, convergencia
            lentísima) tienen que ver con el **learning rate** o el **optimizador**, no con el
            tamaño de la red. Revisa primero lo barato de cambiar.
            """,
        )
    )

    cells.append(section_header("es"))
    cells.append(
        quiz_question(
            "es", 1,
            "¿Qué añade Momentum sobre el descenso de gradiente (SGD) puro?",
            [
                "Nada, son matemáticamente idénticos",
                "Conserva parte de la 'velocidad' de pasos anteriores, suavizando y acelerando la trayectoria",
                "Elimina por completo la necesidad de elegir un learning rate",
                "Solo funciona si la red tiene una única capa",
            ],
            1,
            "Momentum acumula una fracción del paso anterior, como una bola rodando que conserva velocidad, en vez de decidir cada paso solo con la pendiente actual.",
        )
    )
    cells.append(
        quiz_question(
            "es", 2,
            "En un gráfico de magnitud de gradiente por capa (como el de la Sección 4), ¿cómo se reconoce un problema de gradientes que desaparecen?",
            [
                "Todas las capas tienen una magnitud similar, sin caer con la profundidad",
                "La magnitud cae varios órdenes de magnitud a medida que nos alejamos de la capa de salida",
                "La magnitud sube exponencialmente cerca de la entrada",
                "El gráfico no tiene relación con el problema; hay que mirar la pérdida",
            ],
            1,
            "El patrón característico es una caída pronunciada (multiplicativa, por eso se ve en escala log) de la magnitud del gradiente en las capas más cercanas a la entrada.",
        )
    )
    cells.append(
        quiz_question(
            "es", 3,
            "Según la tabla de diagnóstico de la Sección 5, ¿qué deberías sospechar primero si la pérdida de entrenamiento y de validación se separan (la de validación empeora)?",
            [
                "Un learning rate demasiado alto",
                "Sobreajuste — tema que se desarrolla en el Notebook 4",
                "Un gradiente desvanecido",
                "Un error de tipeo en la función de pérdida",
            ],
            1,
            "Train mejorando mientras validación empeora es la firma clásica de sobreajuste, que este notebook solo señala — se trabaja en profundidad en el Notebook 4.",
        )
    )
    cells.append(
        quiz_question(
            "es", 4,
            "En el experimento de la Sección 2, ¿qué le pasó a la trayectoria con `lr=1.05`?",
            [
                "Convergió más rápido que con `lr=0.3`",
                "Se quedó exactamente en el punto de partida",
                "Cada paso se alejó más del mínimo que el anterior: diverge",
                "Llegó al mínimo pero osciló ligeramente alrededor de él",
            ],
            2,
            "Con un learning rate mayor que el rango estable para esta parábola, cada actualización sobrepasa el mínimo por un margen creciente: la trayectoria diverge en vez de converger.",
        )
    )
    cells.append(
        quiz_question(
            "es", 5,
            "¿Qué hace Adam de forma distinta a Momentum, a nivel conceptual (sin fórmulas)?",
            [
                "Adam no usa learning rate en absoluto",
                "Adam adapta el tamaño del paso por cada parámetro individualmente, usando estadísticas acumuladas del gradiente",
                "Adam solo sirve para clasificación con Softmax",
                "Adam es idéntico a SGD sin momentum",
            ],
            1,
            "Adam combina la idea de Momentum con pasos adaptativos por parámetro, calculados a partir de estadísticas acumuladas de gradientes recientes (media y varianza, a grandes rasgos).",
        )
    )
    cells.append(
        open_question(
            "es", 6,
            "¿Por qué crees que Adam se convirtió en la opción por defecto en la industria? Explícalo con tus propias palabras.",
            [
                "Menciona que suele converger rápido sin apenas ajustar hiperparámetros a mano.",
                "Conecta la idea de pasos adaptativos por parámetro con problemas donde distintos weights necesitan distinta escala de actualización.",
                "Reconoce que 'por defecto razonable' no significa 'siempre el mejor' — sigue siendo una elección que se valida.",
            ],
        )
    )
    cells.append(
        open_question(
            "es", 7,
            "Un compañero de equipo te dice: 'la pérdida de mi red está plana desde la época 1'. Describe, paso a paso, cómo lo diagnosticarías.",
            [
                "Empieza revisando el learning rate (¿demasiado pequeño?) antes de tocar la arquitectura.",
                "Menciona revisar la magnitud del gradiente por capa para descartar un gradiente desvanecido.",
                "Contempla la posibilidad de neuronas ReLU muertas o datos de entrada sin normalizar.",
                "Sigue un orden razonado en vez de cambiar cosas al azar, apoyándose en la tabla de diagnóstico de la Sección 5.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 ¡Enhorabuena! Completaste el Notebook 3: Optimizadores 🎉",
                "Ya sabes cómo un optimizador usa el gradiente para dar pasos, qué aporta cada "
                "uno de SGD/Momentum/Adam, y cómo diagnosticar una red que no aprende. En el "
                "Notebook 4 entrenamos de verdad: épocas, batches y cómo evitar el sobreajuste.",
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
            # NeuroTrain Lab — Notebook 3: Optimizers

            **Topic:** how an optimizer actually uses the gradient (Notebook 2) to move the
            weights, what role the learning rate plays, how SGD, Momentum, and Adam differ,
            and why gradients sometimes "vanish" in deep networks.

            > Notebook 3 of 4. You already know how to compute the loss's gradient. Now we
            > use that gradient to **take steps** that reduce the error — and we'll see not
            > all steps are equal.
            """
        )
    )
    cells.append(
        markdown(
            """
            ## 🎯 What you'll learn in this notebook

            1. What gradient descent does, step by step, over a loss function.
            2. Why the learning rate is the most delicate decision in all of training.
            3. What Momentum adds over plain SGD, and what makes Adam different (conceptually).
            4. How to recognize a vanishing gradient in a deep network, and why it happens.
            5. A quick diagnostic for "my network isn't learning" before touching code blindly.

            **Mental map:** `gradient → descent step → learning rate → SGD → Momentum → Adam → vanishing gradients`
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
            from sklearn.datasets import make_moons

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "data" / "breast_cancer_wisconsin.csv").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent.parent
            sys.path.insert(0, str(PROJECT_ROOT / "src"))

            from neurotrain.celebrations import celebrate
            from neurotrain.visualization import plot_gradient_magnitude_by_layer

            RANDOM_STATE = 42
            np.random.seed(RANDOM_STATE)
            tf.keras.utils.set_random_seed(RANDOM_STATE)

            print("NumPy:", np.__version__, "| TensorFlow:", tf.__version__)
            print("Project root:", PROJECT_ROOT)
            """
        )
    )

    cells.append(
        markdown(
            """
            ## 0. Where we are

            In Notebook 2 you learned to compute $\\partial L / \\partial w$ for every weight
            in the network — the gradient tells you **which direction** the loss grows in.
            What's missing is the rule that decides, gradient in hand, **how much and how** to
            move each weight. That rule is the **optimizer**.
            """
        )
    )

    cells.append(markdown("## 1. Gradient descent: the hiker in the fog"))
    cells.append(
        callout(
            "concept",
            "en",
            "A hiker who can't see the valley",
            """
            Picture someone descending a mountain wrapped in thick fog. They can't see the
            valley or the full map — they can only feel, underfoot, which way the ground
            slopes down **at that exact spot**. They take a step in that direction, feel the
            slope again, take another step, and so on.

            That's exactly gradient descent: at each point we only know the **local**
            gradient (the slope under our feet), not the full shape of the loss function. We
            update like this:

            $$w \\leftarrow w - \\eta \\cdot \\frac{\\partial L}{\\partial w}$$

            where $\\eta$ (eta) is the **learning rate** — the size of the hiker's step.
            """,
        )
    )
    cells.append(
        code(
            """
            def f(w):
                \"\"\"A toy 1D 'loss': a bowl with its minimum at w=3.\"\"\"
                return (w - 3) ** 2 + 1


            def gradient(w):
                return 2 * (w - 3)


            def gradient_descent(w_start, lr, steps):
                trajectory = [w_start]
                w = w_start
                for _ in range(steps):
                    w = w - lr * gradient(w)
                    trajectory.append(w)
                return np.array(trajectory)


            trajectory = gradient_descent(w_start=-2.0, lr=0.2, steps=15)
            print("w positions:", trajectory.round(3))
            print("Final loss:", round(f(trajectory[-1]), 4), "(true minimum is 1.0 at w=3)")
            """
        )
    )
    cells.append(
        code(
            """
            w_curve = np.linspace(-3, 8, 200)
            plt.figure(figsize=(7, 4.5))
            plt.plot(w_curve, f(w_curve), color="#94A3B8", label="f(w) = (w-3)² + 1")
            plt.plot(trajectory, f(trajectory), "o-", color="#7C3AED", label="descent steps")
            for i in range(len(trajectory) - 1):
                plt.annotate(
                    "", xy=(trajectory[i + 1], f(trajectory[i + 1])),
                    xytext=(trajectory[i], f(trajectory[i])),
                    arrowprops=dict(arrowstyle="->", color="#F97316", alpha=0.6),
                )
            plt.scatter([3], [1], color="#22C55E", zorder=5, label="true minimum (w=3)")
            plt.title("The hiker descending the loss bowl, step by step")
            plt.xlabel("w")
            plt.ylabel("f(w)")
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
            Complete `gradient_descent_manual`: at each step, compute the gradient at the
            current `w` and update it by subtracting `lr * gradient`. Test it with
            `w_start=6.0, lr=0.3, steps=10` and confirm `w` gets close to 3.
            """,
            starter_code="""
            def gradient_descent_manual(w_start, lr, steps):
                w = w_start
                for _ in range(steps):
                    g = gradient(w)
                    w = w - ✏️✏️✏️
                return w


            final_w = gradient_descent_manual(w_start=6.0, lr=0.3, steps=10)
            print("Final w:", round(final_w, 4))
            """,
            solution_code="""
            def gradient_descent_manual(w_start, lr, steps):
                w = w_start
                for _ in range(steps):
                    g = gradient(w)
                    w = w - lr * g
                return w


            final_w = gradient_descent_manual(w_start=6.0, lr=0.3, steps=10)
            print("Final w:", round(final_w, 4))
            """,
        )
    )

    cells.append(markdown("## 2. The effect of the learning rate"))
    cells.append(
        callout(
            "doubt",
            "en",
            "Why not always use a huge learning rate to get there faster?",
            """
            Because if the hiker takes steps that are too large, they can leap right over the
            valley and land on the opposite slope — or even farther from where they started.
            A small step is safe but slow; a large step is fast but can bounce forever, or
            diverge outright. Let's see it with numbers.
            """,
        )
    )
    cells.append(
        code(
            """
            scenarios = {
                "lr too small (0.01)": gradient_descent(-2.0, lr=0.01, steps=40),
                "good lr (0.3)": gradient_descent(-2.0, lr=0.3, steps=40),
                "lr too large (1.05)": gradient_descent(-2.0, lr=1.05, steps=40),
            }

            plt.figure(figsize=(8, 4.5))
            colors = ["#2563EB", "#22C55E", "#F97316"]
            for (name, scenario_trajectory), color in zip(scenarios.items(), colors):
                plt.plot(scenario_trajectory, marker=".", label=name, color=color)
            plt.axhline(3, color="#94A3B8", linestyle="--", linewidth=0.8, label="minimum (w=3)")
            plt.title("Same starting point, three different learning rates")
            plt.xlabel("step")
            plt.ylabel("w")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()

            for name, scenario_trajectory in scenarios.items():
                print(f"{name}: w after 40 steps = {scenario_trajectory[-1]:.3f}")
            """
        )
    )
    cells.append(
        markdown(
            """
            With `lr=0.01` the hiker has barely moved after 40 steps: still far from the
            minimum. With `lr=0.3` it converges smoothly. With `lr=1.05` every step sends it
            **farther** than the last — it's diverging, not converging.
            """
        )
    )
    cells.append(
        callout(
            "mistake",
            "en",
            "A learning rate that's too high isn't always obvious at first glance",
            """
            In a real network, too high a learning rate sometimes produces a loss that
            **drops a bit at first** and then starts oscillating or blows up to `NaN`. Don't
            assume "if it started dropping, the learning rate is fine" — keep watching the
            curve for several more epochs.
            """,
        )
    )
    cells.extend(
        exercise_cell(
            "en",
            prompt="""
            Pick a fourth learning rate to try (any positive number other than 0.01, 0.3, or
            1.05). **Before running the cell**, write in a comment what you think will happen
            (smooth convergence / too slow / diverges). Then run it and check if you were right.
            """,
            starter_code="""
            # My prediction: ✏️✏️✏️ (write smooth convergence / too slow / diverges here)
            my_lr = 0.6
            my_trajectory = gradient_descent(-2.0, lr=my_lr, steps=40)
            print("w after 40 steps:", round(my_trajectory[-1], 3))
            plt.plot(my_trajectory, marker=".", color="#7C3AED")
            plt.axhline(3, color="#94A3B8", linestyle="--")
            plt.title(f"My learning rate = {my_lr}")
            plt.show()
            """,
            solution_code="""
            # My prediction: with lr=0.6 the step is larger than optimal but still within the
            # range that converges (0 < lr < 1 for this parabola); I'd expect damped
            # oscillation that still ends up close to the minimum.
            my_lr = 0.6
            my_trajectory = gradient_descent(-2.0, lr=my_lr, steps=40)
            print("w after 40 steps:", round(my_trajectory[-1], 3))
            plt.plot(my_trajectory, marker=".", color="#7C3AED")
            plt.axhline(3, color="#94A3B8", linestyle="--")
            plt.title(f"My learning rate = {my_lr}")
            plt.show()
            """,
        )
    )

    cells.append(milestone("en", 1))

    cells.append(markdown("## 3. SGD, Momentum, and Adam: the same MLP, three optimizers"))
    cells.append(
        callout(
            "concept",
            "en",
            "Momentum: a ball that keeps some of its speed",
            """
            The hiker from Section 1 decides each step by looking **only** at the current
            slope, as if starting fresh every time. **Momentum** is different: instead of a
            hiker, picture a ball rolling downhill — it keeps part of the velocity it already
            had. That lets it roll through small bumps without stalling out, and speed up
            along stretches where the slope keeps pointing the same way.

            **Adam** goes a step further: it adapts the step size **separately for each
            parameter**, using accumulated statistics of recent gradients. You don't need to
            memorize its formula — just know it combines the idea of Momentum with adaptive
            per-parameter steps, which is why it tends to converge fast "out of the box"
            without much manual learning-rate tuning.
            """,
        )
    )
    cells.append(
        code(
            """
            X_moons, y_moons = make_moons(n_samples=300, noise=0.25, random_state=RANDOM_STATE)

            plt.figure(figsize=(4.5, 4))
            plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap="RdBu_r", edgecolor="white")
            plt.title("make_moons (noisier than Notebook 1): the challenge for our 3 optimizers")
            plt.show()
            """
        )
    )
    cells.append(
        code(
            """
            def build_mlp():
                tf.keras.utils.set_random_seed(RANDOM_STATE)
                return tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(2,)),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(16, activation="relu"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ])


            optimizers = {
                "SGD": tf.keras.optimizers.SGD(learning_rate=0.05),
                "SGD + Momentum": tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
                "Adam": tf.keras.optimizers.Adam(learning_rate=0.05),
            }

            histories = {}
            for name, optimizer in optimizers.items():
                model = build_mlp()
                model.compile(optimizer=optimizer, loss="binary_crossentropy")
                history = model.fit(X_moons, y_moons, epochs=50, verbose=0)
                histories[name] = history.history["loss"]
                print(f"{name:16s} -> initial loss {history.history['loss'][0]:.3f}, "
                      f"final loss {history.history['loss'][-1]:.3f}")
            """
        )
    )
    cells.append(
        code(
            """
            plt.figure(figsize=(7.5, 4.5))
            optimizer_colors = {"SGD": "#F97316", "SGD + Momentum": "#2563EB", "Adam": "#22C55E"}
            for name, losses in histories.items():
                plt.plot(losses, label=name, color=optimizer_colors[name])
            plt.title("Same network, same data, same learning rate: only the optimizer changes")
            plt.xlabel("epoch")
            plt.ylabel("loss (binary cross-entropy)")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            With the same `learning_rate=0.05` for all three, you'll see plain SGD takes the
            longest to bring the loss down, Momentum speeds that descent up, and Adam
            typically converges faster and more steadily from the earliest epochs — without
            touching any other hyperparameter.
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "en",
            "Adam isn't magic, it's a reasonable default",
            """
            Adam tends to work well "as is" on many problems, which is why it's such a
            popular starting point. But "reasonable default" isn't "always optimal": for
            specific problems, well-tuned SGD + Momentum sometimes generalizes better.
            Choosing an optimizer is still a decision you validate, not a fixed rule.
            """,
        )
    )

    cells.append(markdown("## 4. Vanishing gradients: why depth isn't free"))
    cells.append(
        callout(
            "concept",
            "en",
            "Multiplying many small numbers, many times",
            """
            In Notebook 2 you saw backpropagation uses the **chain rule**: the gradient at a
            deep layer is computed by multiplying, layer by layer, the local derivatives of
            every layer between it and the output. Sigmoid's derivative never exceeds **0.25**
            (it peaks at $z=0$). Chain together 8-10 layers of Sigmoid and you're multiplying
            8-10 numbers each at most 0.25 — the result shrinks to almost nothing very fast.
            ReLU, by contrast, has derivative 1 (for active neurons) or 0, so it doesn't
            crush the gradient the same way when chained.
            """,
        )
    )
    cells.append(
        code(
            """
            def build_deep_network(activation, depth=9):
                layers = [tf.keras.layers.Input(shape=(2,))]
                for _ in range(depth):
                    layers.append(tf.keras.layers.Dense(16, activation=activation))
                layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
                return tf.keras.Sequential(layers)


            def gradient_magnitudes(model, X, y):
                \"\"\"Mean |gradient| of each Dense layer's kernel, output-to-input order.\"\"\"
                X_t = tf.convert_to_tensor(X, dtype=tf.float32)
                y_t = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)
                with tf.GradientTape() as tape:
                    prediction = model(X_t, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_t, prediction))
                grads = tape.gradient(loss, model.trainable_weights)
                kernels = [g for w, g in zip(model.trainable_weights, grads) if "kernel" in w.name]
                magnitudes = [float(tf.reduce_mean(tf.abs(g))) for g in kernels]
                magnitudes.reverse()  # from the layer closest to the output toward the input
                return magnitudes


            magnitudes_by_activation = {}
            for activation in ["sigmoid", "relu"]:
                tf.keras.utils.set_random_seed(RANDOM_STATE)
                network = build_deep_network(activation)
                magnitudes_by_activation[activation] = gradient_magnitudes(network, X_moons, y_moons)
                print(activation, [f"{m:.2e}" for m in magnitudes_by_activation[activation]])
            """
        )
    )
    cells.append(
        code(
            """
            plot_gradient_magnitude_by_layer(magnitudes_by_activation, lang="en")
            plt.show()
            """
        )
    )
    cells.append(
        markdown(
            """
            With this exact architecture (9 hidden layers of 16 neurons, seed 42), the Sigmoid
            layer closest to the output has a mean gradient of **~4.0 × 10⁻²**, and the layer
            closest to the input drops to **~1.6 × 10⁻⁸** — a fall of **more than 6 orders of
            magnitude**. With ReLU, every layer stays in a similar range, between **~1.6 × 10⁻⁴
            and ~9.4 × 10⁻⁴**, with no tendency to vanish. That layer closest to the input in
            the Sigmoid network receives such a tiny gradient it essentially **stops learning**:
            its weights barely change from one training pass to the next.
            """
        )
    )
    cells.append(
        callout(
            "doubt",
            "en",
            "So should I never use Sigmoid in hidden layers?",
            """
            As a practical rule: use ReLU (or variants like Leaky ReLU) in the **hidden
            layers** of deep networks, and reserve Sigmoid for the **output layer** in binary
            classification, where there's only one such layer and the problem doesn't show
            up. That's exactly what you've been doing since Notebook 1 — now you know why.
            """,
        )
    )

    cells.append(milestone("en", 2))

    cells.append(markdown("## 5. 'Why isn't my network learning?' — quick diagnostic table"))
    cells.append(
        markdown(
            """
            | Symptom | Likely cause | What to check |
            |---|---|---|
            | Loss is flat from epoch 1 | Learning rate too small / vanishing gradient / dead ReLU neurons | Raise the learning rate; check per-layer gradient magnitude (Section 4); check how many ReLU activations always output 0 |
            | Loss explodes or becomes `NaN` | Learning rate too high / numerical instability | Lower the learning rate; check for unnormalized inputs or extreme logits |
            | Training loss improves but validation loss gets worse | Overfitting (the model is memorizing, not generalizing) | Not covered here — this is the central topic of Notebook 4 |
            | Loss drops very slowly but steadily | Learning rate somewhat low, or an optimizer with no momentum on tricky curvature | Try Momentum or Adam (Section 3) before touching the architecture |
            | Loss oscillates without clearly dropping | Learning rate too high for that optimizer | Lower the learning rate; compare with Section 2 |
            """
        )
    )
    cells.append(
        callout(
            "remember",
            "en",
            "Check the optimizer before changing the architecture",
            """
            When a network isn't learning, it's tempting to immediately add layers or
            neurons. But many of the most common symptoms (flat loss, exploding loss,
            painfully slow convergence) come down to the **learning rate** or the
            **optimizer**, not network size. Check the cheap fixes first.
            """,
        )
    )

    cells.append(section_header("en"))
    cells.append(
        quiz_question(
            "en", 1,
            "What does Momentum add over plain gradient descent (SGD)?",
            [
                "Nothing, they're mathematically identical",
                "It keeps part of the 'velocity' from previous steps, smoothing and speeding up the trajectory",
                "It completely removes the need to choose a learning rate",
                "It only works if the network has a single layer",
            ],
            1,
            "Momentum accumulates a fraction of the previous step, like a rolling ball that keeps its speed, instead of deciding each step from the current slope alone.",
        )
    )
    cells.append(
        quiz_question(
            "en", 2,
            "In a per-layer gradient-magnitude chart (like the one in Section 4), how do you recognize a vanishing-gradient problem?",
            [
                "Every layer has a similar magnitude, with no drop across depth",
                "The magnitude drops several orders of magnitude the farther you get from the output layer",
                "The magnitude rises exponentially near the input",
                "The chart is unrelated to the problem; you should look at the loss instead",
            ],
            1,
            "The telltale pattern is a steep (multiplicative, hence visible on a log scale) drop in gradient magnitude in the layers closest to the input.",
        )
    )
    cells.append(
        quiz_question(
            "en", 3,
            "Per the diagnostic table in Section 5, what should you suspect first if training loss and validation loss diverge (validation gets worse)?",
            [
                "A learning rate that's too high",
                "Overfitting — a topic developed further in Notebook 4",
                "A vanishing gradient",
                "A typo in the loss function",
            ],
            1,
            "Train improving while validation worsens is the classic signature of overfitting, which this notebook only flags — it's covered in depth in Notebook 4.",
        )
    )
    cells.append(
        quiz_question(
            "en", 4,
            "In the Section 2 experiment, what happened to the trajectory with `lr=1.05`?",
            [
                "It converged faster than with `lr=0.3`",
                "It stayed exactly at the starting point",
                "Each step moved farther from the minimum than the last: it diverges",
                "It reached the minimum but oscillated slightly around it",
            ],
            2,
            "With a learning rate above the stable range for this parabola, each update overshoots the minimum by a growing margin: the trajectory diverges instead of converging.",
        )
    )
    cells.append(
        quiz_question(
            "en", 5,
            "What does Adam do differently from Momentum, conceptually (no formulas)?",
            [
                "Adam doesn't use a learning rate at all",
                "Adam adapts the step size for each parameter individually, using accumulated gradient statistics",
                "Adam only works for Softmax classification",
                "Adam is identical to SGD without momentum",
            ],
            1,
            "Adam combines the idea of Momentum with per-parameter adaptive steps, computed from accumulated statistics of recent gradients (roughly, mean and variance).",
        )
    )
    cells.append(
        open_question(
            "en", 6,
            "Why do you think Adam became the default choice in industry? Explain in your own words.",
            [
                "Mentions it tends to converge fast with little manual hyperparameter tuning.",
                "Connects the idea of per-parameter adaptive steps to problems where different weights need different update scales.",
                "Acknowledges 'reasonable default' doesn't mean 'always best' — it's still a choice you validate.",
            ],
        )
    )
    cells.append(
        open_question(
            "en", 7,
            "A teammate tells you: 'my network's loss has been flat since epoch 1'. Walk through, step by step, how you'd diagnose it.",
            [
                "Starts by checking the learning rate (too small?) before touching the architecture.",
                "Mentions checking per-layer gradient magnitude to rule out a vanishing gradient.",
                "Considers dead ReLU neurons or unnormalized input data as possibilities.",
                "Follows a reasoned order instead of changing things at random, leaning on the Section 5 diagnostic table.",
            ],
        )
    )

    cells.append(
        code(
            """
            celebrate(
                "🎉 Congratulations! You finished Notebook 3: Optimizers 🎉",
                "You now know how an optimizer turns a gradient into a step, what each of "
                "SGD/Momentum/Adam brings to the table, and how to diagnose a network that "
                "isn't learning. In Notebook 4 we train for real: epochs, batches, and how to "
                "avoid overfitting.",
            )
            """
        )
    )

    return cells
