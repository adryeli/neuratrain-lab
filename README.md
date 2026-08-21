# NeuroTrain Lab

### 🚀 [Accede a la clase introductoria aquí](https://neuratrain-lab.streamlit.app/)

Clase introductorio de redes neuronales artificiales, pensado para **aprender haciendo**: 4 temas, cada uno con un notebook de ejercicios reales y una página interactiva en una app Streamlit, que termina en un laboratorio donde entrenas un modelo de verdad sobre datos clínicos.

Esta aplicación esta en V.1, puede sufrir modificaciones a lo largo del tiempo.

## 1. Explicación rápida

El curso responde, en orden, a 4 preguntas:

1. **El Perceptrón** — ¿qué calcula una neurona, y para qué sirven ReLU/Sigmoid/Softmax?
2. **Pérdida y Backpropagation** — ¿cómo sabe la red que se equivocó, y qué debe cambiar?
3. **Optimizadores** — ¿cómo decide cuánto cambiar cada parámetro (SGD, Momentum, Adam)?
4. **Entrenamiento y sobreajuste** — ¿cómo organizamos miles de actualizaciones sin memorizar de más (epochs, batches, EarlyStopping, Dropout)?

Cada tema tiene:

- Un **notebook con ejercicios** (marcados con ✏️✏️✏️), en `notebooks/es/` y `notebooks/en/` — 8 notebooks en total, autoevaluación y celebración de cierre incluidas.
- Una **página en la app Streamlit** (`app_pages/`) con teoría, analogías, un vídeo y una mini-demo interactiva.

Al terminar los 4 temas, el **Laboratorio · Modo Experimento** deja entrenar una red neuronal real sobre el dataset clínico, comparándola con una regresión logística.

La aplicación no intenta sustituir una decisión médica. El caso clínico se utiliza porque conecta con Clinical Data Analytics y permite explicar sensibilidad, especificidad, falsos negativos y validación responsable.

## 2. Arquitectura

```text
neurotrain-lab/
├── app.py                          Punto de entrada de la app Streamlit
├── app_pages/                      Inicio, 4 temas del recorrido, laboratorio
├── notebooks/
│   ├── es/  01..04                 Los 4 notebooks, en español
│   ├── en/  01..04                 Los 4 notebooks, en inglés
│   └── material_adicional/         Notebook bonus (DOTCSV / Elizabeth Sena)
├── scripts/
│   ├── notebook_builders/          Generador de los 8 notebooks
│   ├── build_notebooks.py          Regenera los notebooks desde código
│   ├── verify_notebooks.py         Ejecuta cada notebook de principio a fin
│   ├── export_dataset.py           Exporta el dataset a CSV
│   └── generate_project_guide.py   Genera docs/GUIA_DEL_PROYECTO.pdf
├── src/neurotrain/
│   ├── config.py                   Hiperparámetros tipados
│   ├── data.py                     Carga, split, escalado
│   ├── modeling.py                 Baseline + red neuronal
│   ├── evaluation.py                Métricas de clasificación
│   ├── visualization.py            Gráficos compartidos (notebooks + app)
│   └── celebrations.py             Confeti de cierre de cada notebook
├── data/                           Dataset real (CSV)
├── docs/                           Esta guía, guía de vídeos, Masterclass, uso responsable
├── tests/
├── Dockerfile, docker-compose.yml, .dockerignore
└── requirements.txt, requirements-app.txt
```

Los notebooks son deliberadamente explícitos y bien comentados. La app y los notebooks reutilizan los mismos módulos de `src/neurotrain/` para evitar duplicación y facilitar tests.

## 3. Dataset real

Se usa **Breast Cancer Wisconsin (Diagnostic)**:

- 569 registros.
- 30 variables numéricas.
- Clasificación binaria: benigno (`B`) o maligno (`M`).
- Sin valores ausentes.
- Licencia CC BY 4.0.

Fuente oficial: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
DOI: https://doi.org/10.24432/C5DW2B

El CSV incluido se exportó de la copia distribuida por scikit-learn. La clase positiva se define como `M = 1`, para que la sensibilidad corresponda a la detección de registros malignos.

## 4. Ejecutar en local

### Herramientas

- Python 3.11 o 3.12.
- VS Code (opcional, con extensiones Python y Jupyter) o Google Colab si prefieres no instalar nada localmente.

### Crear y activar el entorno

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.

Si PowerShell bloquea la activación, ejecuta una sola vez:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Orden recomendado

1. Recorre los notebooks 1 a 4 (en `notebooks/es/` o `notebooks/en/`, según tu idioma), celda a celda — cada uno termina con una autoevaluación y una celebración.
2. Lanza la app:

```powershell
streamlit run app.py
```

3. Sigue el **Recorrido guiado** (barra lateral) y termina en **Laboratorio · Modo Experimento**.

Streamlit mostrará una URL local, normalmente `http://localhost:8501`.

### Usar Google Colab en vez de instalar localmente

Sube el `.ipynb` a https://colab.research.google.com (*Archivo → Subir cuaderno*). Los Notebooks 1-3 no necesitan ningún archivo adicional. El Notebook 4 necesita `data/breast_cancer_wisconsin.csv`: súbelo también con el panel de archivos de Colab antes de ejecutar las celdas que lo cargan.

## 5. Ejecutar con Docker (Pendiente)

```bash
docker compose up --build
```

La imagen usa `requirements-app.txt` (sin Jupyter ni PyTorch, que solo hacen falta para los notebooks) y `tensorflow-cpu` para mantenerla ligera. La app queda disponible en `http://localhost:8501`.

## 6. Verificación

Desde la raíz, con `.venv` activado:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python scripts/verify_notebooks.py
```

Los tests verifican:

- Contrato del dataset.
- Ausencia de solapamiento entre train, validation y test.
- Ajuste del scaler solo sobre train.
- Cálculo conocido de sensibilidad/especificidad.
- Validación de hiperparámetros.
- Forma de salida del modelo cuando TensorFlow está instalado.
- Que el confeti de cierre de cada notebook se genera sin dependencias externas.
- Estructura de los 8 notebooks generados (autoevaluación, ejercicios, paridad ES/EN).

`scripts/verify_notebooks.py` ejecuta cada notebook de principio a fin con los ejercicios ya resueltos, confirmando que el recorrido completo funciona.

## 7. Usar la aplicación

La app tiene 3 zonas en la barra lateral:

- **Inicio** — resumen del recorrido, progreso, material adicional (notebook bonus, presentación, TensorFlow Playground).
- **Recorrido guiado** — 4 páginas de tema, con teoría, vídeo, mini-demo interactiva y descarga del notebook correspondiente.
- **Laboratorio · Modo Experimento** — el entrenamiento real. La barra lateral permite cambiar arquitectura (32→16, 64→32, 128→64), épocas máximas, batch size, dropout, EarlyStopping y paciencia.

Después del entrenamiento puedes:

- Ver el progreso por época en vivo.
- Comparar curvas de train y validation.
- Contrastar la ANN con regresión logística.
- Cambiar el umbral y observar sensibilidad/especificidad.
- Inspeccionar registros reservados para test.
- Ver el código real detrás de cada paso (pestaña "Bajo el capó").
- Descargar el resultado del experimento como JSON.

### Experimento recomendado

| Variante | Arquitectura | Dropout | EarlyStopping | Qué observar |
|---|---|---:|---:|---|
| A | 128→64 | 0 | No | Separación entre `loss` y `val_loss` |
| B | 32→16 | 0.30 | Sí | Época de parada y generalización |

No cambies varias cosas a la vez: si lo haces, no sabrás cuál causó el resultado.

## 8. Material adicional

- `docs/GUIA_ESTUDIO_Y_VIDEOS.md` — recorrido de vídeos recomendado por tema.
- `docs/Masterclass_ANN.pdf` — la presentación de la masterclass.
- `notebooks/material_adicional/3_maneras_de_programar_una_red_neuronal.ipynb` — notebook bonus (DOTCSV, adaptado por Elizabeth Sena): construye una red desde cero con `make_circles`, luego resuélvela con TensorFlow, Keras y scikit-learn.
- https://playground.tensorflow.org/ — actividad interactiva para experimentar con neuronas, learning rate y funciones de activación.

## 9. Buenas prácticas incorporadas

- Semillas reproducibles.
- Split estratificado 70/15/15.
- Scaler ajustado solo con train.
- Validation separada de test.
- `restore_best_weights=True`.
- Baseline sencillo.
- Métricas más allá de accuracy.
- Clase positiva explícita.
- Tests y módulos reutilizables.
- Fuente, DOI, licencia y límites de uso documentados.

¿Dudas? Contacta con Elizabeth Sena en LinkedIn: https://www.linkedin.com/in/elizabeth-sena
