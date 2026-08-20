# Dataset y uso responsable

## Fuente

NeuroTrain Lab usa **Breast Cancer Wisconsin (Diagnostic)**, publicado por UCI Machine Learning Repository.

- Fuente oficial: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- DOI: https://doi.org/10.24432/C5DW2B
- Licencia del dataset: CC BY 4.0.
- Creadores: William Wolberg, Olvi Mangasarian, Nick Street y W. Street.
- Referencia: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository.

El CSV incluido se exportó desde `sklearn.datasets.load_breast_cancer`. Contiene 569 registros, 30 variables reales y ninguna observación ausente. Las variables describen núcleos celulares extraídos de imágenes digitalizadas de punciones aspirativas con aguja fina de masas mamarias.

En el proyecto:

- `M` significa maligno y se codifica como clase positiva `1`.
- `B` significa benigno y se codifica como `0`.
- El identificador clínico original no se incluye.

## Límites

Este proyecto es una demostración educativa de entrenamiento y evaluación. No es un dispositivo médico, no está validado externamente y no debe usarse para diagnóstico, triaje, recomendación terapéutica ni ninguna decisión sobre personas.

El conjunto es pequeño e histórico. Un buen resultado en una única partición no demuestra generalización a otros centros, equipos, poblaciones o periodos. Una versión clínica real exigiría, como mínimo, validación externa, análisis de sesgos y calibración, trazabilidad, gobernanza, supervisión clínica y cumplimiento regulatorio.

