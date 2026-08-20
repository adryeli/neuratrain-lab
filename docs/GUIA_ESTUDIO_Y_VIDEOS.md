# Guía de estudio y vídeos — Entrenamiento de ANNs

No necesitas otra formación completa antes de abrir el notebook. Usa este recorrido corto y vuelve al código después de cada vídeo.

## Recorrido recomendado

1. **Tu serie actual de DotCSV**  
   Termínala para mantener una sola narrativa en español. No intentes memorizar fórmulas: identifica neurona, peso, activación, predicción y error.

2. **3Blue1Brown — What is a neural network?**  
   https://www.youtube.com/watch?v=aircAruvnKk  
   Objetivo: ver una red como una cadena de transformaciones, no como una caja mágica.

3. **3Blue1Brown — Gradient descent**  
   https://www.youtube.com/watch?v=IHZwWFHWa-w  
   Objetivo: comprender por qué los pesos cambian en la dirección que reduce la pérdida.

4. **3Blue1Brown — Backpropagation, intuitively**  
   https://www.youtube.com/watch?v=Ilg3gGewQ5U  
   Objetivo: entender que backpropagation reparte responsabilidad por el error. No necesitas reproducir aún todas las derivadas.

5. **StatQuest — The Essential Main Ideas of Neural Networks**  
   https://www.youtube.com/watch?v=CqOfi41LfDw  
   Objetivo: reforzar la misma idea con otra explicación y vocabulario.

6. **Epochs, Iterations and Batch Size**  
   https://www.youtube.com/watch?v=SftOqbMrGfE  
   Objetivo: cerrar la diferencia entre ejemplo, batch, actualización y época justo antes del laboratorio.

## Vídeos por tema (los mismos que usa la app)

Estos son los vídeos precargados en cada página del Recorrido guiado de la app. Puedes verlos ahí directamente, o aquí como referencia rápida.

- **Tema 1 · El Perceptrón**
  - https://www.youtube.com/watch?v=MRIv2IwFTPg&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=1
  - https://www.youtube.com/watch?v=uwbHOpp9xkc&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=2
- **Tema 2 · Pérdida y Backpropagation**
  - https://www.youtube.com/watch?v=eNIqz_noix8&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=4
- **Tema 3 · Optimizadores**
  - https://www.youtube.com/watch?v=MD2fYip6QsQ&t=337s
- **Tema 4 · Entrenamiento y sobreajuste**
  - https://www.youtube.com/watch?v=7-6X3DTt3R8&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=7
  - https://www.youtube.com/watch?v=ZmLKqZYlYUI&list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&index=8

## Material adicional: notebook "3 Maneras de Programar una Red Neuronal"

Notebook de bonus (DOTCSV, adaptado por Elizabeth Sena) disponible en `notebooks/material_adicional/`, descargable también desde Inicio en la app. Vídeos explicativos:

- https://www.youtube.com/watch?v=W8AeOXa_FqU&list=PL-Ogd76BhmcCO4VeOlIH93BMT5A_kKAXp&index=5
- https://www.youtube.com/watch?v=qTNUbPkR2ao

## Lectura oficial después del laboratorio

- Overfitting y underfitting en TensorFlow: https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
- `EarlyStopping`: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
- `Dropout`: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
- Tutorial oficial de Python en español: https://docs.python.org/es/3/tutorial/
- Recurso señalado por Eli: https://github.com/andalons/python-fundamentals

## Método de 3 preguntas

Después de cada vídeo o bloque de notebook, responde sin mirar:

1. ¿Qué entra en este paso?
2. ¿Qué cambia durante el entrenamiento?
3. ¿Qué evidencia me diría que está aprendiendo mal?

Si no puedes contestar una, vuelve solo al minuto o celda que la explica. No reinicies el curso entero.

## Experimentos para fijar conceptos

Hazlos de uno en uno y anota una hipótesis antes de ejecutar:

1. Cambia `batch_size` de 32 a 8. ¿Cuántas actualizaciones hay por época?
2. Usa 300 épocas, dropout 0 y sin early stopping. ¿Cuándo se separan `loss` y `val_loss`?
3. Activa dropout 0.30 y early stopping. ¿Mejora validación o solo termina antes?
4. Compara 32→16 con 128→64. ¿La red grande generaliza mejor?
5. Baja el umbral de 0.50 a 0.35. ¿Qué ocurre con sensibilidad y falsos positivos?

