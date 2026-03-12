# Resumen: Machine Learning for Absolute Beginners
**Autor:** Oliver Theobald · Tercera Edición · 2021
**Páginas leídas:** 1-80 (capítulos 1-10 de 19)

---

## De qué trata (2-3 líneas)

Introducción práctica y sin matemáticas pesadas al campo del Machine Learning. Cubre los conceptos fundamentales, el flujo de trabajo completo de un proyecto de ML, y los algoritmos más usados en la industria, con ejemplos concretos y quizzes por capítulo. El lenguaje es Python con Scikit-learn.

---

## Conceptos clave

- **Machine Learning**: subfield de la IA donde las máquinas aprenden de datos en lugar de seguir reglas programadas explícitamente. El humano define el algoritmo y los hiperparámetros; la máquina encuentra los patrones.
- **Supervised Learning**: datos con input (X) y output (y) conocidos. El modelo aprende la relación X→y. Ejemplos: regresión lineal, logística, k-NN, SVM, redes neuronales.
- **Unsupervised Learning**: solo inputs (X), sin output conocido. El modelo descubre patrones ocultos. Ejemplo clave: k-means clustering.
- **Semi-supervised Learning**: mezcla de datos etiquetados y no etiquetados. Se entrena con los etiquetados y se usan para etiquetar el resto.
- **Reinforcement Learning**: el modelo aprende por trial & error, acumulando recompensas/penalizaciones. Ejemplo: Q-learning, videojuegos, autos autónomos.
- **Feature (variable)**: cada columna de un dataset. X son las independientes (inputs), y es la dependiente (output).
- **Hyperparameter**: configuración del algoritmo que controla cómo aprende (no los parámetros del modelo en sí).
- **Training / Test data**: split del dataset para entrenar (70-80%) y evaluar el modelo (20-30%). Regla: nunca testear con los mismos datos de entrenamiento.
- **Cross-validation (k-fold)**: dividir los datos en k buckets, usar uno como test en cada ronda. Maximiza el uso de datos disponibles y reduce el error de predicción.
- **Data Scrubbing**: limpieza del dataset antes de entrenar. Es la tarea que más tiempo consume en ML.
- **One-hot Encoding**: convertir variables categóricas en columnas binarias (0/1) para que los algoritmos puedan procesarlas.
- **Normalization**: reescalar features a un rango fijo como [0,1]. Útil cuando las magnitudes de las variables distorsionan el modelo.
- **Standardization**: convertir a distribución normal con media 0 y desviación estándar 1. Recomendado para SVM, PCA y k-NN.
- **Linear Regression**: predice valores continuos (y = bx + a). El modelo ajusta una línea (hiperplano) que minimiza el error residual.
- **Multiple Linear Regression**: múltiples variables independientes → y = a + b₁x₁ + b₂x₂ + ... Cuidado con la multicolinealidad entre variables.
- **Logistic Regression**: clasifica en categorías discretas. Usa la función sigmoide (S-curve) para convertir inputs en probabilidades entre 0 y 1. Ideal para clasificación binaria.
- **k-Nearest Neighbors (k-NN)**: clasifica un punto nuevo según la mayoría de sus k vecinos más cercanos. Simple pero costoso computacionalmente con datasets grandes.
- **k-Means Clustering**: algoritmo no supervisado que divide datos en k grupos según similitud. Útil para segmentación de clientes, detección de fraudes.
- **GPU**: unidad de procesamiento especializada en operaciones matriciales paralelas. Clave para entrenar modelos grandes. Andrew Ng demostró en 2009 que clústeres de GPUs podían hacer en un día lo que una CPU tardaba semanas.
- **MAE / RMSE**: métricas de error para modelos de regresión. Si el MAE en test es mucho mayor que en train → overfitting.

---

## Capítulos / Secciones principales

### Cap. 2 — What is Machine Learning?
ML = aprender de datos, no de comandos explícitos. Diferencia con Data Mining: ML analiza input Y output para mejorar predicciones futuras; Data Mining solo analiza inputs para descubrir patrones sin self-learning.

### Cap. 3 — Machine Learning Categories
Cuatro categorías: Supervised (etiquetado), Unsupervised (sin etiquetas), Semi-supervised (mixto), Reinforcement (trial & error con recompensas).

### Cap. 4 — The ML Toolbox
Tres compartimentos: (1) Datos — estructurados (tablas CSV) o no estructurados (imágenes, audio); (2) Infraestructura — Python + Jupyter + NumPy + Pandas + Scikit-learn para principiantes; TensorFlow/PyTorch + GPU para avanzados; (3) Algoritmos — shallow (Scikit-learn) y deep learning (TensorFlow/PyTorch).

### Cap. 5 — Data Scrubbing
Pipeline de limpieza:
1. Feature selection: eliminar columnas irrelevantes o redundantes
2. Row compression: fusionar filas similares
3. One-hot encoding: variables texto → binario
4. Binning: convertir valores continuos en categorías cuando la magnitud exacta no importa
5. Normalization / Standardization: uniformizar escala de variables
6. Missing data: rellenar con mode (categórico), median (continuo), o eliminar filas

### Cap. 6 — Setting Up Your Data
Split 70/30 u 80/20, siempre randomizando antes. Para datasets pequeños: usar k-fold cross-validation. Mínimo de datos: 10x el número de features. Algoritmo según tamaño: clustering y reducción de dimensión (<10k), regresión/clasificación (<100k), redes neuronales (>100k).

### Cap. 7 — Linear Regression
"Hello World" del ML supervisado. Fórmula: y = bx + a. La línea (hiperplano) minimiza el error residual. Multiple regression: y = a + b₁x₁ + b₂x₂... Problema a evitar: multicolinealidad (dos variables independientes muy correladas entre sí se cancelan mutuamente).

### Cap. 8 — Logistic Regression
Para predecir clases discretas (no valores continuos). Usa función sigmoide: y = 1/(1+e⁻ˣ). Punto de corte en 0.5. Mejor en clasificación binaria. Para multiclase: usar decision trees o SVM.

### Cap. 9 — k-Nearest Neighbors
Clasifica por proximidad a los k vecinos más cercanos. k impar para evitar empates. Requiere standardización previa. Lento en datasets grandes (O(n) por predicción). Evitar variables binarias no críticas.

### Cap. 10 — k-Means Clustering
Algoritmo no supervisado que asigna cada punto al centroide más cercano, recalcula centroides iterativamente hasta convergencia. Se define k manualmente. Útil para segmentación sin etiquetas.

---

## Lo más importante que te llevás

1. **El flujo siempre es el mismo:** datos crudos → scrubbing → split train/test → elegir algoritmo → entrenar → evaluar → ajustar hiperparámetros. Memorizá este loop.

2. **El scrubbing es el 80% del trabajo real.** Los algoritmos son triviales de invocar con Scikit-learn. El arte está en preparar los datos correctamente.

3. **Elegir el algoritmo según el problema:** predecir número continuo → regresión lineal; clasificar en categorías → regresión logística / k-NN / SVM; agrupar sin etiquetas → k-means.

4. **GPU no es un lujo, es necesidad.** Para cualquier modelo medianamente serio, necesitás GPU. El paper de Andrew Ng (Stanford, 2009) marcó el cambio: GPU clusters hacen en horas lo que una CPU tardaría semanas.

5. **La data relevante vale más que más data.** "When looking for the needle, the last thing you want to do is pile lots more hay on it" — Bruce Schneier. Más data no siempre mejora el modelo.

---

## Cómo se conecta con la plataforma de fine-tuning

| Concepto del libro | Aplicación directa en tu plataforma |
|---|---|
| Data Scrubbing | El pipeline de validación y sanitización de datasets JSONL que recibís del cliente antes de fine-tuning. One-hot encoding, manejo de nulls, normalización — todo esto pasa antes de que el job empiece. |
| Train/Test Split | En fine-tuning, el split del dataset del cliente entre datos de entrenamiento y datos de evaluación. Controla la calidad del modelo resultante. |
| Hyperparameters | Los parámetros que el usuario configura al lanzar el job: learning rate, epochs, LoRA rank. Son exactamente los hyperparameters de este libro, solo que en contexto LLM. |
| GPU como infraestructura avanzada | El capítulo explica por qué las GPU son necesarias para neural networks. Tu plataforma usa A100/H100 — exactamente esto. |
| MAE / métricas de error | En tu plataforma: perplexity, loss curves en MLflow. Son el equivalente al MAE del libro para evaluar si el modelo mejoró. |
| k-Means Clustering | Caso de uso concreto para tus clientes: segmentación de clientes en Onfit (gimnasios) sin necesidad de etiquetas previas. |
| Unsupervised Learning | La base conceptual detrás de los embeddings y representaciones internas de los LLMs que vas a fine-tunear. |
