# Examen: Machine Learning for Absolute Beginners
**Dificultad progresiva · 14 preguntas**

---

## Sección 1 — Conceptos fundamentales (comprensión)

**1.** ¿Cuál de las siguientes afirmaciones describe mejor la diferencia entre Machine Learning y Data Mining?

A. Data Mining analiza tanto input como output para mejorar predicciones futuras
B. Machine Learning solo analiza inputs para encontrar patrones sin self-learning
C. Machine Learning analiza input y output para mejorar predicciones; Data Mining analiza inputs sin self-learning
D. Son sinónimos, no existe diferencia relevante

---

**2.** Verdadero o Falso: En k-Nearest Neighbors, usar un valor par de k es recomendado para evitar sesgos en la clasificación.

---

**3.** ¿Qué problema describe el siguiente escenario? Un modelo tiene un MAE de $3,000 en el training set y un MAE de $140,000 en el test set.

A. Underfitting (alto bias)
B. Overfitting (alta variance)
C. El modelo está perfectamente calibrado
D. Error en el scrubbing de datos

---

**4.** ¿Cuál es el propósito del Kernel Trick en SVM?

A. Reducir el número de features del dataset
B. Proyectar datos a una dimensión más alta para hacer separables clases no linealmente separables
C. Aumentar el valor de C para endurecer el margen
D. Aplicar normalización antes del entrenamiento

---

**5.** ¿Cuál de los siguientes NO es un tipo de ensemble modeling?

A. Bagging
B. Stacking
C. Binning
D. Boosting

---

## Sección 2 — Aplicación práctica

**6.** Tenés un dataset de clientes de e-commerce con 50,000 filas y las siguientes columnas: edad (numérica), país (texto), producto_comprado (texto), monto_gasto (numérico), churn (binario: 1/0). Querés predecir si un cliente va a hacer churn. Describí los pasos de scrubbing necesarios antes de entrenar el modelo.

---

**7.** Explicá la diferencia entre Random Forests y Gradient Boosting en términos de: (a) cómo se construyen los árboles, (b) velocidad de entrenamiento, y (c) sensibilidad a outliers. ¿Cuándo usarías cada uno?

---

**8.** En el ejercicio del libro (Melbourne Housing), el modelo inicial tenía `max_depth=30` y produjo MAE train de $27k vs MAE test de $168k. Se redujo `max_depth` a 5 y el MAE train subió a $135k. ¿Por qué subir el error de training puede ser una mejora del modelo? ¿Qué concepto explica esto?

---

**9.** Un cliente te pide construir un sistema para aprobar o rechazar préstamos bancarios. El equipo legal exige poder explicar cada decisión a los clientes rechazados. ¿Usarías una red neuronal profunda o un decision tree? Justificá tu respuesta considerando el black-box dilemma.

---

## Sección 3 — Síntesis y criterio

**10.** Describí el tradeoff Bias-Variance. ¿Por qué no se puede tener simultáneamente bias bajo y variance bajo? Incluí en tu respuesta el concepto de regularización y cómo impacta en cada término.

---

**11.** Comparar estos tres algoritmos para clasificación en un dataset de 500 filas y 8 features: Logistic Regression, k-NN, y Random Forests. ¿Cuál recomendarías y por qué? Considerá: transparencia, velocidad, overfitting, y cantidad de datos.

---

**12.** El libro describe el workflow de 6 pasos para construir un modelo en Python. Describí cada paso y explicá por qué el orden importa (¿qué pasa si se hacen los pasos fuera de orden?).

---

## Sección 4 — Aplicado a la plataforma

**13.** En una plataforma de fine-tuning de LLMs, los clientes suben datasets JSONL para entrenar modelos. ¿Cómo se mapean los conceptos de Bias-Variance y Model Optimization del libro a la experiencia de un cliente en la plataforma? Describí un escenario concreto donde el cliente vea underfitting y cómo debería corregirlo.

---

**14.** La plataforma permite al cliente configurar hiperparámetros antes de lanzar un job de fine-tuning (learning rate, epochs, LoRA rank). Diseñá una funcionalidad de "hyperparameter optimization" basada en los conceptos de Grid Search y RandomizedSearch del libro. ¿Qué trade-offs tendría cada opción en el contexto de una plataforma SaaS (costo de GPU, tiempo de espera, experiencia de usuario)?

---

## Respuestas

**1.** C — Machine Learning analiza input y output para mejorar predicciones futuras. Data Mining solo analiza inputs para descubrir patrones sin self-learning.

**2.** Falso — k impar es el valor recomendado para evitar empates en la clasificación por mayoría.

**3.** B — Overfitting (alta variance). El modelo aprendió muy bien los patrones del training set pero no generaliza a datos nuevos. El enorme gap entre train y test MAE es el indicador clásico.

**4.** B — El Kernel Trick proyecta datos a una dimensión más alta (ej: 2D → 3D) donde una frontera lineal puede separar clases que no eran linealmente separables en el espacio original.

**5.** C — Binning es una técnica de data scrubbing (convertir valores continuos en categorías), no un método de ensemble modeling.

**6.** Pasos de scrubbing:
- Feature selection: evaluar si `producto_comprado` aporta señal o tiene demasiada cardinalidad (puede requerir agrupación o eliminación)
- One-hot encoding: convertir `país` y `producto_comprado` en columnas binarias con `pd.get_dummies()`
- Missing values: verificar nulls en cada columna; rellenar con mediana para `edad` y `monto_gasto` (continuas), o eliminar filas si son pocas
- La variable objetivo es `churn` (y=1/0); las restantes son features (X)
- Standardización: aplicar antes de entrenar si se usa k-NN o SVM; no necesario para decision trees/random forests

**7.**
- (a) Random Forests: construye árboles en paralelo con muestras aleatorias y limitando features por split (bootstrap sampling). Gradient Boosting: construye árboles secuencialmente, cada uno corrigiendo los errores del anterior con pesos.
- (b) Random Forests: más rápido (paralelo). Gradient Boosting: más lento (secuencial).
- (c) Random Forests: más robusto con outliers (el voto de 100+ árboles diluye el impacto). Gradient Boosting: más sensible a outliers porque cada árbol está forzado a aprender de los errores anteriores, incluidos los causados por outliers.
- Usar Random Forests cuando: el dataset tiene muchos outliers, se necesita un modelo rápido de benchmark, o el tiempo de entrenamiento es una restricción. Usar Gradient Boosting cuando: se necesita máxima accuracy y el dataset tiene patrones consistentes.

**8.** Subir el MAE de training es una mejora porque indica que el modelo dejó de memorizar el training set y aprendió patrones más generalizables. El concepto que lo explica es el Bias-Variance tradeoff: al reducir max_depth, aumentamos el bias (el modelo es menos flexible) pero reducimos la variance (generaliza mejor). El gap entre train y test MAE se achica, lo que es el objetivo real: predecir bien datos nuevos, no predecir bien los datos que ya vio.

**9.** Decision tree. El black-box dilemma de las redes neuronales impide explicar las razones de cada decisión — no se puede reconstruir el camino de variables que llevó a rechazar un préstamo. Un decision tree, en cambio, es completamente transparente: se puede mostrar al cliente exactamente qué variables y umbrales llevaron al rechazo ("Ingresos < $X y Edad < Y → Rechazado"). El requisito legal de explicabilidad no es negociable, y los decision trees son la única opción que lo garantiza en este caso.

**10.** El tradeoff Bias-Variance dice que reducir uno de los dos términos tiende a aumentar el otro. Un modelo simple (pocas features, poca profundidad) tiene alto bias (no captura todos los patrones) pero baja variance (predice consistentemente). Un modelo complejo tiene bajo bias (captura todos los patrones del training) pero alta variance (sobreajusta, falla con datos nuevos). La regularización es el mecanismo que penaliza la complejidad del modelo, empujando hacia un punto de equilibrio: reduce variance (menos overfitting) a costa de aumentar ligeramente el bias. El hiperparámetro de regularización controla cuánta penalización se aplica.

**11.** Recomendación: Random Forests. Justificación:
- 500 filas es un dataset pequeño-mediano; tanto k-NN como Logistic Regression funcionan en este rango, pero Random Forests es más robusto al overfitting
- k-NN en 500 filas puede ser lento (O(n) por predicción) y sensible a features no relevantes
- Logistic Regression es el más interpretable pero solo funciona bien cuando la relación es aproximadamente lineal
- Random Forests maneja bien variables mixtas (numéricas y categóricas), no requiere standardización, es robusto a outliers, y con 100-150 árboles produce un buen modelo de baseline rápidamente
- Si el cliente necesita explicabilidad, entonces Decision Tree simple, aunque con riesgo de overfitting mayor

**12.** Los 6 pasos y por qué el orden importa:
1. Importar librerías: primero para que el resto del código tenga acceso a las funciones
2. Importar dataset: debe existir antes de procesarse
3. Scrubbing: debe hacerse antes del split — si se hace después, el modelo puede aprender patrones del test set (data leakage). Las transformaciones (one-hot encoding, dropna) deben calcularse solo sobre training data
4. Split train/test: debe hacerse después del scrubbing pero antes de entrenar
5. Configurar e instanciar el algoritmo: debe estar definido antes de llamar a `model.fit()`
6. Evaluar: debe ser el último paso, usando datos que el modelo nunca vio

Orden incorrecto crítico: hacer scrubbing después del split puede causar data leakage. Evaluar con los mismos datos de training invalida la evaluación completamente.

**13.** Escenario de underfitting en fine-tuning:
- El cliente sube un dataset de 500 ejemplos para fine-tunear un modelo de clasificación de soporte técnico
- Configura 1 epoch y learning_rate=0.00001 (muy conservador)
- Resultado: la loss curve no converge, el modelo en evaluación tiene perplexity alta en el val set — equivalente a un MAE de training alto con gap pequeño respecto al test
- Esto es underfitting / alto bias: el modelo no aprendió suficiente del dataset
- Corrección: aumentar epochs (de 1 a 3-5), aumentar learning_rate ligeramente, o usar un LoRA rank mayor para darle al modelo más capacidad de aprendizaje
- La plataforma debería mostrar la loss curve en tiempo real para que el cliente pueda identificar visualmente si el modelo está convergiendo

**14.** Grid Search vs RandomizedSearch en una plataforma SaaS:
- Grid Search: el cliente define rangos (ej: epochs=[3,5,10], lr=[0.0001,0.001,0.01]) y la plataforma lanza N×M×K jobs. Pros: sistemático y exhaustivo. Contras: costo de GPU potencialmente muy alto (3×3×3 = 27 jobs), tiempo de espera largo, puede ser excesivo para usuarios con presupuesto limitado
- RandomizedSearch: el cliente define rangos y un número máximo de intentos (ej: 5 jobs). La plataforma muestrea combinaciones aleatorias. Pros: costo controlable, resultados aceptables en menos tiempo, mejor UX. Contras: puede perderse la combinación óptima
- Recomendación para la plataforma: ofrecer RandomizedSearch por defecto con un límite de jobs configurable (ej: 3-5 jobs), y Grid Search como opción avanzada con estimación de costo antes de confirmar. Mostrar el mejor resultado con sus hiperparámetros para que el cliente pueda iterar manualmente desde ese punto.
