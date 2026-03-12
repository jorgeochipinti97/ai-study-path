# Resumen: Machine Learning for Absolute Beginners
**Autor:** Oliver Theobald · Tercera Edición · 2021
**Páginas leídas:** 1-160 (capítulos 1-18, libro completo)

---

## De qué trata

Introducción práctica y sin matemáticas pesadas al Machine Learning. Cubre los conceptos fundamentales, el flujo de trabajo completo de un proyecto de ML, y los algoritmos más usados en la industria, con ejemplos concretos y quizzes por capítulo. El lenguaje es Python con Scikit-learn. El capítulo final construye un modelo real de predicción de precios de casas con el dataset Melbourne Housing, de principio a fin.

---

## Conceptos clave

- **Machine Learning**: subfield de la IA donde las máquinas aprenden de datos en lugar de seguir reglas programadas explícitamente. El humano define el algoritmo y los hiperparámetros; la máquina encuentra los patrones.
- **Supervised Learning**: datos con input (X) y output (y) conocidos. El modelo aprende la relación X→y. Ejemplos: regresión lineal, logística, k-NN, SVM, redes neuronales.
- **Unsupervised Learning**: solo inputs (X), sin output conocido. El modelo descubre patrones ocultos. Ejemplo clave: k-means clustering.
- **Semi-supervised Learning**: mezcla de datos etiquetados y no etiquetados. Se entrena con los etiquetados y se usan para etiquetar el resto.
- **Reinforcement Learning**: el modelo aprende por trial & error acumulando recompensas y penalizaciones. No hay dataset — el modelo genera sus propios datos interactuando con el entorno. Ejemplo: Q-learning, videojuegos, autos autónomos.
- **Feature (variable)**: cada columna de un dataset. X = variables independientes (inputs), y = variable dependiente (output).
- **Hyperparameter**: configuración del algoritmo definida antes del entrenamiento que controla cómo aprende. No es un parámetro interno del modelo (esos los aprende la máquina).
- **Training / Test data**: split del dataset para entrenar (70-80%) y evaluar (20-30%). Regla fundamental: nunca testear con los mismos datos usados para entrenar.
- **Cross-validation (k-fold)**: dividir los datos en k buckets, usar uno como test en cada ronda, rotar. Maximiza el uso de datos disponibles. Recomendado para datasets pequeños.
- **Data Scrubbing**: limpieza del dataset antes de entrenar. Es la tarea que más tiempo consume en cualquier proyecto de ML.
- **One-hot Encoding**: convertir variables categóricas en columnas binarias (0/1). "rojo/azul/verde" → 3 columnas binarias.
- **Normalization**: reescalar features a un rango fijo como [0,1]. Útil cuando las magnitudes de las variables distorsionan el modelo.
- **Standardization**: convertir a distribución normal con media 0 y desviación estándar 1. Recomendado para SVM, PCA y k-NN.
- **Linear Regression**: predice valores continuos (y = bx + a). El modelo ajusta una línea (hiperplano) que minimiza el error residual.
- **Multiple Linear Regression**: múltiples variables independientes → `y = a + b₁x₁ + b₂x₂ + ...` Cuidado con la multicolinealidad.
- **Logistic Regression**: clasifica en categorías discretas usando la función sigmoide. Produce probabilidades entre 0 y 1. Ideal para clasificación binaria.
- **k-Nearest Neighbors (k-NN)**: clasifica un punto nuevo según la mayoría de sus k vecinos más cercanos. Simple pero costoso computacionalmente (O(n) por predicción).
- **k-Means Clustering**: algoritmo no supervisado que divide datos en k grupos. Distancia euclidiana: `d = √((x₂-x₁)² + (y₂-y₁)²)`. Estimación inicial de k: `√(n/2)`.
- **Bias**: error por suposiciones demasiado simples del modelo (underfitting). Brecha sistemática entre predicción y realidad.
- **Variance**: dispersión de predicciones ante datos nuevos (overfitting). El modelo memorizó el training set pero no generaliza.
- **Regularización**: hiperparámetro que penaliza la complejidad del modelo para combatir el overfitting.
- **SVM (Support Vector Machine)**: algoritmo de clasificación que encuentra el hiperplano con el mayor margen entre clases. Margen = distancia desde el límite hasta el punto más cercano × 2.
- **Kernel Trick**: proyecta datos a una dimensión más alta (ej: 2D→3D) para hacer linealmente separable lo que no lo era en el espacio original.
- **Soft Margin (C bajo)**: más tolerancia a errores → mejor generalización, menos overfitting.
- **Hard Margin (C alto)**: menos errores en training → riesgo de overfitting.
- **Función de activación**: umbral que determina si un nodo "dispara". Binaria (perceptrón: ≥0 → 1), sigmoide (0-1), o tanh (-1 a 1).
- **Backpropagation**: proceso de ajuste de pesos que fluye en reversa desde el output hacia el input, minimizando el cost value iterativamente.
- **Black-box Dilemma**: las redes neuronales producen predicciones precisas pero no revelan cómo cada variable influye en el resultado.
- **Perceptrón**: unidad básica de red neuronal (Frank Rosenblatt, 1950s). Output binario (0 o 1). Limitación: cambios pequeños en pesos pueden flipear el output drásticamente.
- **MLP (Multilayer Perceptron)**: red con múltiples capas (input, hidden, output). Agrega múltiples modelos en uno.
- **Deep Learning**: redes con 5-10+ capas ocultas. Capacidad de descomponer patrones complejos (imágenes, texto, video). CNN, RNN, RNTN, Deep Belief Networks.
- **Decision Tree**: árbol jerárquico que divide datos recursivamente en subconjuntos homogéneos. Transparente e interpretable. Vulnerable al overfitting por el algoritmo greedy.
- **Entropía / Information Gain**: medida de desorden en un nodo. El objetivo es seleccionar la variable que minimiza la entropía en el siguiente split. Fórmula: `(-p₁ log p₁ - p₂ log p₂) / log2`.
- **ID3**: algoritmo greedy de J.R. Quinlan que elige la variable con menor entropía en cada split. "Greedy" = optimiza localmente sin considerar el impacto global.
- **Bagging**: crece N árboles con muestras bootstrap aleatorias del dataset y combina por votación (clasificación) o promedio (regresión). Reduce variance.
- **Random Forests**: como bagging, pero también limita las variables evaluadas en cada split → árboles menos correlados → modelo más robusto. Se entrena en paralelo. 100-150 árboles como punto de partida.
- **Gradient Boosting**: construye árboles secuencialmente — cada árbol corrige los errores del anterior. Muy preciso pero lento y sensible a outliers.
- **Ensemble Modeling**: combinar múltiples modelos produce predicciones que superan a cualquier modelo individual. Tipos: bagging, boosting, bucket of models, stacking.
- **Stacking**: modelos level-0 corren en paralelo, sus outputs alimentan a un blender level-1. Ganó el Netflix Prize (BellKor's Pragmatic Chaos, 2009).
- **Grid Search (GridSearchCV)**: búsqueda exhaustiva sobre todas las combinaciones de hiperparámetros. Sistemático pero exponencialmente lento.
- **RandomizedSearchCV**: muestrea combinaciones aleatorias por ronda. Más rápido; número de iteraciones configurable.
- **MAE (Mean Absolute Error)**: promedio de las diferencias absolutas entre predicciones y valores reales. Si el MAE test es mucho mayor que el MAE train → overfitting.

---

## Capítulos principales

### Cap. 2 — What is Machine Learning?
ML = aprender de datos, no de comandos explícitos. Diferencia con Data Mining: ML analiza input Y output para mejorar predicciones futuras; Data Mining solo analiza inputs para descubrir patrones existentes sin self-learning.

### Cap. 3 — Machine Learning Categories
Cuatro categorías:
- **Supervised**: datos etiquetados (X, y). Objetivo: predecir y a partir de X.
- **Unsupervised**: sin etiquetas. Objetivo: encontrar estructura oculta en X.
- **Semi-supervised**: mayormente sin etiquetar, algunos etiquetados. Entrena en los etiquetados, extiende a los no etiquetados.
- **Reinforcement**: agente aprende vía recompensas y penalizaciones. No hay dataset — el modelo genera sus propios datos interactuando.

### Cap. 4 — The ML Toolbox
Tres compartimentos:
1. **Datos**: estructurados (tablas CSV) o no estructurados (imágenes, audio, texto)
2. **Infraestructura**: Python + Jupyter + NumPy + Pandas + Scikit-learn para principiantes; TensorFlow/PyTorch + GPU para avanzados
3. **Algoritmos**: shallow ML (Scikit-learn) vs deep learning (TensorFlow/PyTorch)

Nota GPU: Andrew Ng (Stanford, 2009) demostró que clusters de GPU hacen en un día lo que una CPU tarda semanas. Las GPUs manejan las multiplicaciones matriciales paralelas que requiere el entrenamiento de ML.

### Cap. 5 — Data Scrubbing
La etapa más lenta. El orden de los pasos importa:

1. **Feature selection**: eliminar columnas irrelevantes primero (reduce la superficie para el resto de los pasos)
2. **Row compression**: fusionar filas similares si aplica
3. **One-hot encoding**: convertir variables texto en columnas binarias
4. **Binning**: agrupar valores continuos en categorías cuando la magnitud exacta no importa (ej: edad → "18-25", "26-35")
5. **Normalization / Standardization**: uniformizar escala de variables
6. **Missing data**: rellenar con mode (categórico), median (continuo), o eliminar filas

Regla crítica: eliminar columnas irrelevantes *antes* de hacer dropna — si no, podés perder filas enteras por un NaN en una columna que ibas a borrar igual.

### Cap. 6 — Setting Up Your Data
- Split: 70/30 u 80/20. Siempre randomizá antes de hacer el split.
- Datasets pequeños: usar k-fold cross-validation para maximizar el uso de datos.
- Regla de datos mínimos: 10× el número de features.
- Algoritmo según tamaño del dataset:
  - < 10k filas → clustering, reducción de dimensionalidad
  - < 100k filas → regresión, clasificación
  - > 100k filas → redes neuronales

### Cap. 7 — Linear Regression
"Hello World" del ML supervisado. Fórmula: `y = bx + a`. La línea (hiperplano) minimiza la suma de errores cuadráticos (distancias desde cada punto hasta la línea).

Multiple regression: `y = a + b₁x₁ + b₂x₂ + ...`

Problema a evitar: **multicolinealidad** — dos variables independientes muy correladas entre sí. Si x₁ y x₂ ambas encodifican "ingresos" en diferentes unidades, sus coeficientes se cancelan mutuamente y el modelo se vuelve poco confiable. Solución: eliminar una de ellas.

### Cap. 8 — Logistic Regression
Para predecir clases discretas, no valores continuos. Usa la **función sigmoide**:

```
y = 1 / (1 + e^(-x))
```

Output siempre entre 0 y 1. Umbral por defecto: 0.5 (arriba → clase 1, abajo → clase 0). Mejor para clasificación binaria. Para multiclase, usá decision trees o SVM.

### Cap. 9 — k-Nearest Neighbors
Clasifica un nuevo punto mirando sus k vecinos más cercanos y tomando voto mayoritario. Distancia euclidiana. Reglas:
- Usar k impar para evitar empates
- Requiere standardización previa (sensible a la escala)
- Evitar variables binarias no críticas (agregan ruido)
- Lento en datasets grandes: cada predicción requiere calcular distancia a todos los n puntos de training → O(n) por predicción

### Cap. 10 — k-Means Clustering
Algoritmo no supervisado. Pasos:
1. Elegir k (número de clusters)
2. Colocar k centroides aleatoriamente
3. Asignar cada punto al centroide más cercano (distancia euclidiana)
4. Recalcular cada centroide como el promedio de sus puntos asignados
5. Repetir pasos 3-4 hasta que los centroides no se muevan (convergencia)

Cómo elegir k:
- Estimación rápida: `k ≈ √(n/2)`
- Mejor: graficar SSE (suma de errores cuadráticos) vs k → usar el punto "codo" donde agregar más clusters deja de reducir el SSE significativamente (scree plot / elbow method)

### Cap. 11 — Bias & Variance
El tradeoff central de todo el ML:

| | Bias Alto | Bias Bajo |
|---|---|---|
| **Variance Alta** | Peor caso | Overfitting |
| **Variance Baja** | Underfitting | Ideal (difícil de lograr) |

- **Underfitting (bias alto)**: modelo demasiado simple, no captura el patrón. Error de training alto, error de test alto. Fix: más features, modelo más complejo.
- **Overfitting (variance alta)**: modelo demasiado complejo, memorizó los datos de training. Error de training bajo, error de test mucho más alto. Fix: regularización, menos complejidad, más datos.
- **Regularización**: penaliza la complejidad del modelo. Aumenta levemente el bias pero reduce significativamente la variance. El hiperparámetro de regularización controla la fuerza de la penalización.

### Cap. 12 — SVM (Support Vector Machines)
Encuentra el hiperplano que crea el máximo margen entre clases. Los "support vectors" son los puntos más cercanos al límite — ellos definen el margen.

- **Margen** = distancia desde el límite hasta el punto más cercano × 2
- **Soft margin (C bajo)**: permite algunas clasificaciones incorrectas → mejor generalización
- **Hard margin (C alto)**: fuerza la clasificación correcta en training → riesgo de overfitting
- **Kernel Trick**: cuando los datos no son linealmente separables en 2D, proyectarlos a 3D (o mayor) donde un plano lineal sí puede separarlos

Limitaciones de SVM:
- Requiere standardización (sensible a la escala de features)
- Lento en datasets con bajo ratio feature-to-row
- Excelente para datasets pequeños/medianos con alta dimensionalidad

### Cap. 13 — Artificial Neural Networks
Inspiradas en las neuronas del cerebro. Estructura: nodos (neuronas) conectados por edges (axones). Cada edge tiene un **peso**. La suma de inputs ponderados pasa por una **función de activación** para decidir si el nodo dispara.

Cómputo de un nodo:
```
sum = x1*w1 + x2*w2 + x3*w3
output = función_de_activación(sum)
```

**Ejemplo con perceptrón** (del libro):
- Input 1: x1 = 24, peso w1 = 0.5 → contribución: 12
- Input 2: x2 = 16, peso w2 = -1.0 → contribución: -16
- Suma: 12 + (-16) = -4 → función de activación (≥0 = 1, si no = 0) → output: **0** (no disparó)
- Ajustar pesos: w2 se vuelve -0.5 → contribución: -8
- Nueva suma: 12 + (-8) = 4 → output: **1** (dispara)

**Backpropagation**: después de cada forward pass, se mide el error (cost value). El algoritmo fluye en reversa ajustando cada peso proporcionalmente a su contribución al error. Esto se repite hasta que el cost converge a un mínimo.

**Tipos de función de activación**:
- **Perceptrón**: step binario (0 o 1). Debilidad: cambios pequeños en los pesos pueden flipear el output.
- **Neurona sigmoide**: output entre 0 y 1. Más estable ante ajustes pequeños de pesos.
- **Tanh**: output entre -1 y 1. Puede representar relaciones negativas.

**Black-box dilemma**: dos redes con arquitecturas diferentes pueden producir el mismo output, haciendo imposible rastrear qué variables impulsaron la predicción. Los decision trees son la alternativa transparente.

**Arquitectura de la red**:
- **Input layer**: un nodo por feature
- **Hidden layer(s)**: procesamiento intermedio. Más capas ocultas = más capacidad para encontrar patrones complejos.
- **Output layer**: un nodo por clase (clasificación) o un nodo (regresión)

**Deep Learning**: 5-10+ capas ocultas. "Deep" se refiere a la profundidad de capas, no a la cantidad de neuronas. Los sistemas de reconocimiento de objetos (autos autónomos) usan 150+ layers. Aplicaciones: reconocimiento de imágenes (CNN), habla/texto (RNN), series de tiempo (RNN), clasificación (MLP, Deep Belief Networks).

### Cap. 14 — Decision Trees
Estructura del árbol: root node → branches (splits) → leaf nodes. Terminal node = hoja sin más splits.

Construcción del árbol: en cada nodo, el algoritmo selecciona la variable que minimiza la **entropía** (desorden) en el siguiente nivel. Este es el **algoritmo ID3** (Iterative Dichotomizer 3, J.R. Quinlan). Proceso: **recursive partitioning** — se repite hasta cumplir el stopping criterion (< 3-5 items por hoja, o todos los items pertenecen a una sola clase).

**Fórmula de entropía**: `(-p₁ log p₁ - p₂ log p₂) / log2`

**Ejemplo trabajado** del libro (10 empleados, predecir promoción):
| Variable | Entropía | Resultado |
|---|---|---|
| Exceeded KPIs | 0 bits | Split perfecto — dos grupos homogéneos |
| Aged < 30 | 0.6895 bits | Un grupo homogéneo |
| Leadership Capability | 0.9508 bits | Ambos grupos mezclados |

Ganador: **Exceeded KPIs** (entropía = 0). El árbol se parte en esta variable primero y termina ahí — no hacen falta más splits.

**Overfitting en decision trees**: el algoritmo greedy optimiza localmente en cada split sin considerar el impacto global. Un primer split ligeramente peor podría producir un modelo globalmente mejor. Como un chico que se come el mejor cupcake primero sin pensar en la comida completa.

**Bagging**: crece N árboles sobre muestras bootstrap aleatorias del dataset, combina predicciones por votación (clasificación) o promedio (regresión). Reduce variance exponiendo distintos árboles a distintos datos.

**Random Forests**: extiende el bagging también limitando la cantidad de variables consideradas en cada split. Esto fuerza a los árboles a usar variables distintas, haciéndolos menos correlados entre sí. Menos correlación = errores más independientes = promedio más confiable. Se entrena en paralelo. 100-150 árboles como punto de partida.

**Gradient Boosting**: ensemble secuencial. Cada árbol nuevo se enfoca en los ejemplos que los árboles anteriores erraron. Los errores del round N reciben mayor peso en el round N+1. Muy preciso pero: (1) lento porque los árboles son secuenciales, (2) tiende a overfit con muchos outliers.

Random Forest vs Gradient Boosting:
- Outliers: gana Random Forest (el voto diluye el impacto)
- Accuracy: gana Gradient Boosting (aprendizaje más enfocado)
- Velocidad: gana Random Forest (paralelo)

### Cap. 15 — Ensemble Modeling
Idea central: agregar múltiples modelos reduce el riesgo de que cualquier modelo individual esté equivocado.

- **Clasificación**: combinar por votación (mayoría gana)
- **Regresión**: combinar por promedio numérico

Cuatro técnicas:
1. **Bagging**: paralelo, homogéneo (mismo algoritmo, distintas muestras). Reduce variance.
2. **Boosting** (Gradient Boosting, AdaBoost): secuencial, homogéneo. Reduce bias.
3. **Bucket of Models**: entrena múltiples algoritmos distintos, elige el mejor en test data. Heterogéneo.
4. **Stacking**: todos corren en paralelo (level-0), sus outputs alimentan a un meta-learner blender (level-1). Red neuronal + decision tree es un stack clásico: la red maneja datos completos; el árbol maneja missing values.

Stacking ganó el Netflix Prize (2006-2009): BellKor's Pragmatic Chaos usó stacking lineal de cientos de modelos distintos → mejora del 10.06% en accuracy de recomendaciones.

### Cap. 16 — Development Environment
Setup recomendado: **Anaconda** (incluye Jupyter, NumPy, Pandas, Scikit-learn) + **Jupyter Notebook** (editor web en `localhost:8888`).

Comandos clave de Pandas:
```python
import pandas as pd

df = pd.read_csv('~/Downloads/dataset.csv')  # cargar CSV
df.head()                                     # ver primeras 5 filas
df.head(10)                                   # ver primeras N filas
df.iloc[100]                                  # obtener fila en índice 100
df.columns                                    # listar todos los nombres de columnas
```

Nota: Python indexa desde 0. `df.iloc[100]` devuelve la fila 101.

### Cap. 17 — Building a Model in Python
Modelo completo de gradient boosting con el dataset Melbourne Housing (34,857 filas, 21 variables, prediciendo precio de casas).

**Flujo de 6 pasos:**

**Paso 1 — Importar librerías**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
```

**Paso 2 — Importar dataset**
```python
df = pd.read_csv('~/Downloads/Melbourne_housing_FULL.csv')
```

**Paso 3 — Scrubbing del dataset**
```python
# Eliminar columnas irrelevantes
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']   # nota: mal escrito en el archivo fuente
del df['Longtitude']  # nota: mal escrito en el archivo fuente
del df['Regionname']
del df['Propertycount']

# Eliminar filas con valores faltantes
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# One-hot encoding para columnas categóricas
df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])

# Asignar X e y
X = df.drop('Price', axis=1)
y = df['Price']
```

**Paso 4 — Split del dataset**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)
```

**Paso 5 — Configurar algoritmo e hiperparámetros**
```python
model = ensemble.GradientBoostingRegressor(
    n_estimators=150,      # número de árboles de decisión
    learning_rate=0.1,     # encoge la contribución de cada árbol
    max_depth=30,          # capas máximas por árbol (ATENCIÓN: causa overfitting)
    min_samples_split=4,   # mínimo de samples para crear un nuevo branch
    min_samples_leaf=6,    # mínimo de samples requerido en cada hoja
    max_features=0.6,      # fracción de features considerados por split
    loss='huber'           # función de error (robusto a outliers)
)

model.fit(X_train, y_train)
```

Referencia de hiperparámetros:
| Param | Efecto |
|---|---|
| `n_estimators` | Más árboles = más accuracy (hasta cierto punto) + más lento |
| `learning_rate` | Más bajo = cada árbol contribuye menos → mejor generalización |
| `max_depth` | Más alto = árboles más complejos → riesgo de overfitting |
| `min_samples_split` | Más alto = más difícil crear nuevos branches → árboles más simples |
| `min_samples_leaf` | Más alto = cada hoja necesita más samples → menos overfitting |
| `max_features` | Float (0.6) = 60% de features seleccionados aleatoriamente por split |
| `loss` | `huber` = combinación de `ls` y `lad`, robusto a outliers |

**Paso 6 — Evaluar resultados**
```python
mae_train = mean_absolute_error(y_train, model.predict(X_train))
mae_test  = mean_absolute_error(y_test,  model.predict(X_test))

print("MAE Training: %.2f" % mae_train)   # → $27,834.12
print("MAE Test:     %.2f" % mae_test)    # → $168,262.14
```

Análisis del resultado: el error de training ($27k) es mucho menor que el de test ($168k) → **overfitting**. El modelo memorizó los patrones del training set pero no generaliza. Causa raíz: `max_depth=30` hizo cada árbol demasiado complejo.

### Cap. 18 — Model Optimization
Partiendo del modelo con overfitting (max_depth=30), dos optimizaciones:

**Optimización 1 — Reducir max_depth de 30 a 5**
```
MAE Training: $135,283.69   (era $27k — subió)
Gap train/test se achicó → menos overfitting
```
El error de training subió, pero el modelo generaliza mejor. Correcto: training error más alto = menos overfitting.

**Optimización 2 — Aumentar n_estimators de 150 a 250**
```
MAE Training: $124,469.48
MAE Test:     $161,602.45   (era $168k — mejoró)
```

**Grid Search — búsqueda exhaustiva de hiperparámetros:**
```python
from sklearn.model_selection import GridSearchCV

model = ensemble.GradientBoostingRegressor()

hyperparameters = {
    'n_estimators':      [200, 300],
    'max_depth':         [4, 6],
    'min_samples_split': [3, 4],
    'min_samples_leaf':  [5, 6],
    'learning_rate':     [0.01, 0.02],
    'max_features':      [0.8, 0.9],
    'loss':              ['ls', 'lad', 'huber']
}

grid = GridSearchCV(model, hyperparameters, n_jobs=4)
grid.fit(X_train, y_train)

print(grid.best_params_)   # devuelve la combinación óptima

mae_train = mean_absolute_error(y_train, grid.predict(X_train))
mae_test  = mean_absolute_error(y_test,  grid.predict(X_test))
```

Limitación del grid search: 2×2×2×2×2×2×3 = 192 combinaciones × folds de cross-validation = muy lento. Estrategia: primero grid grueso (potencias de 10: 0.01, 0.1, 1, 10), identificar la mejor región, luego grid fino alrededor de esa región.

**RandomizedSearchCV**: en lugar de probar todas las combinaciones, muestrea valores aleatorios de cada rango. Más rápido y permite controlar exactamente cuántos trials correr vía `n_iter`.

Principios de optimización:
- Cambiar un hiperparámetro a la vez
- Feature selection: agregar/quitar variables de a una y medir impacto en MAE suele ser más efectivo que un grid search exhaustivo
- El gap entre MAE train y MAE test es la señal clave — minimizá el gap, no solo el error de test

---

## Lo más importante que te llevás

1. **El flujo siempre es el mismo:** datos crudos → scrubbing → split train/test → elegir algoritmo → entrenar → evaluar → ajustar hiperparámetros. Memorizá este loop.

2. **El scrubbing es el 80% del trabajo real.** Los algoritmos son triviales de invocar con Scikit-learn. El arte está en preparar los datos correctamente. Eliminá columnas irrelevantes antes de hacer dropna.

3. **Elegir el algoritmo según el problema:** predecir número continuo → regresión lineal; clasificar en categorías → regresión logística / k-NN / SVM / decision trees; agrupar sin etiquetas → k-means; patrones complejos con muchas features → ANN / deep learning; máxima accuracy en datos tabulares → gradient boosting.

4. **Bias-Variance es el tradeoff central de todo el ML.** Un gap grande entre MAE train y MAE test siempre señala overfitting. Reducir la complejidad del modelo (max_depth, regularización) o agregar más datos son las correcciones principales.

5. **La transparencia tiene un costo de accuracy.** Los decision trees son interpretables pero frágiles. Las redes neuronales y los modelos ensemble son más precisos pero son black-box. Elegir según el caso de uso.

6. **En producción, el ensemble gana casi siempre.** Random forests o gradient boosting superan a modelos individuales en la gran mayoría de los casos. El Netflix Prize lo ganó stacking de cientos de modelos distintos.

7. **La optimización de hiperparámetros es iterativa.** Cambiar uno a la vez y medir el impacto antes de cambiar otro. Grid search es exhaustivo pero exponencialmente lento; RandomizedSearch es más práctico para exploración inicial.

---

## Cómo se conecta con la plataforma de fine-tuning

| Concepto del libro | Aplicación directa en la plataforma |
|---|---|
| Data Scrubbing | El pipeline de validación y sanitización de datasets JSONL que recibís del cliente antes de fine-tuning. One-hot encoding, manejo de nulls, normalización — todo esto corre antes de que el job empiece. |
| Train/Test Split | En fine-tuning: el split del dataset del cliente entre datos de entrenamiento y datos de evaluación. Controla la calidad del modelo resultante. |
| Hyperparameters | Los parámetros que el usuario configura al lanzar el job: learning rate, epochs, LoRA rank. Son exactamente los hyperparameters de este libro, solo que en contexto LLM. |
| GPU como infraestructura | El capítulo explica por qué las GPU son necesarias para neural networks. La plataforma usa A100/H100 — exactamente esto. |
| MAE / métricas de error | En la plataforma: perplexity, loss curves en MLflow. Son el equivalente al MAE del libro para evaluar si el modelo mejoró. |
| Bias-Variance tradeoff | En fine-tuning: underfitting cuando los epochs son pocos o el learning rate es demasiado bajo; overfitting cuando el modelo memorizó el dataset de entrenamiento. La perplexity en el val set es exactamente el MAE de test del libro. |
| ANN / Deep Learning | La base conceptual de todos los LLMs. Backpropagation, funciones de activación, layers — son los mecanismos exactos que operan dentro de los transformers que vas a fine-tunear. |
| Gradient Boosting | Análogo conceptual al fine-tuning iterativo: cada paso mejora el modelo respecto al error previo. El `learning_rate` en GBR y en LoRA fine-tuning cumplen el mismo rol. |
| Ensemble Modeling | La plataforma puede ofrecer evaluación multi-modelo (comparar resultados de distintos checkpoints fine-tuneados) como servicio de valor agregado. |
| Grid Search / RandomizedSearch | Equivalente a los experimentos de hyperparameter tuning en MLflow. Los clientes deberían poder lanzar múltiples jobs con distintas configs y comparar resultados lado a lado. |
| Model Optimization (Cap. 18) | El loop que la plataforma debería exponer al usuario: ajustar un hiperparámetro, re-entrenar, comparar loss, iterar. El gap entre train loss y val loss es la señal clave a mostrar en el UI. |
| k-Means Clustering | Caso de uso concreto para clientes: segmentación de usuarios sin etiquetas previas (ej: agrupar usuarios por comportamiento para prompting personalizado). |
