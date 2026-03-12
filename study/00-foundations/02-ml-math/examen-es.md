# Examen: Mathematics for Machine Learning
**Dificultad progresiva · 14 preguntas**

---

## Sección 1 — Conceptos fundamentales

**Pregunta 1**

¿Cuál de las siguientes afirmaciones sobre la pseudoinversa de Moore-Penrose es **correcta**?

A) θ† = (X^TX)^{-1}X^T solo existe cuando X^TX es singular.
B) θ† resuelve el problema de mínimos cuadrados ||Xθ - y||² de forma exacta sin importar el rango de X.
C) Si X^TX es invertible, entonces θ† = (X^TX)^{-1}X^T y la solución de mínimos cuadrados θ_ML = θ†y es única.
D) La pseudoinversa maximiza la norma de la solución entre todas las soluciones posibles.

---

**Pregunta 2**

Dado un modelo de regresión lineal con prior θ ~ N(0, b²I) y ruido ε ~ N(0, σ²I), la estimación MAP es:
θ_MAP = (Φ^TΦ + λI)^{-1}Φ^Ty

¿Qué representa λ en esta expresión?

A) La varianza del ruido σ²
B) La razón σ²/b² entre la varianza del ruido y la varianza del prior
C) La varianza del prior b²
D) El número de parámetros del modelo

---

**Pregunta 3**

Considerá el siguiente fragmento de Python conceptual:

```python
# Forward pass
z1 = W1 @ x + b1          # Capa 1: W1 ∈ ℝ^{m×n}, x ∈ ℝ^n
a1 = relu(z1)              # Activación
z2 = W2 @ a1 + b2          # Capa 2: W2 ∈ ℝ^{p×m}
loss = mse(z2, y)

# Backward pass
dL_dz2 = 2 * (z2 - y) / n
dL_dW2 = dL_dz2 @ ???      # ← ¿qué va acá?
```

¿Qué expresión es correcta para el gradiente de la loss respecto a W2?

A) `dL_dz2 @ W2`
B) `dL_dz2.T @ a1.T`
C) `dL_dz2 @ a1.T`
D) `a1 @ dL_dz2.T`

---

**Pregunta 4**

Verdadero o Falso (justificá tu respuesta):

"En el algoritmo EM para GMMs, la log-likelihood del modelo puede disminuir entre una iteración y la siguiente."

---

**Pregunta 5**

Un modelo de regresión polinomial se ajusta a un dataset de entrenamiento. Se observan los siguientes resultados:

| Grado M | Train MSE | Validation MSE |
|---|---|---|
| 1 | 0.82 | 0.85 |
| 4 | 0.12 | 0.15 |
| 9 | 0.001 | 2.47 |

¿Qué diagnóstico es correcto para M=9?

A) El modelo de grado 9 es el mejor porque tiene el menor error de entrenamiento.
B) El modelo de grado 9 presenta overfitting: memorizó el ruido del training set y no generaliza.
C) El modelo de grado 9 presenta underfitting porque la diferencia entre train y validation es demasiado grande.
D) El validation MSE alto de M=9 indica que el learning rate fue demasiado alto durante el entrenamiento.

---

## Sección 2 — Aplicación práctica

**Pregunta 6**

Tenés un dataset de clasificación binaria con 500 ejemplos en ℝ^{10000} (10,000 features, más features que datos). Querés entrenar una SVM con kernel RBF.

Describí:
1. Por qué la formulación dual es preferible sobre la primal en este caso.
2. Cuáles son los vectores de soporte y qué pasa con los ejemplos que no son vectores de soporte.
3. Cómo elegirías los hiperparámetros C y ℓ (bandwidth del kernel RBF), y qué trade-off controla cada uno.

---

**Pregunta 7**

Tenés un dataset de 10,000 imágenes de 1024×1024 píxeles en escala de grises. Querés reducir dimensionalidad con PCA para visualización (2D) y para comprimir la representación (100 componentes).

1. La matriz de covarianza S sería de tamaño 1,048,576 × 1,048,576. ¿Cómo calculás PCA sin construir esa matriz?
2. Después de centrar y estandarizar, ¿cuánto de la varianza debería capturar idealmente un buen conjunto de 100 componentes? ¿Cómo lo medís?
3. Describí el problema que puede aparecer si no centrás los datos antes de aplicar PCA.

---

**Pregunta 8**

El siguiente código tiene un error sutil:

```python
def bayesian_linear_regression_predict(Phi_star, m_N, S_N, sigma2):
    """
    Phi_star: feature vector for new point, shape (K,)
    m_N: posterior mean, shape (K,)
    S_N: posterior covariance, shape (K, K)
    sigma2: noise variance (scalar)
    """
    mean_pred = Phi_star @ m_N
    # Varianza predictiva
    var_epistemic = Phi_star @ S_N @ Phi_star  # incertidumbre epistémica
    var_pred = var_epistemic + sigma2
    return mean_pred, var_pred
```

Identificá el error, explicá por qué es un error (no solo de código sino conceptual), y escribí la versión corregida.

---

**Pregunta 9**

Estás ajustando un GMM con K=3 componentes a un dataset de 1,000 puntos en ℝ². Después de 50 iteraciones de EM, la log-likelihood se estabilizó pero el resultado no se ve bien: un componente captura el 95% de los datos (π₁ ≈ 0.95) y los otros dos son casi vacíos (π₂, π₃ ≈ 0.025).

1. ¿Qué causó este resultado?
2. ¿Cómo modificarías el procedimiento de entrenamiento para evitarlo?
3. ¿Cómo decidirías si K=3 es el número correcto de componentes?

---

## Sección 3 — Síntesis y criterio

**Pregunta 10**

Relacioná los conceptos del libro para responder: ¿Por qué la estimación MAP con prior Gaussiano es equivalente a mínimos cuadrados regularizados con penalización L2? ¿Cuándo preferirías usar inferencia Bayesiana completa en lugar de MAP?

Desarrollá la derivación matemática y explicá las implicancias prácticas de cada elección para un modelo de producción.

---

**Pregunta 11**

Un colega afirma: "Siempre conviene usar un kernel RBF en lugar del kernel lineal en una SVM, porque el RBF puede aprender cualquier función no lineal y el lineal es solo un caso especial."

Evaluá esta afirmación. ¿En qué tiene razón? ¿En qué falla? ¿Cuándo el kernel lineal sería la mejor elección?

---

**Pregunta 12**

El teorema de Eckart-Young dice que la mejor aproximación de rango M de una matriz en norma de Frobenius son los M primeros términos del SVD. Explicá:

1. Por qué esto justifica matemáticamente que PCA captura la máxima varianza con M componentes.
2. Cómo se conecta este resultado con LoRA (Low-Rank Adaptation) para fine-tuning de LLMs.
3. Una limitación fundamental de esta justificación cuando se aplica a actualizaciones de pesos de transformers.

---

## Sección 4 — Aplicado a la plataforma

**Pregunta 13 — Diseño de sistema**

Estás construyendo el módulo de fine-tuning de tu plataforma para un cliente de LATAM que tiene:
- Dataset: 3,000 pares (instrucción, respuesta) en español rioplatense
- Modelo base: LLM de 7B parámetros (Llama-3-7B)
- Hardware: 1 GPU A100 de 40GB
- Objetivo: adaptar el modelo a jerga local y estilo conversacional; no cambiar conocimiento factual

Usando los conceptos del libro, respondé:
1. ¿Qué técnica de fine-tuning recomendás (full fine-tuning vs LoRA vs QLoRA)? Justificá usando los conceptos de descomposición de bajo rango y regularización.
2. ¿Cómo entendés el hiperparámetro de rank r de LoRA desde la perspectiva de la aproximación de rango bajo de Eckart-Young?
3. ¿Qué función de pérdida usarías y cómo se conecta con MLE sobre el vocabulario?
4. ¿Cómo evitarías overfitting con solo 3,000 ejemplos? Mencioná al menos dos técnicas del libro.

---

**Pregunta 14 — Diagnóstico de producción**

Un cliente de tu plataforma reporta el siguiente problema: su modelo fine-tuneado tiene validation loss = 1.23 (buena), pero en producción los usuarios reportan respuestas que "suenan raras" y a veces el modelo genera texto irrelevante para queries de ciertos temas.

1. Formulá hipótesis sobre qué puede estar pasando, usando el lenguaje del libro (distribución de datos, generalización, incertidumbre epistémica vs aleatoria).
2. ¿Cómo diagnosticarías si el problema es distributional shift (los datos de producción son diferentes al training set)?
3. ¿Qué métricas o visualizaciones usarías para confirmar el diagnóstico?
4. ¿Qué soluciones propondrías, ordenadas por costo de implementación?

---

## Respuestas

### Respuesta 1

**Correcta: C**

A) Incorrecta: la pseudoinversa existe cuando X^TX es **invertible** (no singular). Si X^TX es singular, se usa la SVD para definir una pseudoinversa generalizada.

B) Incorrecta: θ† resuelve mínimos cuadrados de forma exacta **solo** cuando hay solución exacta (Xθ = y). En el caso general (sistema sobredeterminado), encuentra la θ con mínima norma que minimiza ||Xθ - y||².

C) Correcta: cuando X^TX es invertible (rango completo de columnas), la pseudoinversa es (X^TX)^{-1}X^T y da la solución única de mínimos cuadrados.

D) Incorrecta: es al revés. Entre todas las soluciones que minimizan el residual, la pseudoinversa da la de **mínima norma**.

---

### Respuesta 2

**Correcta: B**

La derivación: θ_MAP = argmin[-log P(X|θ) - log P(θ)]. Con ruido Gaussiano N(0, σ²): -log P(X|θ) ∝ (1/2σ²)||y - Φθ||². Con prior Gaussiano N(0, b²I): -log P(θ) = (1/2b²)||θ||². Sumando y factorizando (1/2σ²): minimizar ||y - Φθ||² + (σ²/b²)||θ||². Por lo tanto λ = σ²/b².

Intuición: λ grande = prior estrecho (parámetros deben ser pequeños) O ruido grande (datos poco informativos) → más regularización.

---

### Respuesta 3

**Correcta: C** — `dL_dz2 @ a1.T`

Por la regla de la cadena: ∂L/∂W2 = ∂L/∂z2 · ∂z2/∂W2. Como z2 = W2 @ a1, entonces ∂z2/∂W2 = a1^T en el sentido de que dL/dW2[i,j] = dL/dz2[i] · a1[j]. En forma matricial: dL/dW2 = (dL_dz2)[:, None] @ a1[None, :] = dL_dz2 @ a1.T (para dL_dz2 de forma (p,) y a1 de forma (m,), da W2_grad de forma (p,m) ✓).

A) Incorrecta: dL_dz2 @ W2 daría dimensiones (p,) @ (p,m) → error de dimensiones.
B) Incorrecta: dL_dz2.T @ a1.T = (p,) @ (m,) → no compatible como producto matricial.
D) Incorrecta: a1 @ dL_dz2.T = (m,) @ (p,) → daría forma (m,p) en lugar de (p,m).

---

### Respuesta 4

**Falso.**

El algoritmo EM garantiza que la log-likelihood no disminuye entre iteraciones. Esto se demuestra formalmente: cada iteración maximiza el ELBO (cota inferior de la log-likelihood), lo que implica L(θ^{t+1}) ≥ L(θ^t). Esta propiedad de convergencia monótona es una de las garantías fundamentales del EM. Lo que sí puede ocurrir es que la log-likelihood mejore en decrementos cada vez más pequeños y "se estabilice" en un máximo local (no global), pero nunca disminuye.

---

### Respuesta 5

**Correcta: B**

M=9 presenta overfitting clásico. El train MSE ≈ 0 indica que el polinomio de grado 9 se ajusta perfectamente a los puntos de entrenamiento (incluyendo el ruido). El validation MSE = 2.47 es 2,470 veces mayor que el train MSE, lo que indica que el modelo memorizó los datos en lugar de aprender la función subyacente.

A) Incorrecta: mínimo error de entrenamiento no es el objetivo — el objetivo es mínimo error de generalización (validation/test).

C) Incorrecta: underfitting es lo contrario — cuando el modelo es demasiado simple para capturar la señal. M=9 es demasiado complejo.

D) Incorrecta: el learning rate no es relevante aquí; la solución de regresión polinomial es analítica (ecuación normal), no iterativa.

---

### Respuesta 6

**Respuesta modelo:**

**1. Por qué usar el dual:** En la formulación primal, la variable de optimización es w ∈ ℝ^D con D = 10,000. En el dual, las variables son α ∈ ℝ^N con N = 500. Dado que N << D, el problema dual es mucho más pequeño (500 variables vs 10,000). Además, el dual depende solo de productos internos k(xᵢ, xⱼ), lo que permite usar el kernel trick para mapear a espacios de features implícitos de alta (o infinita) dimensión sin costo adicional. En el primal con kernel, habría que trabajar explícitamente con φ(x) que puede ser de dimensión infinita.

**2. Vectores de soporte:** Los vectores de soporte son los puntos con αₙ > 0. Por la condición de complementariedad KKT: λₙgₙ(x*) = 0, lo que implica αₙ > 0 solo cuando el constraint está activo, es decir, cuando yₙ(⟨w,xₙ⟩ + b) = 1 (puntos exactamente en el margen). Los puntos con αₙ = 0 (bien clasificados con margen > 1) no contribuyen a w* = Σₙ αₙ yₙ xₙ — el hiperplano no depende de ellos en absoluto. Para datos de 500 ejemplos, típicamente solo 10-50 serán vectores de soporte.

**3. Selección de C y ℓ:** Usar validación cruzada K-fold (K=5 o K=10) en una grilla log-escala: C ∈ {0.01, 0.1, 1, 10, 100} y ℓ ∈ {0.1, 1, 10, 100}. Trade-offs: C grande → penaliza violaciones fuertemente → margen más pequeño → posible overfitting; C pequeño → más tolerante a errores → margen mayor → posible underfitting. ℓ grande → el kernel RBF decae lento → puntos lejanos son similares → modelo más suave; ℓ pequeño → decae rápido → solo vecinos cercanos importan → modelo más local.

---

### Respuesta 7

**Respuesta modelo:**

**1. Truco de alta dimensión:** En lugar de calcular S = X^TX ∈ ℝ^{D×D} (imposible para D = 1,048,576), calcular C = XX^T ∈ ℝ^{N×N} con N = 10,000. Los eigenvalores no nulos de C son los mismos que los de S (escalados). Los eigenvectores de S se recuperan: si Ccₙ = λₙcₙ, entonces bₙ = X^Tcₙ / ||X^Tcₙ|| es el n-ésimo eigenvector de S. Complejidad: O(N²D) para C vs O(D²N) para S — con N=10,000 y D=1,048,576, la diferencia es de factor ~100x.

**2. Varianza capturada:** V_100 = Σᵢ₌₁^{100} λᵢ / Σⱼ₌₁^D λⱼ. Para imágenes naturales, los primeros 100 componentes típicamente capturan entre 80-95% de la varianza (los eigenvalores de imágenes tienen decaimiento rápido). Medirlo: calcular el ratio acumulado de eigenvalores y graficar el "codo" de la curva.

**3. Problema sin centrar:** Sin centrar, el primer componente principal capturará la dirección de la media de los datos (el "color promedio" de las imágenes) en lugar de la dirección de mayor varianza. Todos los componentes estarán sesgados. En el peor caso, la media consuma casi toda la varianza y los componentes restantes sean irrelevantes. Centrar: X ← X - μ (restar la imagen media pixel a pixel).

---

### Respuesta 8

**Error:** La varianza epistémica debería ser `Phi_star @ S_N @ Phi_star` pero para que sea un escalar (varianza), Phi_star debe tratarse como vector columna: `Phi_star.T @ S_N @ Phi_star` (o equivalentemente `Phi_star @ S_N @ Phi_star` si Phi_star es 1D, que NumPy interpreta correctamente). Sin embargo, el error conceptual más importante: falta el término `+ sigma2`. Pero en el código SÍ se suma sigma2. El error real es que `Phi_star @ S_N @ Phi_star` para Phi_star de shape (K,) y S_N de shape (K,K) da un escalar en NumPy (por broadcasting), lo que parece correcto.

El error sutil: la fórmula correcta es `φ^T(x*) S_N φ(x*)`. Si Phi_star tiene shape (K,) en NumPy, `Phi_star @ S_N @ Phi_star` calcula (K,) @ (K,K) @ (K,) = (K,) @ (K,) = escalar. Esto es correcto numéricamente. El error conceptual está en que si S_N no es simétrica definida positiva (por errores numéricos acumulados), la varianza puede volverse negativa. La versión robusta:

```python
def bayesian_linear_regression_predict(Phi_star, m_N, S_N, sigma2):
    """
    Phi_star: (K,)  — feature vector for new point
    m_N: (K,)       — posterior mean
    S_N: (K, K)     — posterior covariance (must be SPD)
    sigma2: scalar  — noise variance
    """
    mean_pred = Phi_star @ m_N
    # Varianza epistémica: phi^T S_N phi (escalar positivo si S_N SPD)
    var_epistemic = Phi_star @ S_N @ Phi_star
    # Varianza total: epistémica (decrece con datos) + aleatoria (constante)
    var_pred = var_epistemic + sigma2
    # Garantizar positividad ante errores numéricos
    var_pred = max(var_pred, sigma2)
    return mean_pred, var_pred
```

El error conceptual más profundo: no usar `max(var_pred, sigma2)` puede dar varianza negativa si la matriz S_N se vuelve indefinida por errores de punto flotante en muchas iteraciones de actualización.

---

### Respuesta 9

**Respuesta modelo:**

**1. Causa:** Inicialización pobre (probablemente aleatoria con todos los centroides cerca del mismo cluster). El componente 1 "ganó" la competencia temprana porque sus responsabilidades iniciales fueron más altas, lo que retroalimentó más peso en el M-step, haciendo su likelihood aún mayor, y así sucesivamente (colapso de un componente dominante). EM queda en un máximo local subóptimo.

**2. Soluciones:**
- Inicializar con K-means: los centroides de K-means dan un punto de partida razonable para μ_k.
- Correr múltiples inicializaciones (5-20) con semillas aleatorias distintas; quedarse con la que tenga mayor log-likelihood final.
- Usar K-means++ para la inicialización: distribuye los centroides iniciales maximizando la separación.
- Agregar regularización a las covarianzas: Σ_k ← Σ_k + εI para evitar singularidades y colapso.

**3. Selección de K:** Calcular el BIC para K ∈ {1, 2, 3, 4, 5}: BIC(K) = k_params·ln(N) - 2·L(K), donde k_params = K·(D + D(D+1)/2 + 1) - 1 para el caso full covariance. Elegir el K que minimiza BIC. Alternativamente, usar evidencia marginal (Bayes factor) comparando modelos, o inspección visual del elbow en la curva de log-likelihood vs K.

---

### Respuesta 10

**Respuesta modelo:**

**Derivación:**

La estimación MAP minimiza el negativo del log-posterior:
-log p(θ|X,y) = -log p(y|X,θ) - log p(θ) + cte

Para regresión lineal con ruido Gaussiano y ~ N(Φθ, σ²I):
-log p(y|X,θ) = (1/2σ²)||y - Φθ||² + cte₁

Para prior θ ~ N(0, b²I):
-log p(θ) = (1/2b²)||θ||² + cte₂

Sumando: θ_MAP = argmin (1/2σ²)||y - Φθ||² + (1/2b²)||θ||²
= argmin ||y - Φθ||² + λ||θ||² donde λ = σ²/b²

Esto es exactamente mínimos cuadrados regularizados con penalización L2 (Ridge regression).

**¿Cuándo usar inferencia Bayesiana completa vs MAP?**

MAP da un punto estimado — es computacionalmente simple pero no cuantifica incertidumbre sobre θ. Si θ_MAP está en una región de alta curvatura (muchos parámetros con likelihoods similares), el punto estimado puede dar predicciones con falsa confianza.

La inferencia Bayesiana completa mantiene la distribución posterior p(θ|X,y) = N(m_N, S_N) y predice con:
p(y*|x*) = N(φ^T m_N, φ^T S_N φ + σ²)

El término epistémico φ^T S_N φ cuantifica cuánto desacuerdo hay sobre θ en la dirección de x*. Decrece con más datos en esa región.

**En producción:** Usar inferencia Bayesiana cuando el costo de un error confiado es alto (medicina, decisiones financieras, sistemas donde el modelo debe saber cuándo no sabe). MAP es suficiente cuando el volumen de datos es grande y la incertidumbre residual es irrelevante para la aplicación.

---

### Respuesta 11

**Evaluación:**

**En qué tiene razón:** El kernel RBF tiene capacidad universal de aproximación — con suficientes datos y el bandwidth correcto, puede aproximar cualquier función continua. El kernel lineal solo puede clasificar con hiperplanos, lo que falla para datos con fronteras de decisión no lineales.

**En qué falla — casos donde el kernel lineal es mejor:**

1. **Alta dimensión con datos linealmente separables:** En espacios de dimensión muy alta (texto con TF-IDF en 100,000 features), los datos frecuentemente son linealmente separables. El kernel lineal es O(N) en la representación dual vs O(N²) para RBF. Más rápido, menos parámetros que tunear.

2. **N grande:** El kernel RBF requiere almacenar y evaluar la Gram matrix K ∈ ℝ^{N×N}. Para N > 10,000, esto es O(N²) memoria y tiempo. El kernel lineal puede implementarse directamente con w = Σ αₙyₙxₙ sin materializar K.

3. **Regularización implícita del linear kernel:** En espacios de alta dimensión, el kernel lineal tiene inductive bias apropiado (parsimonia). El RBF con ℓ muy pequeño puede sobreajustar perfectamente.

4. **Interpretabilidad:** Los pesos w del kernel lineal son interpretables: cada componente wᵢ indica la importancia de la feature i. Con RBF, la decisión depende de distancias a vectores de soporte — difícil de interpretar.

**Regla práctica:** Probar siempre primero el kernel lineal. Si los datos tienen estructura no lineal evidente y N es manejable (< 10,000), pasar a RBF con CV sobre C y ℓ.

---

### Respuesta 12

**Respuesta modelo:**

**1. Conexión Eckart-Young → PCA:**

La matriz de datos centrados X ∈ ℝ^{N×D} puede descomponerse X = UΣV^T. La covarianza muestral S = (1/N)X^TX = (1/N)VΣ²V^T. La varianza en la dirección del eigenvector bₘ (= columna m de V) es λₘ = σₘ²/N (el eigenvalor correspondiente).

Por Eckart-Young, la mejor aproximación de rango M de X en norma de Frobenius es X̄_M = U_MΣ_MV_M^T. El error ||X - X̄_M||_F² = Σⱼ₌ₘ₊₁^D σⱼ² = N · J_M donde J_M es la varianza perdida. Minimizar el error de reconstrucción de Frobenius = minimizar la varianza perdida = maximizar la varianza retenida. Ambas formulaciones convergen exactamente a la misma solución: los M eigenvectores con mayores eigenvalores.

**2. Conexión LoRA:**

LoRA parametriza ΔW = BA con B ∈ ℝ^{d×r} y A ∈ ℝ^{r×k}. Esta es una aproximación de rango r de la actualización completa. La justificación empírica de que esto funciona es que los gradientes durante el fine-tuning tienen estructura de bajo rango: la matriz de gradientes ∇W_L tiene decaimiento rápido de valores singulares, lo que significa que el "contenido semántico" del fine-tuning cabe en un subespacio de dimensión r << min(d,k). Eckart-Young dice que la mejor compresión de rango r de ΔW es exactamente la descomposición SVD truncada — que LoRA aproxima.

**3. Limitación:**

Eckart-Young minimiza el error de reconstrucción en norma de Frobenius, que pesa igualmente todos los elementos de la matriz. Pero en un transformer, no todos los pesos son igualmente importantes para la tarea. Los gradientes que importan para fine-tuning pueden estar concentrados en dimensiones específicas que no son las de mayor norma. Además, Eckart-Young es óptimo para aproximar W en sí, pero LoRA actualiza solo el delta ΔW — no hay garantía de que el rango óptimo de ΔW coincida con el de W. En la práctica, el rank r de LoRA es un hiperparámetro que se elige por CV, no derivado formalmente de Eckart-Young.

---

### Respuesta 13

**Respuesta modelo:**

**1. Técnica de fine-tuning recomendada: QLoRA**

Con 1 GPU A100 de 40GB y un modelo de 7B parámetros:
- Full fine-tuning de 7B parámetros en fp32 requiere ~112GB solo para los pesos (28 bytes por parámetro con optimizer states). Imposible en 40GB.
- LoRA: cuantiza el modelo base a fp16/bf16 (~14GB), agrega adaptadores de bajo rango. El modelo base está congelado; solo se entrenan B y A para cada capa. Con rank r=16 en todas las attention matrices de un 7B, se agregan ~20M parámetros entrenables (~0.3% del total).
- QLoRA: cuantiza el modelo base a 4-bit NF4 (~4GB) con double quantization, carga los adaptadores LoRA en fp16. Permite entrenar el modelo en ~8GB total, dejando margen para el batch y gradientes.

Desde la perspectiva de Eckart-Young: el delta ΔW de fine-tuning para adaptación de estilo (no de conocimiento factual) debería tener rango bajo — el cambio es un giro suave en el espacio de representación, no una reescritura completa de la base de conocimiento.

Con 3,000 ejemplos, el fine-tuning es pequeño y el riesgo de catastrophic forgetting es bajo si se usa LoRA (el modelo base está congelado).

**2. Rank r como hiperparámetro:**

r controla cuántos "grados de libertad" tiene el fine-tuning para modificar cada capa. r=1: solo un vector de ajuste por capa (como ajustar el bias de una proyección). r=64: puede capturar cambios más complejos pero requiere más datos para no sobreajustar. Para 3,000 ejemplos y adaptación de estilo, r ∈ {8, 16} es apropiado. Si los valores singulares del delta ΔW entrenan con caída rápida (verificar con SVD de ΔW post-training), confirma que r elegido fue suficiente.

**3. Función de pérdida:**

Cross-entropy sobre el vocabulario: L = -(1/T)Σₜ log p(token_t | tokens_<t ; θ). Esto es MLE: maximizar la probabilidad de los tokens de respuesta. Para instruction fine-tuning, solo se computa la loss sobre los tokens de respuesta (no los de instrucción), para que el modelo aprenda a completar respuestas, no a copiar instrucciones.

**4. Evitar overfitting con 3,000 ejemplos:**

- **Regularización L2 (weight decay):** En AdamW, weight_decay = λ actúa como prior Gaussiano sobre los parámetros de LoRA — equivalente a MAP. Valor típico: 0.01-0.1.
- **Early stopping:** Monitorear la validation loss en un 10% del dataset retenido. Parar cuando la validation loss empiece a subir (overfitting comenzando). Con 3,000 ejemplos, el validation set podría ser 300 pares.
- **Dropout** en los adaptadores LoRA (lora_dropout ∈ 0.05-0.1).
- **Data augmentation**: parafrasear las instrucciones o las respuestas para aumentar la diversidad efectiva del dataset.

---

### Respuesta 14

**Respuesta modelo:**

**1. Hipótesis usando el lenguaje del libro:**

- **Distributional shift:** p_train(x) ≠ p_prod(x). El validation set se tomó de la misma distribución que el training set, por lo que la validation loss es buena. Pero las queries de producción provienen de usuarios reales con distribución diferente (distintos tópicos, longitudes, formalidad de la pregunta).
- **Incertidumbre epistémica alta en zonas no cubiertas:** En regresión Bayesiana, Var_epistémica = φ^T(x*)S_Nφ(x*) es alta para x* fuera de la distribución de entrenamiento. El modelo extrapola en esas regiones — equivalente a hacer predicciones con alta incertidumbre sin reportarla.
- **Cobertura incompleta del dataset de fine-tuning:** Si ciertos temas o tipos de queries del cliente no están representados en los 3,000 pares, el modelo caerá de vuelta al comportamiento del modelo base (que puede no estar alineado con la jerga o el estilo deseado).

**2. Diagnóstico de distributional shift:**

- Extraer embeddings (ej: última capa del transformer) de 500 queries de producción y 500 del training set.
- Aplicar PCA o UMAP sobre los embeddings combinados.
- Si las nubes de puntos se solapan: no hay shift. Si están separadas: hay shift en el espacio de representación.
- Test estadístico: Maximum Mean Discrepancy (MMD) entre las dos distribuciones. Un MMD alto (> umbral calibrado en el training set) confirma shift.
- Alternativamente: entrenar un clasificador binario (train vs producción) sobre los embeddings. Si tiene accuracy > 70%, las distribuciones son distinguibles.

**3. Métricas y visualizaciones:**

- PCA/UMAP 2D de embeddings: visualización directa del shift.
- Distribución de longitudes de queries en train vs producción (histograma).
- Distribución de vocabulario: top-K tokens más frecuentes en producción vs train.
- Perplexity del modelo sobre queries de producción vs training set: perplexity alta en producción indica queries fuera de distribución.
- Heatmap de atención en las capas finales: si las queries problemáticas tienen patrones de atención irregulares (muy uniformes o muy concentrados), puede indicar inputs sin representación en el fine-tuning.

**4. Soluciones ordenadas por costo:**

1. (Bajo costo) **Curar y ampliar el dataset de fine-tuning** con ejemplos de los temas donde el modelo falla. Hacer un segundo round de fine-tuning.
2. (Medio costo) **Retrieval-Augmented Generation (RAG):** para queries sobre temas específicos, recuperar documentos relevantes del corpus del cliente e incluirlos en el contexto. No requiere re-entrenar.
3. (Medio costo) **Filtro de confianza en producción:** usar la perplexity o la entropía de la distribución de tokens para detectar queries fuera de distribución. Si la confianza es baja, devolver un mensaje de fallback o derivar a un operador humano.
4. (Alto costo) **Re-colectar datos de producción** (con consentimiento), anotarlos, y hacer un nuevo round de fine-tuning con la distribución real de producción. Este es el ciclo correcto de MLOps.
