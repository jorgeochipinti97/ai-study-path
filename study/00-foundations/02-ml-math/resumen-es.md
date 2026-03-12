# Resumen: Mathematics for Machine Learning
**Autor:** Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong
**Año:** 2020
**Páginas leídas:** 1-417 (libro completo)

---

## De qué trata

Este libro construye los cimientos matemáticos que subyacen a los algoritmos de ML modernos. No enseña ML directamente: enseña el lenguaje en el que ML está escrito. El lector termina con dominio operacional de álgebra lineal, geometría analítica, descomposiciones matriciales, cálculo vectorial, probabilidad y optimización, y ve cómo cada pieza aparece concretamente en cuatro algoritmos: regresión lineal, PCA, GMMs y SVMs.

Está dirigido a lectores con algo de exposición a matemáticas universitarias (cálculo, álgebra) pero que nunca conectaron esas matemáticas con ML. El stack es puramente matemático —no hay código de implementación; el valor es conceptual y formal. El ángulo diferencial del libro es que cada capítulo matemático cierra mostrando explícitamente dónde aparece ese concepto en los capítulos de algoritmos, lo que evita que la matemática se sienta desconectada.

---

## Conceptos clave

- **Vector**: elemento de un espacio vectorial; en ML representa un punto de datos, un conjunto de parámetros, o un embedding. Operaciones clave: suma, producto escalar, norma. Trampa: confundir columna (n×1) con fila (1×n) cambia el resultado de multiplicaciones.

- **Matriz**: transformación lineal entre espacios vectoriales. Las columnas de A son las imágenes de los vectores base. Trampa: la multiplicación matricial NO es conmutativa (AB ≠ BA en general).

- **Rango (rank)**: número de columnas linealmente independientes = dimensión del espacio imagen. Si rank(A) < n, el sistema Ax = b puede no tener solución única.

- **Espacio nulo (null space / kernel)**: conjunto {x : Ax = 0}. Si dim(ker(A)) > 0, hay infinitas soluciones; ML: los parámetros no son identificables.

- **Pseudoinversa de Moore-Penrose**: X† = (X^TX)^{-1}X^T para el caso columnas independientes. Resuelve mínimos cuadrados: θ_ML = X†y. Trampa: si X^TX es singular (columnas dependientes), usar SVD.

- **Proyección ortogonal**: π_U(x) = B(B^TB)^{-1}B^Tx proyecta x sobre el subespacio generado por las columnas de B. Fundamento geométrico de mínimos cuadrados y PCA.

- **Norma**: medida de magnitud de un vector. ℓ₂: ||x||₂ = √(Σxᵢ²) — usada en regularización L2 y geometría euclidiana. ℓ₁: ||x||₁ = Σ|xᵢ| — induce sparsity (regularización Lasso). Trampa: en alta dimensión, las normas tienen propiedades contraintuitivas (concentración de medida).

- **Producto interno**: generalización del producto punto; define ángulos y ortogonalidad. ⟨x,y⟩ = x^Ty para el producto interno estándar. En espacios de funciones: ⟨f,g⟩ = ∫f(x)g(x)dx.

- **Eigenvalor / eigenvector**: Ax = λx; el vector no cambia de dirección bajo la transformación A, solo escala por λ. Centrales en PCA, SVD, análisis de estabilidad.

- **Teorema espectral**: toda matriz simétrica real S se diagonaliza con vectores propios ortogonales: S = TΛT^{-1} = TΛT^T. Propiedad clave: eigenvalores de covarianza son varianza en cada dirección principal.

- **Descomposición de Cholesky**: A = LL^T para matrices simétricas definidas positivas. Más eficiente que LU para sistemas lineales con matrices de covarianza. Requerida para muestrear de una Gaussiana multivariada.

- **SVD (Descomposición en Valores Singulares)**: A = UΣV^T para cualquier matriz rectangular m×n. U: vectores singulares izquierdos (base del espacio imagen), V: vectores singulares derechos (base del espacio de entrada), Σ: valores singulares (magnitudes). Unifica: pseudoinversa, compresión de rango bajo, PCA.

- **Teorema de Eckart-Young**: la mejor aproximación de rango M de A (en norma de Frobenius) es Â_M = U_MΣ_MV_M^T —los M primeros términos del SVD. Fundamento matemático de PCA y compresión.

- **Derivada parcial**: ∂f/∂xᵢ mide cómo cambia f variando solo xᵢ. En ML: gradiente del loss respecto a cada parámetro.

- **Jacobiano**: matriz J ∈ ℝ^{m×n} de todas las derivadas parciales de f: ℝⁿ → ℝᵐ. Jᵢⱼ = ∂fᵢ/∂xⱼ. Regla de la cadena con Jacobianos: d(g∘f)/dx = (dg/df)(df/dx).

- **Backpropagation**: aplicación de la regla de la cadena en modo reverso (reverse-mode autodiff). Calcula gradientes de la loss respecto a todos los parámetros en un solo backward pass. Eficiente porque reutiliza activaciones intermedias.

- **Hessiano**: H ∈ ℝ^{n×n} de segundas derivadas. Si H ≻ 0 (definida positiva), el punto crítico es mínimo local. Usado en métodos de Newton y análisis de curvatura.

- **Probabilidad condicional**: P(A|B) = P(A∩B)/P(B). Fundamental en inferencia Bayesiana.

- **Teorema de Bayes**: P(θ|X) = P(X|θ)P(θ)/P(X). Posterior ∝ likelihood × prior. Permite actualizar creencias con datos.

- **Distribución Gaussiana**: N(μ, Σ) con pdf p(x) = (2π)^{-D/2}|Σ|^{-1/2} exp(-½(x-μ)^TΣ^{-1}(x-μ)). Cierre bajo marginales y condicionales. El condicional p(x₁|x₂) sigue siendo Gaussiano con media y varianza que dependen de x₂.

- **Familia exponencial**: distribuciones de la forma p(x|η) = h(x)exp(⟨η,T(x)⟩ - A(η)). Incluye Gaussiana, Bernoulli, Poisson. Propiedad clave: el MLE siempre tiene solución de forma cerrada; el gradiente de A(η) da la media de T(x).

- **Función convexa**: f es convexa si f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y). Propiedad clave: todo mínimo local es global. La mayoría de las funciones de pérdida en ML son convexas (o se aproximan como tal).

- **Descenso por gradiente**: θ_{t+1} = θ_t - η∇_θL. Convergencia garantizada para funciones convexas con learning rate adecuado. SGD usa un minibatch para estimar el gradiente —introduce ruido que puede escapar mínimos locales.

- **Multiplicadores de Lagrange / KKT**: para optimización con restricciones. L(x,λ) = f(x) - λg(x). Condiciones KKT: ∇f = λ∇g, g(x*) = 0. Para desigualdades: complementariedad λᵢgᵢ(x*) = 0.

- **Dualidad fuerte**: bajo condiciones de regularidad (Slater), el dual de Lagrange da el mismo valor que el primal. Permite resolver SVMs en el espacio dual donde aparece el producto interno (→ kernel trick).

- **Riesgo empírico (ERM)**: R_emp = (1/N)Σℓ(y_n, f(x_n)). Minimizar ERM sin regularización lleva a overfitting. Con regularización L2: R_reg = R_emp + λ||θ||².

- **MLE (Maximum Likelihood Estimation)**: θ_ML = argmax P(X|θ) = argmin -log P(X|θ). Para regresión con noise Gaussiano, MLE es equivalente a mínimos cuadrados.

- **MAP (Maximum A Posteriori)**: θ_MAP = argmax P(θ|X) = argmin [-log P(X|θ) - log P(θ)]. Con prior Gaussiano N(0, σ_p²I), MAP = mínimos cuadrados regularizado con λ = σ²/σ_p².

- **Inferencia Bayesiana completa**: en lugar de un punto θ*, se mantiene la distribución posterior p(θ|X). La predicción integra sobre todos los parámetros: p(y*|x*) = ∫p(y*|x*,θ)p(θ|X)dθ. Da incertidumbre calibrada.

- **Modelos gráficos (redes Bayesianas)**: representan la estructura de independencia condicional en una distribución conjunta. Notación de placa para datos repetidos. d-separación determina independencia condicional leyendo el grafo.

- **Criterios de selección de modelos**: AIC = 2k - 2ln(L̂), BIC = k·ln(N) - 2ln(L̂). BIC penaliza más la complejidad para N grande. Alternativa Bayesiana: evidencia marginal (Bayes factor).

- **Regresión lineal Bayesiana**: posterior sobre parámetros: p(θ|X,y) = N(m_N, S_N). Distribución predictiva: p(y*|x*) = N(φ^T(x*)m_N, φ^T(x*)S_Nφ(x*)+σ²). El segundo término es varianza epistémica (incertidumbre en parámetros) + varianza aleatoria (ruido del proceso).

- **PCA (Análisis de Componentes Principales)**: proyección lineal que maximiza varianza retenida. Los M componentes principales son los M eigenvectores de la matriz de covarianza con mayores eigenvalores. La varianza capturada es V_M = Σ_{m=1}^M λ_m; la perdida es J_M = Σ_{j=M+1}^D λ_j.

- **GMM (Gaussian Mixture Model)**: p(x|θ) = Σ_k π_k N(x|μ_k,Σ_k). No tiene MLE analítico (log de suma). El algoritmo EM alterna: E-step (responsabilidades r_nk = asignación suave) ↔ M-step (actualizar μ_k, Σ_k, π_k). Converge monótonamente pero puede quedar en máximo local.

- **SVM (Support Vector Machine)**: clasificador de margen máximo. Margen = 2/||w||; maximizar margen = minimizar ||w||². Dual: el clasificador depende solo de productos internos ⟨x_i,x_j⟩ → kernel trick. Solo los vectores de soporte (α_n > 0) determinan el hiperplano.

- **Kernel**: función k(x_i,x_j) = ⟨φ(x_i),φ(x_j)⟩ que computa el producto interno en un espacio de features implícito sin calcular φ explícitamente. El kernel RBF k(x,x') = exp(-||x-x'||²/(2ℓ²)) implica un espacio de dimensión infinita.

---

## Capítulos principales

### Cap. 1 — Introduction and Motivation

El capítulo establece el marco del libro: ML tiene tres pilares —datos, modelo y aprendizaje. Los datos son representaciones del mundo; el modelo es un conjunto de funciones parametrizadas sobre los datos; el aprendizaje es el proceso de encontrar los parámetros óptimos.

La motivación central es que los algoritmos de ML son fórmulas matemáticas. Para entender por qué el gradiente desciente converge, por qué regularizar previene overfitting, o por qué las SVMs tienen buena generalización, hay que entender álgebra lineal, probabilidad y optimización. El libro construye esas herramientas y luego las aplica directamente en los cuatro algoritmos de los capítulos 9-12.

**Conecta con el siguiente:** Empieza con el bloque más fundamental: álgebra lineal (Cap. 2-4), que es el lenguaje de las transformaciones sobre datos.

---

### Cap. 2 — Linear Algebra

El álgebra lineal formaliza operaciones sobre vectores y matrices, que son la representación natural de datos (un ejemplo = vector) y modelos (parámetros = vector, transformaciones = matrices).

**Grupos y espacios vectoriales:** Un grupo (G, ·) requiere clausura, asociatividad, elemento identidad e inverso. Un espacio vectorial V sobre ℝ requiere adicionalmente suma de vectores y multiplicación escalar con sus propiedades. Esta estructura garantiza que las operaciones de ML (promedios, gradientes, interpolaciones) están bien definidas.

**Independencia lineal y base:** Un conjunto de vectores {b₁,...,bₙ} es una base si genera todo el espacio (span) y ninguno puede expresarse como combinación de los otros (independencia lineal). El rango de una matriz A es la dimensión del span de sus columnas. Si rank(A) = n (rango completo de columnas), A^TA es invertible —condición necesaria para que exista solución única en mínimos cuadrados.

**Espacio nulo:** ker(A) = {x : Ax = 0}. Por el teorema rango-nulidad: dim(ker(A)) + rank(A) = n (número de columnas). En ML: si dim(ker(A)) > 0, hay infinitos parámetros que dan la misma predicción —el modelo no es identificable sin regularización.

**Proceso de Gram-Schmidt:** Dado un conjunto de vectores linealmente independientes, construye una base ortonormal iterativamente: b̃ₖ = bₖ - Σⱼ₌₁^{k-1} ⟨bₖ, b̃ⱼ⟩b̃ⱼ, bₖ* = b̃ₖ/||b̃ₖ||. Fundamental para estabilidad numérica en factorizaciones.

**Pseudoinversa de Moore-Penrose:** Para A^TA invertible: A† = (A^TA)^{-1}A^T. Resuelve el problema de mínimos cuadrados x = A†b directamente. Para matrices rectangulares o de rango deficiente, se obtiene vía SVD: A† = VΣ†U^T donde Σ† invierte los valores singulares no nulos.

**Proyección ortogonal:** La proyección de x sobre el subespacio col(A) es π(x) = A(A^TA)^{-1}A^Tx. Geometricamente: π(x) es el punto en col(A) más cercano a x. La diferencia x - π(x) es ortogonal a col(A). Esta es exactamente la interpretación geométrica de mínimos cuadrados: los residuos son ortogonales al espacio de columnas del diseño.

**Conecta con el siguiente:** Las matrices tienen propiedades geométricas (ángulos, ortogonalidad, distancias) que se estudian en Cap. 3.

---

### Cap. 3 — Analytic Geometry

Este capítulo equipa al álgebra lineal con estructura geométrica: normas para medir magnitudes, productos internos para medir ángulos, y proyecciones para entender cómo los vectores se relacionan en el espacio.

**Normas:** Una norma ||·|| satisface: no-negatividad, homogeneidad y desigualdad triangular.

| Norma | Fórmula | Uso en ML |
|---|---|---|
| ℓ₁ | Σ|xᵢ| | Regularización Lasso (sparsity) |
| ℓ₂ | √(Σxᵢ²) | Regularización Ridge, distancia euclidiana |
| ℓ∞ | max|xᵢ| | Análisis de robustez |

**Producto interno generalizado:** ⟨x,y⟩_A = x^TAy para A simétrica definida positiva. El producto interno estándar es el caso A = I. Permite definir geometrías no euclidianas —útil cuando las dimensiones tienen escalas o correlaciones distintas.

**Ángulo y ortogonalidad:** cos(θ) = ⟨x,y⟩/(||x||||y||). x ⊥ y iff ⟨x,y⟩ = 0. Ortogonalidad es la generalización de "independiente" al contexto geométrico.

**Proyección ortogonal (derivación):** Para proyectar x sobre el subespacio generado por las columnas de B ∈ ℝ^{n×k}: π_U(x) = B(B^TB)^{-1}B^Tx. La matriz de proyección P = B(B^TB)^{-1}B^T satisface P² = P (idempotente) y P^T = P (simétrica). Para el caso k=1 (proyección sobre un vector b): π_b(x) = (b^Tx/b^Tb)b.

**Conecta con el siguiente:** Las descomposiciones matriciales (Cap. 4) aprovechan la ortogonalidad para factorizar matrices de maneras numéricamente estables.

---

### Cap. 4 — Matrix Decompositions

Las descomposiciones factorizan matrices para revelar estructura, facilitar cómputo y permitir aproximaciones. Son el corazón computacional del álgebra lineal numérica.

**Determinante:** det(A) = producto de los eigenvalores. |det(A)| = factor de escala del volumen bajo la transformación A. Si det(A) = 0, la matriz es singular (no invertible). Propiedades: det(AB) = det(A)det(B), det(A^T) = det(A).

**Traza:** tr(A) = Σᵢ Aᵢᵢ = suma de eigenvalores. Invariante bajo permutación cíclica: tr(ABC) = tr(CAB) = tr(BCA). Aparece en cálculo de gradientes respecto a matrices.

**Eigenvalores y eigenvectores:** Ax = λx. El polinomio característico det(A - λI) = 0 da los eigenvalores. Para cada λ, el eigenespacio es ker(A - λI). Los eigenvalores de una matriz real simétrica son siempre reales; los eigenvectores son ortogonales.

**Teorema espectral:** Toda matriz simétrica S ∈ ℝ^{n×n} se diagonaliza: S = TΛT^T donde T es ortogonal (T^T = T^{-1}) y Λ = diag(λ₁,...,λₙ). Equivalentemente: S = Σᵢ λᵢ uᵢuᵢ^T (suma de proyecciones de rango 1 escaladas por eigenvalores). La matrix de covarianza Σ siempre es simétrica definida positiva → eigenvalores positivos → representa varianza en cada dirección principal.

**Cholesky:** Para A simétrica definida positiva: A = LL^T donde L es triangular inferior. Complejidad O(n³/3) —la mitad que LU. Prácticamente: para resolver Ax = b → Ly = b (forward substitution) → L^Tx = y (backward substitution). Usada extensamente en GP y regresión Bayesiana.

**SVD:** A = UΣV^T para A ∈ ℝ^{m×n}.
- U ∈ ℝ^{m×m}: matriz ortogonal, columnas = vectores singulares izquierdos
- Σ ∈ ℝ^{m×n}: diagonal con σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0 (valores singulares)
- V ∈ ℝ^{n×n}: matriz ortogonal, columnas = vectores singulares derechos

Relación con eigenvalores: σᵢ² = eigenvalores de A^TA (= eigenvalores de AA^T). La pseudoinversa: A† = VΣ†U^T.

**Teorema de Eckart-Young:** La mejor aproximación de rango M de A en norma de Frobenius es:
Â_M = Σᵢ₌₁^M σᵢ uᵢvᵢ^T = U_M Σ_M V_M^T

El error de aproximación es ||A - Â_M||_F² = Σᵢ₌₁^{n-M} σᵢ₊ₘ². Esto justifica PCA: retener las M componentes con mayores valores singulares captura la mayor varianza posible.

**Conecta con el siguiente:** El cálculo vectorial (Cap. 5) permite optimizar funciones sobre matrices y vectores —indispensable para el aprendizaje.

---

### Cap. 5 — Vector Calculus

El cálculo vectorial generaliza la derivación a funciones de múltiples variables. Es el mecanismo por el que los algoritmos de ML aprenden: ajustan parámetros en la dirección que más reduce la pérdida.

**Derivadas parciales y gradiente:** Para f: ℝⁿ → ℝ, el gradiente ∇f ∈ ℝⁿ tiene componentes (∇f)ᵢ = ∂f/∂xᵢ. El gradiente apunta en la dirección de máximo crecimiento de f. Descenso por gradiente: moverse en la dirección opuesta.

**Jacobiano:** Para f: ℝⁿ → ℝᵐ, J ∈ ℝ^{m×n} con Jᵢⱼ = ∂fᵢ/∂xⱼ. El Jacobiano es la generalización de la derivada: linealización local de f alrededor de un punto.

**Regla de la cadena:** Sea f: ℝⁿ → ℝᵐ y g: ℝᵐ → ℝᵏ, entonces d(g∘f)/dx = (dg/df)(df/dx) — producto matricial de Jacobianos.

**Backpropagation (reverse-mode autodiff):** Para una red con capas x → f₁ → f₂ → ... → fₖ → L (loss escalar), la regla de la cadena da:
∂L/∂xₗ = (∂fₗ₊₁/∂xₗ)^T · ... · (∂fₖ/∂xₖ₋₁)^T · ∂L/∂xₖ

El backward pass computa esto de derecha a izquierda (desde la loss hacia la entrada), reutilizando activaciones del forward pass. Costo: un backward pass es comparable en tiempo a un forward pass, independientemente del número de parámetros. Esto hace el entrenamiento de redes profundas computacionalmente factible.

**Taylor series:** f(x+δ) ≈ f(x) + ⟨∇f(x), δ⟩ + ½δ^T H δ + O(||δ||³). El término lineal es el gradiente; el cuadrático involucra el Hessiano. Los métodos de Newton usan la aproximación cuadrática para convergencia más rápida que GD.

**Hessiano:** H ∈ ℝ^{n×n} con Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ. Si H ≻ 0 (definida positiva), el punto crítico es mínimo local. Si H es indefinida, es punto de silla. En ML con millones de parámetros, calcular H exacto es prohibitivo —se usan aproximaciones (L-BFGS).

**Conecta con el siguiente:** Con el cálculo vectorial podemos derivar funciones de densidad de probabilidad (Cap. 6) y maximizar likelihoods (→ MLE en Cap. 8-9).

---

### Cap. 6 — Probability and Distributions

La probabilidad formaliza la incertidumbre. En ML, los datos son ruidosos, los modelos tienen incertidumbre, y la inferencia requiere actualizar creencias con evidencia.

**Axiomas y probabilidad condicional:** P(A∩B) = P(A|B)P(B). La regla de Bayes:
P(θ|X) = P(X|θ)P(θ) / P(X)
- P(θ|X): posterior (creencia actualizada sobre parámetros)
- P(X|θ): likelihood (qué tan probable son los datos dado θ)
- P(θ): prior (creencia inicial)
- P(X) = ∫P(X|θ)P(θ)dθ: evidencia marginal (normalizador)

**Gaussiana multivariada:** N(x; μ, Σ) = (2π)^{-D/2} |Σ|^{-1/2} exp(-½(x-μ)^T Σ^{-1} (x-μ))

Propiedades cruciales:
- **Marginal:** Si (x₁, x₂) ~ N(μ, Σ), entonces x₁ ~ N(μ₁, Σ₁₁) (marginalizar = tomar sub-bloque de la media y covarianza).
- **Condicional:** p(x₁|x₂) = N(μ₁|₂, Σ₁|₂) donde:
  - μ₁|₂ = μ₁ + Σ₁₂Σ₂₂^{-1}(x₂ - μ₂)
  - Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂^{-1}Σ₂₁ (complemento de Schur)

Esto es la base de los Procesos Gaussianos: el condicional de una Gaussiana es Gaussiana.

**Familia exponencial:** p(x|η) = h(x) · exp(⟨η, φ(x)⟩ - A(η)) donde:
- η: parámetros naturales
- φ(x): estadísticas suficientes
- A(η): log-partition function (normalizador)

Propiedad clave: ∇_η A(η) = E[φ(x)]. En MLE para familia exponencial: ∇_η A(η̂) = (1/N)Σφ(xₙ) — igualar la media empírica de las estadísticas suficientes a su esperanza teórica. Esto siempre tiene solución única (A es convexa).

**Conecta con el siguiente:** La optimización (Cap. 7) provee los algoritmos para encontrar los parámetros que maximizan la likelihood o minimizan la pérdida.

---

### Cap. 7 — Continuous Optimization

La optimización es el mecanismo de aprendizaje: dado un criterio (función de pérdida), encontrar los parámetros que lo minimizan.

**Convexidad:** f es convexa si para todo x, y y λ ∈ [0,1]: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y). Equivalentemente: H ≽ 0 para funciones diferenciables dos veces. Propiedad crítica: todo mínimo local es global. La mayoría de las pérdidas de ML son convexas (MSE, log-loss, hinge loss).

**Descenso por gradiente:**
θ_{t+1} = θ_t - η ∇_θ L(θ_t)

Para f convexa con gradiente L-Lipschitz y learning rate η ≤ 1/L: converge a ritmo O(1/T). Para f fuertemente convexa: converge exponencialmente O(ρ^T) con ρ < 1.

**SGD y minibatches:** El gradiente del ERM es:
∇L = (1/N) Σₙ ∇ℓ(yₙ, f(xₙ; θ))
SGD estima esto con un minibatch B << N: ∇̃L = (1/|B|) Σᵢ∈B ∇ℓ. El gradiente ruidoso actúa como regularizador implícito y permite escapar mínimos locales pobres. Para deep learning, minibatch de 32-256 es típico.

**Multiplicadores de Lagrange:** Para minimizar f(x) sujeto a g(x) = 0:
L(x, λ) = f(x) - λg(x)
Condición necesaria de optimalidad: ∇_x L = 0 y ∇_λ L = g(x) = 0.

**Condiciones KKT (Karush-Kuhn-Tucker):** Para restricciones de desigualdad g(x) ≤ 0 y h(x) = 0:
1. Estacionariedad: ∇f = Σᵢ λᵢ ∇gᵢ + Σⱼ νⱼ ∇hⱼ
2. Factibilidad primal: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. Factibilidad dual: λᵢ ≥ 0
4. Complementariedad: λᵢ gᵢ(x*) = 0

La condición de complementariedad es clave para SVMs: λᵢ = 0 para puntos que no son vectores de soporte.

**Dualidad fuerte (condición de Slater):** Si existe un x factible estricto (gᵢ(x) < 0), el gap dual es cero: p* = d*. El dual de Lagrange da el mismo valor que el primal. En SVMs: el problema dual tiene variables αₙ (una por punto de datos) en lugar de w (dimensión de features) → conveniente cuando n << D.

**Transformada de Legendre-Fenchel:** f*(s) = sup_x(⟨s,x⟩ - f(x)). La conjugada convexa aparece en la derivación del dual de SVMs y en modelos de energía.

**Conecta con el siguiente:** Cap. 8 usa ERM + regularización como el framework general de aprendizaje, y MLE/MAP como instancias de este framework.

---

### Cap. 8 — When Models Meet Data

Este capítulo unifica los conceptos anteriores en el framework del aprendizaje: cómo se define el problema de aprender de datos, cómo se relacionan MLE, MAP y la visión Bayesiana, y cómo se seleccionan modelos.

**Riesgo empírico (ERM):** El riesgo verdadero R[f] = E[ℓ(y, f(x))] no es observable. Se minimiza el riesgo empírico:
R_emp = (1/N) Σₙ ℓ(yₙ, f(xₙ; θ))

Sin restricciones, ERM memoriza los datos de entrenamiento (overfitting). Regularización: R_reg = R_emp + λΩ(θ) donde Ω(θ) = ||θ||² (Ridge) induce parámetros pequeños o Ω(θ) = ||θ||₁ (Lasso) induce sparsity.

**Validación cruzada K-fold:** Divide los datos en K folds. Para cada fold k: entrena en los K-1 restantes, evalúa en el fold k. El error de CV es la media de los K errores de holdout. Estimado no sesgado del error de generalización. Para K = N: leave-one-out CV (costoso pero sin sesgo).

**MLE:** θ_ML = argmax_θ log P(X|θ) = argmin_θ -log P(X|θ). La log-likelihood negativa es la función de pérdida. Para ruido Gaussiano: -log P ∝ MSE → MLE = mínimos cuadrados. Para clasificación binaria: -log P = binary cross-entropy.

**MAP:** θ_MAP = argmax_θ [log P(X|θ) + log P(θ)] = argmin_θ [-log P(X|θ) - log P(θ)]. El log del prior actúa como regularizador:
- Prior Gaussiano N(0, σ_p²I): log P(θ) = -||θ||²/(2σ_p²) → regularización L2 con λ = σ²/σ_p².
- Prior Laplaciano: log P(θ) ∝ -||θ||₁ → regularización L1 (Lasso).

**Inferencia Bayesiana completa:** En lugar de un punto θ*, se computa la distribución posterior p(θ|X) ∝ p(X|θ)p(θ). La predicción marginaliza sobre todos los parámetros:
p(y*|x*, X) = ∫ p(y*|x*, θ) p(θ|X) dθ

Esto evita el overfitting estructuralmente: los parámetros que no explican los datos bien quedan con baja probabilidad posterior.

**Modelos gráficos (Bayesian networks):** Representan p(x₁,...,x_n) = Π p(xᵢ|pa(xᵢ)) donde pa(xᵢ) son los padres de xᵢ en el DAG. Cada nodo es condicionalmente independiente de sus no-descendientes dado sus padres. La notación de placa indica variables repetidas (ej: N observaciones). d-separación: xᵢ ⊥ xⱼ | S si toda trayectoria entre xᵢ y xⱼ está bloqueada por S.

**Selección de modelos:**
- AIC = 2k - 2ln(L̂): penaliza parámetros linealmente. Favorece modelos predictivos.
- BIC = k·ln(N) - 2ln(L̂): penaliza más para N grande. Consistente: selecciona el modelo verdadero asintóticamente.
- Bayes factor: B = p(X|M₁)/p(X|M₂). Ratio de evidencias marginales. Penaliza automáticamente la complejidad (prior predictive más disperso → menor likelihood marginal).

**Conecta con el siguiente:** Estos principios se aplican concretamente a regresión lineal en Cap. 9.

---

### Cap. 9 — Linear Regression

La regresión lineal es el caso más simple donde todos los principios de los capítulos anteriores se hacen concretos y calculables.

**Setup:** y = φ^T(x)θ + ε con ε ~ N(0, σ²). El diseño matricial: y = Φθ + ε donde Φ ∈ ℝ^{N×K} es la matriz de features.

**MLE (solución analítica):** Maximizar la log-likelihood para ruido Gaussiano es equivalente a minimizar MSE:
L(θ) = ||y - Φθ||² / (2σ²)
∂L/∂θ = 0 → Φ^TΦ θ = Φ^Ty → θ_ML = (Φ^TΦ)^{-1}Φ^Ty

Esta es la ecuación normal. Si Φ^TΦ es singular, usar pseudoinversa o regularizar.

**Feature maps φ(x):** La regresión es "lineal en los parámetros" pero puede ser no lineal en los datos:
- Polinomial: φ(x) = [1, x, x², ..., x^M]^T
- RBF: φ(x) = [exp(-||x-c₁||²/2), ..., exp(-||x-cK||²/2)]^T

**Ejemplo numérico del libro (regresión polinomial):**

| Grado M | R² en train | Comportamiento |
|---|---|---|
| 0 | Bajo | Underfitting — constante |
| 1 | Medio | Lineal |
| 3-4 | Alto | Buen ajuste |
| 6-9 | Perfecto | Overfitting — oscilaciones |

El grado óptimo (M=3 o 4 según el dataset del libro) se selecciona por CV o evidencia marginal.

**MAP (regularización):** Con prior θ ~ N(0, b²I):
θ_MAP = (Φ^TΦ + λI)^{-1}Φ^Ty donde λ = σ²/b²

La regularización L2 hace Φ^TΦ + λI siempre invertible. Para λ → ∞: θ_MAP → 0 (underfitting). Para λ → 0: θ_MAP → θ_ML (overfitting posible).

**Regresión lineal Bayesiana:** Mantiene la distribución completa sobre θ.

Prior: p(θ) = N(m₀, S₀)

Posterior (tras N datos):
- S_N = (S₀^{-1} + σ^{-2} Φ^TΦ)^{-1}
- m_N = S_N(σ^{-2}Φ^Ty + S₀^{-1}m₀)

Distribución predictiva para x*:
p(y*|x*) = N(φ^T(x*)m_N, φ^T(x*)S_Nφ(x*) + σ²)

El primer término de la varianza es incertidumbre epistémica (sobre parámetros, decrece con más datos). El segundo es varianza aleatoria (ruido del proceso, irreducible). Esta calibración de incertidumbre es crucial para aplicaciones de producción.

**Evidencia marginal para selección de M:** p(y|X, M) = ∫p(y|X,θ,M)p(θ|M)dθ. El grado óptimo maximiza esta cantidad —penaliza automáticamente la complejidad sin necesidad de validation set separado.

**GLMs → deep networks:** Al reemplazar φ(x; ψ) por una red neuronal con parámetros ψ (en lugar de features fijas), la regresión lineal Bayesiana sobre φ(x; ψ) se convierte en una red neuronal. El entrenamiento aprende tanto los features como los pesos lineales. La última capa es siempre regresión lineal sobre features aprendidas.

**Conecta con el siguiente:** PCA (Cap. 10) usa la misma estructura algebraica pero sin variable de respuesta — aprendizaje no supervisado de representaciones.

---

### Cap. 10 — Dimensionality Reduction with PCA

PCA encuentra la proyección lineal de dimensión reducida que preserva la máxima varianza de los datos. Es el algoritmo de reducción de dimensionalidad clásico y el puente entre SVD y aprendizaje de representaciones.

**Formulación de máxima varianza:** Proyectar datos sobre la dirección b₁ (||b₁|| = 1):
z₁ = b₁^T x_n, varianza = b₁^T S b₁ donde S = (1/N)Σ(xₙ-μ)(xₙ-μ)^T

Maximizar b₁^T S b₁ sujeto a ||b₁||² = 1 → Lagrangiano → Sb₁ = λ₁b₁. El primer componente principal es el eigenvector de S con mayor eigenvalor.

Para M componentes: los M vectores propios de S con los M eigenvalores más grandes. La varianza capturada:
V_M = Σᵢ₌₁^M λᵢ / Σⱼ₌₁^D λⱼ (fracción de varianza explicada)

La varianza perdida:
J_M = Σⱼ₌ₘ₊₁^D λⱼ

**Formulación de error de reconstrucción mínimo:** Proyectar sobre subespacio M-dimensional y reconstruir. El error de reconstrucción ||x - x̃||² se minimiza con los mismos M eigenvectores. Ambas formulaciones (máxima varianza y mínimo error) dan la misma solución — PCA es un autoencoder lineal óptimo.

**Conexión con SVD:** Para datos centrados X ∈ ℝ^{N×D}: X = UΣV^T.
S = (1/N)X^TX = (1/N)VΣU^TUΣ V^T = (1/N)VΣ²V^T

Los eigenvalores de S son σ_d²/N (cuadrados de valores singulares, escalados). Los eigenvectores de S son las columnas de V (vectores singulares derechos).

La mejor aproximación de rango M de X (Eckart-Young): X̄_M = U_MΣ_MV_M^T. Esto proyecta y reconstruye los datos.

**El truco de alta dimensión:** Si N << D (pocos datos, muchas features), la matriz de covarianza S ∈ ℝ^{D×D} es enorme. En cambio, calcular los eigenvalores de C = (1/N)XX^T ∈ ℝ^{N×N}. Los eigenvalores no nulos son los mismos que los de S; los eigenvectores de S se recuperan de los de C: bᵢ = (1/√(Nλᵢ))X^Tcᵢ.

**Pasos prácticos para PCA:**
1. Centrar datos: X ← X - μ
2. Estandarizar (si las features tienen distintas escalas): X ← X / std
3. Calcular matriz de covarianza S = (1/N)X^TX (o usar SVD directamente)
4. Eigendescomponer: S = VΛV^T
5. Proyectar: Z = X V_M (coordenadas en el espacio reducido)

**PPCA (Probabilistic PCA):** Modelo generativo: x = Bz + μ + ε, z ~ N(0, I), ε ~ N(0, σ²I).
- B ∈ ℝ^{D×M}: factor loading matrix
- Posterior: p(z|x) = N(m, C) con m = (B^TB + σ²I)^{-1}B^T(x-μ), C = σ²(B^TB + σ²I)^{-1}
- Para σ² → 0: PPCA converge a PCA clásico

**Ejemplo del libro (MNIST, dígito "8"):**
- Imágenes de 28×28 = 784 dimensiones
- Con 1 PC: borrosamente reconocible
- Con 10 PCs: reconocible como "8"
- Con 100 PCs: casi indistinguible del original
- Con 500 PCs: reconstrucción prácticamente perfecta (varianza capturada > 99%)

**Trampa:** PCA asume que las direcciones de mayor varianza son las más informativas. Si las clases difieren en las direcciones de menor varianza, PCA puede descartar esa información. Para clasificación, LDA (Linear Discriminant Analysis) maximiza separación entre clases en lugar de varianza total.

**Conecta con el siguiente:** GMMs (Cap. 11) extienden la idea de modelar datos con Gaussianas al caso de múltiples clusters — aprendizaje no supervisado de distribuciones.

---

### Cap. 11 — Density Estimation with Gaussian Mixture Models

Los GMMs modelan distribuciones de datos arbitrariamente complejas como mezcla de Gaussianas. El algoritmo EM para ajustarlos ilustra el principio general de optimización con variables latentes.

**Modelo:** p(x|θ) = Σ_k π_k N(x|μ_k, Σ_k)
- π_k ≥ 0, Σπ_k = 1: pesos de mezcla
- μ_k, Σ_k: media y covarianza del componente k
- K componentes en total

**Por qué no hay MLE analítico:** La log-likelihood es:
L = Σₙ log Σ_k π_k N(xₙ|μ_k, Σ_k)

El logaritmo de una suma no tiene forma cerrada. Optimización directa con gradiente es posible pero lenta.

**Responsabilidades (E-step):** r_{nk} = probabilidad de que xₙ provenga del componente k:
r_{nk} = π_k N(xₙ|μ_k, Σ_k) / Σⱼ π_j N(xₙ|μⱼ, Σⱼ)

Son asignaciones suaves: cada punto pertenece a todos los componentes con distintas probabilidades.

**M-step (actualizar parámetros):** Sea N_k = Σₙ r_{nk} (número efectivo de puntos en componente k):
- μ_k^{new} = (1/N_k) Σₙ r_{nk} xₙ (media ponderada)
- Σ_k^{new} = (1/N_k) Σₙ r_{nk}(xₙ - μ_k)(xₙ - μ_k)^T (covarianza ponderada)
- π_k^{new} = N_k / N

**Algoritmo EM completo:**
1. Inicializar θ⁰ = {μ_k, Σ_k, π_k} (ej: con K-means)
2. E-step: calcular r_{nk} con θ actuales
3. M-step: actualizar θ con r_{nk} actuales
4. Calcular log-likelihood; si converge, parar; si no, ir a 2

Propiedad fundamental: cada iteración de EM no disminuye la log-likelihood. Convergencia garantizada a un máximo local (no necesariamente global).

**Ejemplo numérico del libro (7 puntos, 3 componentes):**

Datos: {-2.75, -2.71, -0.50, 0.00, 3.59, 3.64, 3.70}

GMM final convergido: p(x) = 0.29 N(x|−2.75, 0.06) + 0.28 N(x|−0.50, 0.25) + 0.43 N(x|3.64, 1.63)

Interpretación: el componente 3 captura los tres puntos de la derecha (más dispersos, de ahí mayor varianza), mientras que los dos de la izquierda quedan en componentes separados.

**Perspectiva de variable latente:** Definir variable latente zₙ ∈ {0,1}^K (one-hot: zₙ_k = 1 si xₙ proviene del componente k). Entonces:
- p(zₙ_k = 1) = π_k
- p(xₙ|zₙ_k = 1) = N(xₙ|μ_k, Σ_k)
- p(zₙ_k = 1|xₙ) = r_{nk} (posterior = responsabilidad)

El EM es equivalente a maximizar el ELBO (Evidence Lower BOund) = E_q[log p(X,Z|θ)] - KL[q(Z)||p(Z|X,θ)].

**GMM vs K-means:**
| Aspecto | K-means | GMM |
|---|---|---|
| Asignación | Hard (un cluster) | Soft (distribución) |
| Forma clusters | Solo esferas | Elipses arbitrarias |
| Criterio | Varianza intra-cluster | Log-likelihood |
| Incertidumbre | No modela | Sí (π_k, covarianzas) |

**Conecta con el siguiente:** Las SVMs (Cap. 12) abordan el problema de clasificación con un criterio geométrico diferente: maximizar el margen de separación.

---

### Cap. 12 — Classification with Support Vector Machines

Las SVMs son clasificadores binarios que encuentran el hiperplano de separación con máximo margen. El máximo margen provee garantías de generalización, y el kernel trick permite manejar datos no linealmente separables en el espacio original.

**Hiperplano separador:** {x : ⟨w, x⟩ + b = 0}. Las clases: yₙ = +1 si ⟨w, xₙ⟩ + b > 0, yₙ = -1 si ⟨w, xₙ⟩ + b < 0.

**Margen:** El margen es la distancia del hiperplano a los puntos más cercanos de cada clase. Para yₙ ∈ {±1} con ||w|| = 1: margen = 2/||w||. El hiperplano de máximo margen minimiza ||w||.

**SVM de margen duro (hard margin):** Para datos linealmente separables:
min_{w,b} ½||w||² sujeto a yₙ(⟨w, xₙ⟩ + b) ≥ 1 para todo n

Los puntos que satisfacen yₙ(⟨w, xₙ⟩ + b) = 1 son los vectores de soporte.

**SVM de margen suave (soft margin):** Para datos no separables, variables slack ξₙ ≥ 0 permiten violaciones:
min_{w,b,ξ} ½||w||² + C Σₙ ξₙ sujeto a yₙ(⟨w, xₙ⟩ + b) ≥ 1 - ξₙ, ξₙ ≥ 0

C controla el trade-off: C grande = penaliza violaciones fuertemente (margen más pequeño, menos errores de train); C pequeño = permite más violaciones (margen más grande, mejor generalización).

**Hinge loss:** La pérdida efectiva de la SVM es:
ℓ(t) = max{0, 1 - yₙ(⟨w, xₙ⟩ + b)} = max{0, 1 - t}

Para t ≥ 1 (punto correctamente clasificado con margen suficiente): pérdida = 0. Para t < 1: pérdida lineal.

**Formulación dual:** Usando KKT y el Lagrangiano, la solución primal tiene la forma:
w* = Σₙ αₙ yₙ xₙ (teorema del representante)

El problema dual:
min_{α} ½ Σᵢ,ⱼ yᵢ yⱼ αᵢ αⱼ ⟨xᵢ, xⱼ⟩ - Σᵢ αᵢ
sujeto a Σᵢ yᵢ αᵢ = 0, 0 ≤ αᵢ ≤ C

Solo los puntos con αₙ > 0 son vectores de soporte y contribuyen a w*. Por complementariedad KKT: αₙ = 0 para puntos con margen > 1 (bien separados).

El bias: b* = yₙ - ⟨w*, xₙ⟩ calculado en cualquier vector de soporte.

**Kernel trick:** El dual depende solo de ⟨xᵢ, xⱼ⟩ = productos internos. Reemplazamos:
k(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩

sin calcular φ explícitamente. La predicción: sign(Σₙ αₙ yₙ k(x, xₙ) + b*).

**Kernels comunes:**

| Kernel | Fórmula | Espacio implicito |
|---|---|---|
| Lineal | ⟨x, x'⟩ | ℝ^D |
| Polinomial | (⟨x, x'⟩ + c)^p | ℝ^{C(D+p,p)} |
| RBF (Gaussiano) | exp(-||x-x'||²/(2ℓ²)) | ∞-dimensional |
| Rational Quadratic | (1 + ||x-x'||²/(2αℓ²))^{-α} | ∞-dimensional |

**Matriz de Gram:** K ∈ ℝ^{N×N} con Kᵢⱼ = k(xᵢ, xⱼ). Un kernel es válido (Mercer) iff K es simétrica definida positiva para cualquier conjunto de datos.

**Solución numérica:** El dual de la SVM es un problema de programación cuadrática (QP) convexo. Algoritmos especializados como SMO (Sequential Minimal Optimization) lo resuelven eficientemente en O(N²) aproximadamente.

**Vista de convex hull:** Las SVMs encuentran el par de puntos más cercanos en los hulls convexos de las dos clases (uno de cada clase). El hiperplano de máximo margen es perpendicular al segmento que conecta esos dos puntos y lo bisecta.

---

## Lo más importante

1. **La pseudoinversa resuelve el caso general de mínimos cuadrados.** θ = (X^TX)^{-1}X^Ty asume X^TX invertible. Si no lo es (columnas correlacionadas, más features que datos), usar regularización λI o SVD para estabilidad numérica.

2. **SVD = la descomposición universal.** Pseudoinversa, PCA, compresión de imágenes, y la conexión datos-covarianza todos derivan de A = UΣV^T. Para cualquier operación sobre una matriz, pensar primero si SVD la simplifica.

3. **MAP = MLE + prior gaussiano = mínimos cuadrados regularizados.** La elección del prior determina la forma del regularizador. Esto unifica tres perspectivas aparentemente distintas. Elegir λ es equivalente a elegir la relación σ²/σ_p².

4. **La distribución predictiva Bayesiana da incertidumbre calibrada.** Var[y*] = φ^T(x*)S_Nφ(x*) + σ² — el primer término decrece con más datos (incertidumbre epistémica), el segundo no (aleatoriedad irreducible). En producción, esta distinción permite saber cuándo el modelo está extrapolando.

5. **El algoritmo EM maximiza una cota inferior (ELBO) de la log-likelihood.** Converge monótonamente pero puede quedar en máximo local. Inicializar con K-means y correr múltiples inicializaciones aleatorias mitiga esto en GMMs.

6. **El kernel trick convierte productos internos en operaciones en espacios implícitos.** Para usar SVM (o cualquier algoritmo kernelizable) con features no lineales, solo hay que definir k(x,x') — no hay que implementar φ. El kernel RBF implica features de dimensión infinita con costo computacional O(N²).

7. **Backpropagation es la regla de la cadena aplicada en modo reverso.** Un backward pass calcula gradientes respecto a todos los parámetros en tiempo O(forward pass). La diferenciación automática (autograd) automatiza esto — pero entender el Jacobiano subyacente es necesario para diagnosticar NaN gradients o explosión/desaparición de gradientes.

8. **Las condiciones KKT describen el óptimo de cualquier problema de optimización convexo con restricciones.** La complementariedad (λᵢgᵢ = 0) explica por qué las SVMs dependen solo de vectores de soporte, por qué LASSO produce soluciones sparsas, y por qué los multiplicadores duales son precios sombra en programación lineal.

---

## Cómo se conecta con la plataforma

| Concepto del libro | Aplicación en la plataforma |
|---|---|
| SVD y Eckart-Young | Compresión de matrices de pesos (model pruning) y análisis de rango en LoRA (A y B son factores de bajo rango) |
| Backpropagation / autodiff | Base de todo el entrenamiento de fine-tuning; entender el grafo computacional es necesario para diagnosticar problemas de gradientes en redes profundas |
| MAP / regularización | El weight decay en AdamW es regularización L2 sobre los parámetros — equivalente MAP con prior gaussiano sobre los deltas de LoRA |
| Regresión lineal Bayesiana | El head de clasificación/generación al final de un LLM es regresión lineal sobre embeddings; la incertidumbre del head puede calibrarse Bayesianamente |
| GMMs con EM | Clustering de embeddings de documentos para segmentación de corpus de fine-tuning; identificar distribuciones en el dataset de entrenamiento |
| Kernels y producto interno | Los attention scores son productos internos: score(q,k) = q^T k / √d_k — la "atención" es un kernel suave sobre las posiciones |
| Selección de modelos (evidencia marginal) | Seleccionar hiperparámetros de fine-tuning (rank de LoRA, λ de regularización) sin overfitting al validation set |
| PCA / reducción de dim | Visualización de embeddings (UMAP usa estructura similar); análisis de clustering de activaciones para interpretabilidad |
| Optimización (SGD, Adam) | Todo el fine-tuning; Adam = SGD con momentum adaptativo; gradient clipping se entiende en términos de norma del Jacobiano |
| Incertidumbre epistémica vs aleatoria | En inferencia para producción: saber si el modelo está fuera de distribución (incertidumbre epistémica alta) para decidir cuándo escalar a humano |

**Sobre LoRA y SVD:** La técnica LoRA (Low-Rank Adaptation) para fine-tuning eficiente representa el delta de pesos como ΔW = BA donde B ∈ ℝ^{d×r} y A ∈ ℝ^{r×k} con r << min(d,k). Esto es exactamente una aproximación de bajo rango — el capítulo 4 (Eckart-Young) justifica por qué las actualizaciones de fine-tuning pueden ser comprimidas: los cambios semánticamente relevantes tienden a residir en un subespacio de baja dimensión. Entender el rango y los valores singulares de ΔW permite estimar qué tanto cambió el modelo respecto al pre-entrenado.

**Sobre la atención como kernel:** El mecanismo de atención de un transformer calcula scores = QK^T / √d_k — esto es una matriz de Gram escalada. Cada fila es el producto interno del query de una posición con las keys de todas las demás. Los conceptos de kernel PSD del capítulo 12 (la Gram matrix debe ser definida positiva para que el kernel sea válido) aparecen en el análisis teórico de la atención y en técnicas como Performers que aproximan la atención con kernels de rango bajo.

**Sobre calibración de incertidumbre en producción Latam:** La regresión lineal Bayesiana del Cap. 9 separa la incertidumbre epistémica (el modelo no tiene suficientes datos) de la aleatoria (ruido irreducible). En la plataforma, esto se traduce en: cuando un cliente hace una query fuera de distribución de su corpus de fine-tuning, el modelo debería reportar alta incertidumbre en lugar de generar con confianza. Implementar esto en LLMs requiere técnicas como conformal prediction o Monte Carlo dropout, pero el fundamento conceptual es exactamente la distribución predictiva Bayesiana del Cap. 9.

---

*Páginas leídas: 1-417 (libro completo)*
