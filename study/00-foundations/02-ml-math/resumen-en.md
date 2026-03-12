# Summary: Mathematics for Machine Learning
**Authors:** Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong
**Year:** 2020
**Pages read:** 1-417 (complete book)

---

## What it is about

This book builds the mathematical foundations underlying modern ML algorithms. It does not teach ML directly — it teaches the language in which ML is written. The reader finishes with operational mastery of linear algebra, analytic geometry, matrix decompositions, vector calculus, probability, and optimization, and sees how each piece appears concretely in four algorithms: linear regression, PCA, GMMs, and SVMs.

It targets readers with some exposure to university-level mathematics (calculus, algebra) who have never connected those concepts to ML. The stack is purely mathematical — there is no implementation code; the value is conceptual and formal. The book's differential angle is that each math chapter closes by explicitly showing where that concept appears in the algorithm chapters, preventing the math from feeling disconnected.

---

## Key concepts

- **Vector**: element of a vector space; in ML represents a data point, a set of parameters, or an embedding. Key operations: addition, scalar multiplication, norm. Trap: confusing column (n×1) with row (1×n) changes multiplication results.

- **Matrix**: linear transformation between vector spaces. Columns of A are images of basis vectors. Trap: matrix multiplication is NOT commutative (AB ≠ BA in general).

- **Rank**: number of linearly independent columns = dimension of the image space. If rank(A) < n, the system Ax = b may not have a unique solution.

- **Null space (kernel)**: set {x : Ax = 0}. If dim(ker(A)) > 0, there are infinite solutions; in ML: parameters are not identifiable without regularization.

- **Moore-Penrose pseudoinverse**: X† = (X^TX)^{-1}X^T for the case of independent columns. Solves least squares: θ_ML = X†y. Trap: if X^TX is singular (dependent columns), use SVD instead.

- **Orthogonal projection**: π_U(x) = B(B^TB)^{-1}B^Tx projects x onto the subspace spanned by columns of B. Geometric foundation of least squares and PCA.

- **Norm**: magnitude measure for vectors. ℓ₂: ||x||₂ = √(Σxᵢ²) — used in L2 regularization and Euclidean geometry. ℓ₁: ||x||₁ = Σ|xᵢ| — induces sparsity (Lasso regularization). Trap: in high dimensions, norms have counterintuitive properties (measure concentration).

- **Inner product**: generalization of the dot product; defines angles and orthogonality. ⟨x,y⟩ = x^Ty for the standard inner product. In function spaces: ⟨f,g⟩ = ∫f(x)g(x)dx.

- **Eigenvalue / eigenvector**: Ax = λx; the vector does not change direction under transformation A, only scales by λ. Central to PCA, SVD, and stability analysis.

- **Spectral theorem**: every real symmetric matrix S is diagonalizable with orthogonal eigenvectors: S = TΛT^{-1} = TΛT^T. Key property: eigenvalues of a covariance matrix are the variances in each principal direction.

- **Cholesky decomposition**: A = LL^T for symmetric positive definite matrices. More efficient than LU for linear systems with covariance matrices. Required for sampling from a multivariate Gaussian.

- **SVD (Singular Value Decomposition)**: A = UΣV^T for any rectangular matrix m×n. U: left singular vectors (basis of image space), V: right singular vectors (basis of input space), Σ: singular values (magnitudes). Unifies: pseudoinverse, low-rank compression, PCA.

- **Eckart-Young theorem**: the best rank-M approximation of A (in Frobenius norm) is Â_M = U_MΣ_MV_M^T — the first M terms of the SVD. Mathematical foundation of PCA and compression.

- **Partial derivative**: ∂f/∂xᵢ measures how f changes varying only xᵢ. In ML: gradient of loss with respect to each parameter.

- **Jacobian**: matrix J ∈ ℝ^{m×n} of all partial derivatives of f: ℝⁿ → ℝᵐ. Jᵢⱼ = ∂fᵢ/∂xⱼ. Chain rule with Jacobians: d(g∘f)/dx = (dg/df)(df/dx).

- **Backpropagation**: application of chain rule in reverse mode (reverse-mode autodiff). Computes gradients of loss with respect to all parameters in a single backward pass. Efficient because it reuses intermediate activations.

- **Hessian**: H ∈ ℝ^{n×n} of second derivatives. If H ≻ 0 (positive definite), the critical point is a local minimum. Used in Newton's methods and curvature analysis.

- **Conditional probability**: P(A|B) = P(A∩B)/P(B). Fundamental in Bayesian inference.

- **Bayes' theorem**: P(θ|X) = P(X|θ)P(θ)/P(X). Posterior ∝ likelihood × prior. Allows updating beliefs with data.

- **Gaussian distribution**: N(μ, Σ) with pdf p(x) = (2π)^{-D/2}|Σ|^{-1/2} exp(-½(x-μ)^TΣ^{-1}(x-μ)). Closed under marginalization and conditioning. The conditional p(x₁|x₂) remains Gaussian with mean and variance depending on x₂.

- **Exponential family**: distributions of the form p(x|η) = h(x)exp(⟨η,T(x)⟩ - A(η)). Includes Gaussian, Bernoulli, Poisson. Key property: MLE always has a closed-form solution; the gradient of A(η) gives the mean of T(x).

- **Convex function**: f is convex if f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y). Key property: every local minimum is global. Most ML loss functions are convex (or approximated as such).

- **Gradient descent**: θ_{t+1} = θ_t - η∇_θL. Convergence guaranteed for convex functions with appropriate learning rate. SGD uses a minibatch to estimate the gradient — introduces noise that can escape local minima.

- **Lagrange multipliers / KKT**: for constrained optimization. L(x,λ) = f(x) - λg(x). KKT conditions: ∇f = λ∇g, g(x*) = 0. For inequalities: complementarity λᵢgᵢ(x*) = 0.

- **Strong duality (Slater's condition)**: if a strictly feasible point exists (gᵢ(x) < 0), the duality gap is zero: p* = d*. The Lagrangian dual gives the same value as the primal. In SVMs: the dual problem has variables αₙ (one per data point) instead of w (feature dimension) → convenient when n << D.

- **Legendre-Fenchel transform**: f*(s) = sup_x(⟨s,x⟩ - f(x)). The convex conjugate appears in SVM dual derivation and energy-based models.

- **Empirical risk (ERM)**: R_emp = (1/N)Σℓ(y_n, f(x_n)). Minimizing ERM without regularization leads to overfitting. With L2 regularization: R_reg = R_emp + λ||θ||².

- **MLE (Maximum Likelihood Estimation)**: θ_ML = argmax P(X|θ) = argmin -log P(X|θ). For regression with Gaussian noise, MLE is equivalent to least squares.

- **MAP (Maximum A Posteriori)**: θ_MAP = argmax P(θ|X) = argmin [-log P(X|θ) - log P(θ)]. With Gaussian prior N(0, σ_p²I), MAP = regularized least squares with λ = σ²/σ_p².

- **Full Bayesian inference**: instead of a point θ*, maintain the posterior distribution p(θ|X). Prediction integrates over all parameters: p(y*|x*) = ∫p(y*|x*,θ)p(θ|X)dθ. Gives calibrated uncertainty.

- **Graphical models (Bayesian networks)**: represent the conditional independence structure in a joint distribution. Plate notation for repeated data. d-separation determines conditional independence by reading the graph.

- **Model selection criteria**: AIC = 2k - 2ln(L̂), BIC = k·ln(N) - 2ln(L̂). BIC penalizes complexity more for large N. Bayesian alternative: marginal likelihood (Bayes factor).

- **Bayesian linear regression**: posterior over parameters: p(θ|X,y) = N(m_N, S_N). Predictive distribution: p(y*|x*) = N(φ^T(x*)m_N, φ^T(x*)S_Nφ(x*)+σ²). The second term is epistemic variance (parameter uncertainty) + aleatoric variance (process noise).

- **PCA (Principal Component Analysis)**: linear projection maximizing retained variance. The M principal components are the M eigenvectors of the covariance matrix with the largest eigenvalues. Captured variance: V_M = Σᵢ₌₁^M λᵢ; lost variance: J_M = Σⱼ₌ₘ₊₁^D λⱼ.

- **GMM (Gaussian Mixture Model)**: p(x|θ) = Σ_k π_k N(x|μ_k,Σ_k). No analytical MLE (log of sum). The EM algorithm alternates: E-step (responsibilities r_nk = soft assignments) ↔ M-step (update μ_k, Σ_k, π_k). Monotonically converges but may find a local maximum.

- **SVM (Support Vector Machine)**: maximum-margin binary classifier. Margin = 2/||w||; maximize margin = minimize ||w||². Dual: classifier depends only on inner products ⟨x_i,x_j⟩ → kernel trick. Only support vectors (α_n > 0) determine the hyperplane.

- **Kernel**: function k(x_i,x_j) = ⟨φ(x_i),φ(x_j)⟩ computing the inner product in an implicit feature space without computing φ explicitly. The RBF kernel k(x,x') = exp(-||x-x'||²/(2ℓ²)) implies an infinite-dimensional space.

---

## Main chapters

### Ch. 1 — Introduction and Motivation

The chapter establishes the book's framework: ML has three pillars — data, model, and learning. Data are representations of the world; the model is a parametrized set of functions over data; learning is the process of finding optimal parameters.

The central motivation is that ML algorithms are mathematical formulas. To understand why gradient descent converges, why regularization prevents overfitting, or why SVMs generalize well, one must understand linear algebra, probability, and optimization. The book builds these tools and then applies them directly in the four algorithms of chapters 9-12.

**Connects to next:** Starts with the most fundamental block: linear algebra (Ch. 2-4), the language of transformations on data.

---

### Ch. 2 — Linear Algebra

Linear algebra formalizes operations on vectors and matrices, which are the natural representation of data (one example = vector) and models (parameters = vector, transformations = matrices).

**Groups and vector spaces:** A group (G, ·) requires closure, associativity, identity element, and inverse. A vector space V over ℝ additionally requires vector addition and scalar multiplication with their properties. This structure guarantees that ML operations (averages, gradients, interpolations) are well-defined.

**Linear independence and basis:** A set of vectors {b₁,...,bₙ} is a basis if it spans the whole space and no vector can be expressed as a combination of the others. The rank of a matrix A is the dimension of its column span. If rank(A) = n (full column rank), A^TA is invertible — necessary condition for a unique solution in least squares.

**Null space:** ker(A) = {x : Ax = 0}. By the rank-nullity theorem: dim(ker(A)) + rank(A) = n (number of columns). In ML: if dim(ker(A)) > 0, there are infinite parameters giving the same prediction — the model is not identifiable without regularization.

**Gram-Schmidt process:** Given linearly independent vectors, builds an orthonormal basis iteratively: b̃ₖ = bₖ - Σⱼ₌₁^{k-1} ⟨bₖ, b̃ⱼ⟩b̃ⱼ, bₖ* = b̃ₖ/||b̃ₖ||. Fundamental for numerical stability in factorizations.

**Moore-Penrose pseudoinverse:** For invertible A^TA: A† = (A^TA)^{-1}A^T. Directly solves the least squares problem x = A†b. For rectangular or rank-deficient matrices, obtained via SVD: A† = VΣ†U^T where Σ† inverts non-zero singular values.

**Orthogonal projection:** The projection of x onto col(A) is π(x) = A(A^TA)^{-1}A^Tx. Geometrically: π(x) is the closest point in col(A) to x. The difference x - π(x) is orthogonal to col(A). This is exactly the geometric interpretation of least squares: residuals are orthogonal to the column space of the design matrix.

**Connects to next:** Matrices have geometric properties (angles, orthogonality, distances) studied in Ch. 3.

---

### Ch. 3 — Analytic Geometry

This chapter equips linear algebra with geometric structure: norms to measure magnitudes, inner products to measure angles, and projections to understand how vectors relate in space.

**Norms:** A norm ||·|| satisfies: non-negativity, homogeneity, and triangle inequality.

| Norm | Formula | Use in ML |
|---|---|---|
| ℓ₁ | Σ|xᵢ| | Lasso regularization (sparsity) |
| ℓ₂ | √(Σxᵢ²) | Ridge regularization, Euclidean distance |
| ℓ∞ | max|xᵢ| | Robustness analysis |

**Generalized inner product:** ⟨x,y⟩_A = x^TAy for A symmetric positive definite. The standard inner product is the case A = I. Allows defining non-Euclidean geometries — useful when dimensions have different scales or correlations.

**Angle and orthogonality:** cos(θ) = ⟨x,y⟩/(||x||||y||). x ⊥ y iff ⟨x,y⟩ = 0. Orthogonality is the generalization of "independence" to the geometric context.

**Orthogonal projection (derivation):** To project x onto the subspace spanned by columns of B ∈ ℝ^{n×k}: π_U(x) = B(B^TB)^{-1}B^Tx. The projection matrix P = B(B^TB)^{-1}B^T satisfies P² = P (idempotent) and P^T = P (symmetric). For k=1 (projection onto a vector b): π_b(x) = (b^Tx/b^Tb)b.

**Connects to next:** Matrix decompositions (Ch. 4) leverage orthogonality to factorize matrices in numerically stable ways.

---

### Ch. 4 — Matrix Decompositions

Decompositions factorize matrices to reveal structure, facilitate computation, and enable approximations. They are the computational heart of numerical linear algebra.

**Determinant:** det(A) = product of eigenvalues. |det(A)| = volume scaling factor under transformation A. If det(A) = 0, the matrix is singular (non-invertible). Properties: det(AB) = det(A)det(B), det(A^T) = det(A).

**Trace:** tr(A) = Σᵢ Aᵢᵢ = sum of eigenvalues. Invariant under cyclic permutation: tr(ABC) = tr(CAB) = tr(BCA). Appears in gradient computations with respect to matrices.

**Eigenvalues and eigenvectors:** Ax = λx. The characteristic polynomial det(A - λI) = 0 gives the eigenvalues. For each λ, the eigenspace is ker(A - λI). Eigenvalues of a real symmetric matrix are always real; eigenvectors are orthogonal.

**Spectral theorem:** Every symmetric matrix S ∈ ℝ^{n×n} is diagonalizable: S = TΛT^T where T is orthogonal (T^T = T^{-1}) and Λ = diag(λ₁,...,λₙ). Equivalently: S = Σᵢ λᵢ uᵢuᵢ^T (sum of rank-1 projections scaled by eigenvalues). The covariance matrix Σ is always symmetric positive definite → positive eigenvalues → represents variance in each principal direction.

**Cholesky:** For A symmetric positive definite: A = LL^T where L is lower triangular. Complexity O(n³/3) — half of LU. Practically: to solve Ax = b → Ly = b (forward substitution) → L^Tx = y (backward substitution). Widely used in GP and Bayesian regression.

**SVD:** A = UΣV^T for A ∈ ℝ^{m×n}.
- U ∈ ℝ^{m×m}: orthogonal matrix, columns = left singular vectors
- Σ ∈ ℝ^{m×n}: diagonal with σ₁ ≥ σ₂ ≥ ... ≥ σ_r > 0 (singular values)
- V ∈ ℝ^{n×n}: orthogonal matrix, columns = right singular vectors

Relation to eigenvalues: σᵢ² = eigenvalues of A^TA (= eigenvalues of AA^T). Pseudoinverse: A† = VΣ†U^T.

**Eckart-Young theorem:** The best rank-M approximation of A in Frobenius norm is:
Â_M = Σᵢ₌₁^M σᵢ uᵢvᵢ^T = U_M Σ_M V_M^T

Approximation error: ||A - Â_M||_F² = Σᵢ₌₁^{n-M} σᵢ₊ₘ². This justifies PCA: retaining M components with the largest singular values captures the maximum possible variance.

**Connects to next:** Vector calculus (Ch. 5) enables optimizing functions over matrices and vectors — indispensable for learning.

---

### Ch. 5 — Vector Calculus

Vector calculus generalizes differentiation to functions of multiple variables. It is the mechanism by which ML algorithms learn: adjusting parameters in the direction that most reduces loss.

**Partial derivatives and gradient:** For f: ℝⁿ → ℝ, the gradient ∇f ∈ ℝⁿ has components (∇f)ᵢ = ∂f/∂xᵢ. The gradient points in the direction of maximum growth of f. Gradient descent: move in the opposite direction.

**Jacobian:** For f: ℝⁿ → ℝᵐ, J ∈ ℝ^{m×n} with Jᵢⱼ = ∂fᵢ/∂xⱼ. The Jacobian is the generalization of the derivative: local linearization of f around a point.

**Chain rule:** Let f: ℝⁿ → ℝᵐ and g: ℝᵐ → ℝᵏ, then d(g∘f)/dx = (dg/df)(df/dx) — matrix product of Jacobians.

**Backpropagation (reverse-mode autodiff):** For a network with layers x → f₁ → f₂ → ... → fₖ → L (scalar loss), the chain rule gives:
∂L/∂xₗ = (∂fₗ₊₁/∂xₗ)^T · ... · (∂fₖ/∂xₖ₋₁)^T · ∂L/∂xₖ

The backward pass computes this right-to-left (from loss toward input), reusing forward pass activations. Cost: a backward pass is comparable in time to a forward pass, regardless of the number of parameters. This makes training deep networks computationally feasible.

**Taylor series:** f(x+δ) ≈ f(x) + ⟨∇f(x), δ⟩ + ½δ^T H δ + O(||δ||³). The linear term is the gradient; the quadratic involves the Hessian. Newton's methods use the quadratic approximation for faster convergence than GD.

**Hessian:** H ∈ ℝ^{n×n} with Hᵢⱼ = ∂²f/∂xᵢ∂xⱼ. If H ≻ 0 (positive definite), the critical point is a local minimum. If H is indefinite, it is a saddle point. In ML with millions of parameters, computing H exactly is prohibitive — approximations are used (L-BFGS).

**Connects to next:** With vector calculus we can differentiate probability density functions (Ch. 6) and maximize likelihoods (→ MLE in Ch. 8-9).

---

### Ch. 6 — Probability and Distributions

Probability formalizes uncertainty. In ML, data is noisy, models have uncertainty, and inference requires updating beliefs with evidence.

**Axioms and conditional probability:** P(A∩B) = P(A|B)P(B). Bayes' rule:
P(θ|X) = P(X|θ)P(θ) / P(X)
- P(θ|X): posterior (updated belief about parameters)
- P(X|θ): likelihood (how probable the data is given θ)
- P(θ): prior (initial belief)
- P(X) = ∫P(X|θ)P(θ)dθ: marginal evidence (normalizer)

**Multivariate Gaussian:** N(x; μ, Σ) = (2π)^{-D/2} |Σ|^{-1/2} exp(-½(x-μ)^T Σ^{-1} (x-μ))

Crucial properties:
- **Marginal:** If (x₁, x₂) ~ N(μ, Σ), then x₁ ~ N(μ₁, Σ₁₁) (marginalize = take sub-block of mean and covariance).
- **Conditional:** p(x₁|x₂) = N(μ₁|₂, Σ₁|₂) where:
  - μ₁|₂ = μ₁ + Σ₁₂Σ₂₂^{-1}(x₂ - μ₂)
  - Σ₁|₂ = Σ₁₁ - Σ₁₂Σ₂₂^{-1}Σ₂₁ (Schur complement)

This is the basis of Gaussian Processes: the conditional of a Gaussian is Gaussian.

**Exponential family:** p(x|η) = h(x) · exp(⟨η, φ(x)⟩ - A(η)) where:
- η: natural parameters
- φ(x): sufficient statistics
- A(η): log-partition function (normalizer)

Key property: ∇_η A(η) = E[φ(x)]. For MLE in the exponential family: ∇_η A(η̂) = (1/N)Σφ(xₙ) — match the empirical mean of sufficient statistics to their theoretical expectation. This always has a unique solution (A is convex).

**Connects to next:** Optimization (Ch. 7) provides the algorithms to find parameters that maximize likelihood or minimize loss.

---

### Ch. 7 — Continuous Optimization

Optimization is the learning mechanism: given a criterion (loss function), find the parameters that minimize it.

**Convexity:** f is convex if for all x, y and λ ∈ [0,1]: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y). Equivalently: H ≽ 0 for twice-differentiable functions. Critical property: every local minimum is global. Most ML losses are convex (MSE, log-loss, hinge loss).

**Gradient descent:**
θ_{t+1} = θ_t - η ∇_θ L(θ_t)

For convex f with L-Lipschitz gradient and learning rate η ≤ 1/L: converges at rate O(1/T). For strongly convex f: converges exponentially O(ρ^T) with ρ < 1.

**SGD and minibatches:** The gradient of ERM is:
∇L = (1/N) Σₙ ∇ℓ(yₙ, f(xₙ; θ))
SGD estimates this with a minibatch B << N: ∇̃L = (1/|B|) Σᵢ∈B ∇ℓ. The noisy gradient acts as an implicit regularizer and allows escaping poor local minima. For deep learning, minibatch of 32-256 is typical.

**Lagrange multipliers:** To minimize f(x) subject to g(x) = 0:
L(x, λ) = f(x) - λg(x)
Necessary optimality condition: ∇_x L = 0 and ∇_λ L = g(x) = 0.

**KKT conditions (Karush-Kuhn-Tucker):** For inequality constraints g(x) ≤ 0 and equalities h(x) = 0:
1. Stationarity: ∇f = Σᵢ λᵢ ∇gᵢ + Σⱼ νⱼ ∇hⱼ
2. Primal feasibility: gᵢ(x*) ≤ 0, hⱼ(x*) = 0
3. Dual feasibility: λᵢ ≥ 0
4. Complementarity: λᵢ gᵢ(x*) = 0

The complementarity condition is key for SVMs: λᵢ = 0 for points that are not support vectors.

**Strong duality (Slater's condition):** If there exists a strictly feasible point (gᵢ(x) < 0), the duality gap is zero: p* = d*. The Lagrangian dual gives the same value as the primal. In SVMs: the dual problem has variables αₙ (one per data point) instead of w (feature dimension) → convenient when n << D.

**Legendre-Fenchel transform:** f*(s) = sup_x(⟨s,x⟩ - f(x)). The convex conjugate appears in the SVM dual derivation and energy-based models.

**Connects to next:** Ch. 8 uses ERM + regularization as the general learning framework, with MLE/MAP as instances of this framework.

---

### Ch. 8 — When Models Meet Data

This chapter unifies the previous concepts into the learning framework: how the problem of learning from data is defined, how MLE, MAP, and the Bayesian view relate, and how models are selected.

**Empirical risk (ERM):** The true risk R[f] = E[ℓ(y, f(x))] is not observable. The empirical risk is minimized:
R_emp = (1/N) Σₙ ℓ(yₙ, f(xₙ; θ))

Without constraints, ERM memorizes training data (overfitting). Regularization: R_reg = R_emp + λΩ(θ) where Ω(θ) = ||θ||² (Ridge) induces small parameters or Ω(θ) = ||θ||₁ (Lasso) induces sparsity.

**K-fold cross-validation:** Divides data into K folds. For each fold k: train on the remaining K-1 folds, evaluate on fold k. The CV error is the mean of the K holdout errors. Unbiased estimate of generalization error. For K = N: leave-one-out CV (expensive but unbiased).

**MLE:** θ_ML = argmax_θ log P(X|θ) = argmin_θ -log P(X|θ). The negative log-likelihood is the loss function. For Gaussian noise: -log P ∝ MSE → MLE = least squares. For binary classification: -log P = binary cross-entropy.

**MAP:** θ_MAP = argmax_θ [log P(X|θ) + log P(θ)] = argmin_θ [-log P(X|θ) - log P(θ)]. The log of the prior acts as regularizer:
- Gaussian prior N(0, σ_p²I): log P(θ) = -||θ||²/(2σ_p²) → L2 regularization with λ = σ²/σ_p².
- Laplacian prior: log P(θ) ∝ -||θ||₁ → L1 regularization (Lasso).

**Full Bayesian inference:** Instead of a point θ*, compute the posterior distribution p(θ|X) ∝ p(X|θ)p(θ). Prediction marginalizes over all parameters:
p(y*|x*, X) = ∫ p(y*|x*, θ) p(θ|X) dθ

This structurally avoids overfitting: parameters that do not explain data well retain low posterior probability.

**Graphical models (Bayesian networks):** Represent p(x₁,...,x_n) = Π p(xᵢ|pa(xᵢ)) where pa(xᵢ) are the parents of xᵢ in the DAG. Each node is conditionally independent of its non-descendants given its parents. Plate notation indicates repeated variables (e.g., N observations). d-separation: xᵢ ⊥ xⱼ | S if every path between xᵢ and xⱼ is blocked by S.

**Model selection:**
- AIC = 2k - 2ln(L̂): penalizes parameters linearly. Favors predictive models.
- BIC = k·ln(N) - 2ln(L̂): penalizes more for large N. Consistent: asymptotically selects the true model.
- Bayes factor: B = p(X|M₁)/p(X|M₂). Ratio of marginal evidences. Automatically penalizes complexity (more diffuse prior predictive → lower marginal likelihood).

**Connects to next:** These principles are applied concretely to linear regression in Ch. 9.

---

### Ch. 9 — Linear Regression

Linear regression is the simplest case where all the principles from previous chapters become concrete and computable.

**Setup:** y = φ^T(x)θ + ε with ε ~ N(0, σ²). Matrix design: y = Φθ + ε where Φ ∈ ℝ^{N×K} is the feature matrix.

**MLE (analytical solution):** Maximizing the log-likelihood for Gaussian noise is equivalent to minimizing MSE:
L(θ) = ||y - Φθ||² / (2σ²)
∂L/∂θ = 0 → Φ^TΦ θ = Φ^Ty → θ_ML = (Φ^TΦ)^{-1}Φ^Ty

These are the normal equations. If Φ^TΦ is singular, use pseudoinverse or regularize.

**Feature maps φ(x):** Regression is "linear in parameters" but can be nonlinear in data:
- Polynomial: φ(x) = [1, x, x², ..., x^M]^T
- RBF: φ(x) = [exp(-||x-c₁||²/2), ..., exp(-||x-cK||²/2)]^T

**Numerical example (polynomial regression):**

| Degree M | Train R² | Behavior |
|---|---|---|
| 0 | Low | Underfitting — constant |
| 1 | Medium | Linear |
| 3-4 | High | Good fit |
| 6-9 | Perfect | Overfitting — oscillations |

The optimal degree (M=3 or 4 per the book's dataset) is selected via CV or marginal likelihood.

**MAP (regularization):** With prior θ ~ N(0, b²I):
θ_MAP = (Φ^TΦ + λI)^{-1}Φ^Ty where λ = σ²/b²

L2 regularization makes Φ^TΦ + λI always invertible. As λ → ∞: θ_MAP → 0 (underfitting). As λ → 0: θ_MAP → θ_ML (potential overfitting).

**Bayesian linear regression:** Maintains the full distribution over θ.

Prior: p(θ) = N(m₀, S₀)

Posterior (after N data points):
- S_N = (S₀^{-1} + σ^{-2} Φ^TΦ)^{-1}
- m_N = S_N(σ^{-2}Φ^Ty + S₀^{-1}m₀)

Predictive distribution for x*:
p(y*|x*) = N(φ^T(x*)m_N, φ^T(x*)S_Nφ(x*) + σ²)

The first variance term is epistemic uncertainty (over parameters, decreases with more data). The second is aleatoric variance (process noise, irreducible). This uncertainty calibration is crucial for production applications.

**Marginal likelihood for M selection:** p(y|X, M) = ∫p(y|X,θ,M)p(θ|M)dθ. The optimal degree maximizes this quantity — automatically penalizes complexity without a separate validation set.

**GLMs → deep networks:** By replacing φ(x; ψ) with a neural network with parameters ψ (instead of fixed features), Bayesian linear regression over φ(x; ψ) becomes a neural network. Training learns both features and linear weights. The last layer is always linear regression over learned features.

**Connects to next:** PCA (Ch. 10) uses the same algebraic structure but without a response variable — unsupervised learning of representations.

---

### Ch. 10 — Dimensionality Reduction with PCA

PCA finds the low-dimensional linear projection that preserves the maximum variance of the data. It is the classic dimensionality reduction algorithm and the bridge between SVD and representation learning.

**Maximum variance formulation:** Project data onto direction b₁ (||b₁|| = 1):
z₁ = b₁^T x_n, variance = b₁^T S b₁ where S = (1/N)Σ(xₙ-μ)(xₙ-μ)^T

Maximize b₁^T S b₁ subject to ||b₁||² = 1 → Lagrangian → Sb₁ = λ₁b₁. The first principal component is the eigenvector of S with the largest eigenvalue.

For M components: the M eigenvectors of S with the M largest eigenvalues. Captured variance:
V_M = Σᵢ₌₁^M λᵢ / Σⱼ₌₁^D λⱼ (fraction of explained variance)

Lost variance:
J_M = Σⱼ₌ₘ₊₁^D λⱼ

**Minimum reconstruction error formulation:** Project onto M-dimensional subspace and reconstruct. The reconstruction error ||x - x̃||² is minimized with the same M eigenvectors. Both formulations (maximum variance and minimum error) give the same solution — PCA is an optimal linear autoencoder.

**Connection to SVD:** For centered data X ∈ ℝ^{N×D}: X = UΣV^T.
S = (1/N)X^TX = (1/N)VΣU^TUΣ V^T = (1/N)VΣ²V^T

The eigenvalues of S are σ_d²/N (squared singular values, scaled). Eigenvectors of S are the columns of V (right singular vectors).

Best rank-M approximation of X (Eckart-Young): X̄_M = U_MΣ_MV_M^T. This projects and reconstructs the data.

**High-dimensional trick:** If N << D (few data, many features), the covariance matrix S ∈ ℝ^{D×D} is huge. Instead, compute eigenvalues of C = (1/N)XX^T ∈ ℝ^{N×N}. Non-zero eigenvalues are the same as those of S; eigenvectors of S are recovered from those of C: bᵢ = (1/√(Nλᵢ))X^Tcᵢ.

**Practical steps for PCA:**
1. Center data: X ← X - μ
2. Standardize (if features have different scales): X ← X / std
3. Compute covariance matrix S = (1/N)X^TX (or use SVD directly)
4. Eigendecompose: S = VΛV^T
5. Project: Z = X V_M (coordinates in the reduced space)

**PPCA (Probabilistic PCA):** Generative model: x = Bz + μ + ε, z ~ N(0, I), ε ~ N(0, σ²I).
- B ∈ ℝ^{D×M}: factor loading matrix
- Posterior: p(z|x) = N(m, C) with m = (B^TB + σ²I)^{-1}B^T(x-μ), C = σ²(B^TB + σ²I)^{-1}
- As σ² → 0: PPCA converges to classical PCA

**Book example (MNIST, digit "8"):**
- Images: 28×28 = 784 dimensions
- With 1 PC: blurrily recognizable
- With 10 PCs: recognizable as "8"
- With 100 PCs: near-indistinguishable from original
- With 500 PCs: virtually perfect reconstruction (captured variance > 99%)

**Trap:** PCA assumes directions of maximum variance are most informative. If classes differ along low-variance directions, PCA may discard that information. For classification, LDA (Linear Discriminant Analysis) maximizes class separation instead of total variance.

**Connects to next:** GMMs (Ch. 11) extend the idea of modeling data with Gaussians to the multi-cluster case — unsupervised learning of distributions.

---

### Ch. 11 — Density Estimation with Gaussian Mixture Models

GMMs model arbitrarily complex data distributions as a mixture of Gaussians. The EM algorithm for fitting them illustrates the general principle of optimization with latent variables.

**Model:** p(x|θ) = Σ_k π_k N(x|μ_k, Σ_k)
- π_k ≥ 0, Σπ_k = 1: mixture weights
- μ_k, Σ_k: mean and covariance of component k
- K components total

**Why no analytical MLE:** The log-likelihood is:
L = Σₙ log Σ_k π_k N(xₙ|μ_k, Σ_k)

The logarithm of a sum has no closed form. Direct gradient optimization is possible but slow.

**Responsibilities (E-step):** r_{nk} = probability that xₙ came from component k:
r_{nk} = π_k N(xₙ|μ_k, Σ_k) / Σⱼ π_j N(xₙ|μⱼ, Σⱼ)

These are soft assignments: each point belongs to all components with different probabilities.

**M-step (update parameters):** Let N_k = Σₙ r_{nk} (effective number of points in component k):
- μ_k^{new} = (1/N_k) Σₙ r_{nk} xₙ (weighted mean)
- Σ_k^{new} = (1/N_k) Σₙ r_{nk}(xₙ - μ_k)(xₙ - μ_k)^T (weighted covariance)
- π_k^{new} = N_k / N

**Full EM algorithm:**
1. Initialize θ⁰ = {μ_k, Σ_k, π_k} (e.g., with K-means)
2. E-step: compute r_{nk} with current θ
3. M-step: update θ with current r_{nk}
4. Compute log-likelihood; if converged, stop; otherwise go to 2

Fundamental property: each EM iteration does not decrease the log-likelihood. Convergence guaranteed to a local maximum (not necessarily global).

**Numerical example from the book (7 points, 3 components):**

Data: {-2.75, -2.71, -0.50, 0.00, 3.59, 3.64, 3.70}

Converged GMM: p(x) = 0.29 N(x|−2.75, 0.06) + 0.28 N(x|−0.50, 0.25) + 0.43 N(x|3.64, 1.63)

Interpretation: component 3 captures the three rightmost points (more spread out, hence larger variance), while the two leftmost are in separate components.

**Latent variable perspective:** Define latent variable zₙ ∈ {0,1}^K (one-hot: zₙ_k = 1 if xₙ comes from component k). Then:
- p(zₙ_k = 1) = π_k
- p(xₙ|zₙ_k = 1) = N(xₙ|μ_k, Σ_k)
- p(zₙ_k = 1|xₙ) = r_{nk} (posterior = responsibility)

EM is equivalent to maximizing the ELBO (Evidence Lower BOund) = E_q[log p(X,Z|θ)] - KL[q(Z)||p(Z|X,θ)].

**GMM vs K-means:**
| Aspect | K-means | GMM |
|---|---|---|
| Assignment | Hard (one cluster) | Soft (distribution) |
| Cluster shape | Spheres only | Arbitrary ellipses |
| Criterion | Within-cluster variance | Log-likelihood |
| Uncertainty | Not modeled | Yes (π_k, covariances) |

**Connects to next:** SVMs (Ch. 12) tackle the classification problem with a different geometric criterion: maximizing the separation margin.

---

### Ch. 12 — Classification with Support Vector Machines

SVMs are binary classifiers that find the separating hyperplane with maximum margin. The maximum margin provides generalization guarantees, and the kernel trick handles data not linearly separable in the original space.

**Separating hyperplane:** {x : ⟨w, x⟩ + b = 0}. Classes: yₙ = +1 if ⟨w, xₙ⟩ + b > 0, yₙ = -1 if ⟨w, xₙ⟩ + b < 0.

**Margin:** The margin is the distance from the hyperplane to the nearest points of each class. For yₙ ∈ {±1} with ||w|| = 1: margin = 2/||w||. The maximum-margin hyperplane minimizes ||w||.

**Hard margin SVM:** For linearly separable data:
min_{w,b} ½||w||² subject to yₙ(⟨w, xₙ⟩ + b) ≥ 1 for all n

Points satisfying yₙ(⟨w, xₙ⟩ + b) = 1 are the support vectors.

**Soft margin SVM:** For non-separable data, slack variables ξₙ ≥ 0 allow violations:
min_{w,b,ξ} ½||w||² + C Σₙ ξₙ subject to yₙ(⟨w, xₙ⟩ + b) ≥ 1 - ξₙ, ξₙ ≥ 0

C controls the trade-off: large C = penalizes violations strongly (smaller margin, fewer training errors); small C = allows more violations (larger margin, better generalization).

**Hinge loss:** The effective loss of the SVM is:
ℓ(t) = max{0, 1 - yₙ(⟨w, xₙ⟩ + b)} = max{0, 1 - t}

For t ≥ 1 (correctly classified with sufficient margin): loss = 0. For t < 1: linear loss.

**Dual formulation:** Using KKT and the Lagrangian, the primal solution has the form:
w* = Σₙ αₙ yₙ xₙ (representer theorem)

Dual problem:
min_{α} ½ Σᵢ,ⱼ yᵢ yⱼ αᵢ αⱼ ⟨xᵢ, xⱼ⟩ - Σᵢ αᵢ
subject to Σᵢ yᵢ αᵢ = 0, 0 ≤ αᵢ ≤ C

Only points with αₙ > 0 are support vectors and contribute to w*. By KKT complementarity: αₙ = 0 for points with margin > 1 (well separated).

Bias: b* = yₙ - ⟨w*, xₙ⟩ computed at any support vector.

**Kernel trick:** The dual depends only on ⟨xᵢ, xⱼ⟩ = inner products. We replace:
k(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩

without computing φ explicitly. Prediction: sign(Σₙ αₙ yₙ k(x, xₙ) + b*).

**Common kernels:**

| Kernel | Formula | Implicit space |
|---|---|---|
| Linear | ⟨x, x'⟩ | ℝ^D |
| Polynomial | (⟨x, x'⟩ + c)^p | ℝ^{C(D+p,p)} |
| RBF (Gaussian) | exp(-||x-x'||²/(2ℓ²)) | ∞-dimensional |
| Rational Quadratic | (1 + ||x-x'||²/(2αℓ²))^{-α} | ∞-dimensional |

**Gram matrix:** K ∈ ℝ^{N×N} with Kᵢⱼ = k(xᵢ, xⱼ). A kernel is valid (Mercer) iff K is symmetric positive definite for any dataset.

**Numerical solution:** The SVM dual is a convex quadratic programming (QP) problem. Specialized algorithms like SMO (Sequential Minimal Optimization) solve it efficiently in approximately O(N²).

**Convex hull view:** SVMs find the nearest pair of points in the convex hulls of the two classes (one from each class). The maximum-margin hyperplane is perpendicular to the segment connecting those two points and bisects it.

---

## Most important takeaways

1. **The pseudoinverse solves the general least squares case.** θ = (X^TX)^{-1}X^Ty assumes X^TX is invertible. If not (correlated features, more features than data), use λI regularization or SVD for numerical stability.

2. **SVD is the universal decomposition.** Pseudoinverse, PCA, image compression, and the data-covariance connection all derive from A = UΣV^T. For any matrix operation, first ask whether SVD simplifies it.

3. **MAP = MLE + Gaussian prior = regularized least squares.** The choice of prior determines the regularizer form. This unifies three apparently distinct perspectives. Choosing λ is equivalent to choosing the ratio σ²/σ_p².

4. **The Bayesian predictive distribution gives calibrated uncertainty.** Var[y*] = φ^T(x*)S_Nφ(x*) + σ² — the first term decreases with more data (epistemic uncertainty), the second does not (irreducible randomness). In production, this distinction enables knowing when the model is extrapolating.

5. **The EM algorithm maximizes a lower bound (ELBO) of the log-likelihood.** It converges monotonically but may find a local maximum. Initialize with K-means and run multiple random initializations to mitigate this in GMMs.

6. **The kernel trick converts inner products into operations in implicit spaces.** To use SVMs (or any kernelizable algorithm) with nonlinear features, only define k(x,x') — no need to implement φ. The RBF kernel implies infinite-dimensional features at O(N²) computational cost.

7. **Backpropagation is the chain rule applied in reverse mode.** A backward pass computes gradients with respect to all parameters in O(forward pass) time. Automatic differentiation (autograd) automates this — but understanding the underlying Jacobian is necessary to diagnose NaN gradients or gradient explosion/vanishing.

8. **KKT conditions describe the optimum of any convex constrained optimization problem.** Complementarity (λᵢgᵢ = 0) explains why SVMs depend only on support vectors, why LASSO produces sparse solutions, and why dual multipliers are shadow prices in linear programming.

---

## Connection to the platform

| Book concept | Platform application |
|---|---|
| SVD and Eckart-Young | Weight matrix compression (model pruning) and rank analysis in LoRA (A and B are low-rank factors) |
| Backpropagation / autodiff | Foundation of all fine-tuning training; understanding the computational graph is necessary to diagnose gradient issues in deep networks |
| MAP / regularization | Weight decay in AdamW is L2 regularization over parameters — MAP equivalent with Gaussian prior over LoRA deltas |
| Bayesian linear regression | The classification/generation head at the end of an LLM is linear regression over embeddings; the head uncertainty can be calibrated Bayesianly |
| GMMs with EM | Clustering document embeddings for fine-tuning corpus segmentation; identifying data distributions in the training dataset |
| Kernels and inner product | Attention scores are inner products: score(q,k) = q^T k / √d_k — attention is a soft kernel over positions |
| Model selection (marginal evidence) | Selecting fine-tuning hyperparameters (LoRA rank, regularization λ) without overfitting to the validation set |
| PCA / dimensionality reduction | Embedding visualization (UMAP uses similar structure); activation clustering analysis for interpretability |
| Optimization (SGD, Adam) | All fine-tuning; Adam = SGD with adaptive momentum; gradient clipping understood in terms of Jacobian norm |
| Epistemic vs aleatoric uncertainty | In production inference: knowing when the model is out of distribution (high epistemic uncertainty) to decide when to escalate to a human |

**On LoRA and SVD:** The LoRA (Low-Rank Adaptation) technique for efficient fine-tuning represents weight deltas as ΔW = BA where B ∈ ℝ^{d×r} and A ∈ ℝ^{r×k} with r << min(d,k). This is exactly a low-rank approximation — Chapter 4 (Eckart-Young) justifies why fine-tuning updates can be compressed: semantically relevant changes tend to reside in a low-dimensional subspace. Understanding the rank and singular values of ΔW enables estimating how much the model changed from the pre-trained checkpoint.

**On attention as a kernel:** The transformer attention mechanism computes scores = QK^T / √d_k — this is a scaled Gram matrix. Each row is the inner product of one position's query against all other positions' keys. The kernel PSD concepts from Chapter 12 (the Gram matrix must be positive definite for a valid kernel) appear in theoretical attention analysis and in techniques like Performers that approximate attention with low-rank kernels.

**On uncertainty calibration in Latam production:** Bayesian linear regression in Ch. 9 separates epistemic uncertainty (the model lacks sufficient data) from aleatoric uncertainty (irreducible noise). On the platform, this translates to: when a client submits a query outside the distribution of their fine-tuning corpus, the model should report high uncertainty rather than generate confidently. Implementing this in LLMs requires techniques like conformal prediction or Monte Carlo dropout, but the conceptual foundation is exactly the Bayesian predictive distribution from Ch. 9.

---

*Pages read: 1-417 (complete book)*
