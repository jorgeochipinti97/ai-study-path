# Exam: Mathematics for Machine Learning
**Progressive difficulty · 14 questions**

---

## Section 1 — Fundamental concepts

**Question 1**

Which of the following statements about the Moore-Penrose pseudoinverse is **correct**?

A) θ† = (X^TX)^{-1}X^T only exists when X^TX is singular.
B) θ† solves the least squares problem ||Xθ - y||² exactly regardless of the rank of X.
C) If X^TX is invertible, then θ† = (X^TX)^{-1}X^T and the least squares solution θ_ML = θ†y is unique.
D) The pseudoinverse maximizes the norm of the solution among all possible solutions.

---

**Question 2**

Given a linear regression model with prior θ ~ N(0, b²I) and noise ε ~ N(0, σ²I), the MAP estimate is:
θ_MAP = (Φ^TΦ + λI)^{-1}Φ^Ty

What does λ represent in this expression?

A) The noise variance σ²
B) The ratio σ²/b² between the noise variance and prior variance
C) The prior variance b²
D) The number of model parameters

---

**Question 3**

Consider the following conceptual Python snippet:

```python
# Forward pass
z1 = W1 @ x + b1          # Layer 1: W1 ∈ ℝ^{m×n}, x ∈ ℝ^n
a1 = relu(z1)              # Activation
z2 = W2 @ a1 + b2          # Layer 2: W2 ∈ ℝ^{p×m}
loss = mse(z2, y)

# Backward pass
dL_dz2 = 2 * (z2 - y) / n
dL_dW2 = dL_dz2 @ ???      # ← what goes here?
```

Which expression is correct for the gradient of the loss with respect to W2?

A) `dL_dz2 @ W2`
B) `dL_dz2.T @ a1.T`
C) `dL_dz2 @ a1.T`
D) `a1 @ dL_dz2.T`

---

**Question 4**

True or False (justify your answer):

"In the EM algorithm for GMMs, the log-likelihood of the model can decrease between consecutive iterations."

---

**Question 5**

A polynomial regression model is fit to a training dataset. The following results are observed:

| Degree M | Train MSE | Validation MSE |
|---|---|---|
| 1 | 0.82 | 0.85 |
| 4 | 0.12 | 0.15 |
| 9 | 0.001 | 2.47 |

What is the correct diagnosis for M=9?

A) The degree-9 model is the best because it has the lowest training error.
B) The degree-9 model is overfitting: it memorized training set noise and fails to generalize.
C) The degree-9 model is underfitting because the gap between train and validation is too large.
D) The high validation MSE of M=9 indicates the learning rate was too high during training.

---

## Section 2 — Practical application

**Question 6**

You have a binary classification dataset with 500 examples in ℝ^{10000} (10,000 features, more features than data). You want to train an SVM with RBF kernel.

Describe:
1. Why the dual formulation is preferable over the primal in this case.
2. What the support vectors are and what happens to examples that are not support vectors.
3. How you would choose the hyperparameters C and ℓ (RBF kernel bandwidth), and what trade-off each controls.

---

**Question 7**

You have a dataset of 10,000 grayscale images of 1024×1024 pixels. You want to reduce dimensionality with PCA for visualization (2D) and to compress the representation (100 components).

1. The covariance matrix S would be of size 1,048,576 × 1,048,576. How do you compute PCA without building that matrix?
2. After centering and standardizing, how much of the variance should ideally be captured by 100 good components? How do you measure it?
3. Describe the problem that can arise if you do not center the data before applying PCA.

---

**Question 8**

The following code has a subtle error:

```python
def bayesian_linear_regression_predict(Phi_star, m_N, S_N, sigma2):
    """
    Phi_star: feature vector for new point, shape (K,)
    m_N: posterior mean, shape (K,)
    S_N: posterior covariance, shape (K, K)
    sigma2: noise variance (scalar)
    """
    mean_pred = Phi_star @ m_N
    # Predictive variance
    var_epistemic = Phi_star @ S_N @ Phi_star  # epistemic uncertainty
    var_pred = var_epistemic + sigma2
    return mean_pred, var_pred
```

Identify the error, explain why it is an error (not just syntactically but conceptually), and write the corrected version.

---

**Question 9**

You are fitting a GMM with K=3 components to a dataset of 1,000 points in ℝ². After 50 EM iterations, the log-likelihood has stabilized but the result looks poor: one component captures 95% of the data (π₁ ≈ 0.95) and the other two are nearly empty (π₂, π₃ ≈ 0.025).

1. What caused this result?
2. How would you modify the training procedure to avoid it?
3. How would you decide whether K=3 is the correct number of components?

---

## Section 3 — Synthesis and judgment

**Question 10**

Connect the book's concepts to answer: Why is MAP estimation with a Gaussian prior equivalent to regularized least squares with L2 penalty? When would you prefer full Bayesian inference over MAP?

Develop the mathematical derivation and explain the practical implications of each choice for a production model.

---

**Question 11**

A colleague claims: "You should always use an RBF kernel instead of the linear kernel in an SVM, because RBF can learn any nonlinear function while the linear is just a special case."

Evaluate this claim. Where is it right? Where does it fail? When would the linear kernel be the best choice?

---

**Question 12**

The Eckart-Young theorem states that the best rank-M approximation of a matrix in Frobenius norm consists of the first M terms of the SVD. Explain:

1. Why this mathematically justifies that PCA captures maximum variance with M components.
2. How this result connects to LoRA (Low-Rank Adaptation) for LLM fine-tuning.
3. A fundamental limitation of this justification when applied to transformer weight updates.

---

## Section 4 — Applied to the platform

**Question 13 — System design**

You are building the fine-tuning module of your platform for a LATAM client with:
- Dataset: 3,000 (instruction, response) pairs in Rioplatense Spanish
- Base model: 7B parameter LLM (Llama-3-7B)
- Hardware: 1 A100 GPU with 40GB
- Objective: adapt the model to local slang and conversational style; do not change factual knowledge

Using concepts from the book, answer:
1. What fine-tuning technique do you recommend (full fine-tuning vs LoRA vs QLoRA)? Justify using low-rank decomposition and regularization concepts.
2. How do you understand LoRA's rank hyperparameter r from the perspective of Eckart-Young low-rank approximation?
3. What loss function would you use and how does it connect to MLE over the vocabulary?
4. How would you avoid overfitting with only 3,000 examples? Mention at least two techniques from the book.

---

**Question 14 — Production diagnosis**

A client on your platform reports the following problem: their fine-tuned model has validation loss = 1.23 (good), but in production users report that responses "sound weird" and the model sometimes generates irrelevant text for queries on certain topics.

1. Formulate hypotheses about what could be happening, using the book's language (data distribution, generalization, epistemic vs aleatoric uncertainty).
2. How would you diagnose whether the problem is distributional shift (production data is different from the training set)?
3. What metrics or visualizations would you use to confirm the diagnosis?
4. What solutions would you propose, ordered by implementation cost?

---

## Answers

### Answer 1

**Correct: C**

A) Incorrect: the pseudoinverse exists when X^TX is **invertible** (not singular). If X^TX is singular, SVD is used to define a generalized pseudoinverse.

B) Incorrect: θ† solves least squares **exactly** only when an exact solution exists (Xθ = y). In the general case (overdetermined system), it finds the θ with minimum norm that minimizes ||Xθ - y||².

C) Correct: when X^TX is invertible (full column rank), the pseudoinverse is (X^TX)^{-1}X^T and gives the unique least squares solution.

D) Incorrect: it is the opposite. Among all solutions that minimize the residual, the pseudoinverse gives the one with **minimum norm**.

---

### Answer 2

**Correct: B**

Derivation: θ_MAP = argmin[-log P(X|θ) - log P(θ)]. With Gaussian noise N(0, σ²): -log P(X|θ) ∝ (1/2σ²)||y - Φθ||². With Gaussian prior N(0, b²I): -log P(θ) = (1/2b²)||θ||². Adding and factoring (1/2σ²): minimize ||y - Φθ||² + (σ²/b²)||θ||². Therefore λ = σ²/b².

Intuition: large λ = narrow prior (parameters must be small) OR large noise (data are not very informative) → more regularization.

---

### Answer 3

**Correct: C** — `dL_dz2 @ a1.T`

By the chain rule: ∂L/∂W2 = ∂L/∂z2 · ∂z2/∂W2. Since z2 = W2 @ a1, we have ∂z2/∂W2 = a1^T in the sense that dL/dW2[i,j] = dL/dz2[i] · a1[j]. In matrix form: dL/dW2 = (dL_dz2)[:, None] @ a1[None, :] = dL_dz2 @ a1.T (for dL_dz2 of shape (p,) and a1 of shape (m,), this gives W2_grad of shape (p,m) ✓).

A) Incorrect: dL_dz2 @ W2 would give dimensions (p,) @ (p,m) → dimension error.
B) Incorrect: dL_dz2.T @ a1.T = (p,) @ (m,) → not compatible as matrix product.
D) Incorrect: a1 @ dL_dz2.T = (m,) @ (p,) → gives shape (m,p) instead of (p,m).

---

### Answer 4

**False.**

The EM algorithm guarantees that the log-likelihood does not decrease between iterations. This is proven formally: each iteration maximizes the ELBO (lower bound on the log-likelihood), which implies L(θ^{t+1}) ≥ L(θ^t). This monotone convergence property is one of EM's fundamental guarantees. What can happen is that the log-likelihood improves in increasingly small increments and "stabilizes" at a local maximum (not global), but it never decreases.

---

### Answer 5

**Correct: B**

M=9 exhibits classic overfitting. Train MSE ≈ 0 indicates the degree-9 polynomial fits the training points perfectly (including noise). Validation MSE = 2.47 is 2,470 times larger than train MSE, indicating the model memorized the data rather than learning the underlying function.

A) Incorrect: minimum training error is not the objective — the objective is minimum generalization error (validation/test).

C) Incorrect: underfitting is the opposite — when the model is too simple to capture the signal. M=9 is too complex.

D) Incorrect: learning rate is not relevant here; the polynomial regression solution is analytical (normal equations), not iterative.

---

### Answer 6

**Model answer:**

**1. Why use the dual:** In the primal formulation, the optimization variable is w ∈ ℝ^D with D = 10,000. In the dual, the variables are α ∈ ℝ^N with N = 500. Since N << D, the dual problem is much smaller (500 variables vs 10,000). Additionally, the dual depends only on inner products k(xᵢ, xⱼ), enabling the kernel trick to map to implicit high- (or infinite-) dimensional feature spaces at no extra cost. In the primal with kernels, one would need to explicitly work with φ(x) which may be infinite-dimensional.

**2. Support vectors:** Support vectors are points with αₙ > 0. By the KKT complementarity condition: λₙgₙ(x*) = 0, which implies αₙ > 0 only when the constraint is active, i.e., when yₙ(⟨w,xₙ⟩ + b) = 1 (points exactly on the margin). Points with αₙ = 0 (correctly classified with margin > 1) do not contribute to w* = Σₙ αₙ yₙ xₙ — the hyperplane does not depend on them at all. For 500 examples, typically only 10-50 will be support vectors.

**3. Selecting C and ℓ:** Use K-fold cross-validation (K=5 or K=10) on a log-scale grid: C ∈ {0.01, 0.1, 1, 10, 100} and ℓ ∈ {0.1, 1, 10, 100}. Trade-offs: large C → penalizes violations strongly → smaller margin → possible overfitting; small C → more tolerant of errors → larger margin → possible underfitting. Large ℓ → RBF kernel decays slowly → distant points are similar → smoother model; small ℓ → decays quickly → only nearby neighbors matter → more local model.

---

### Answer 7

**Model answer:**

**1. High-dimensional trick:** Instead of computing S = X^TX ∈ ℝ^{D×D} (impossible for D = 1,048,576), compute C = XX^T ∈ ℝ^{N×N} with N = 10,000. The non-zero eigenvalues of C are the same as those of S (scaled). Eigenvectors of S are recovered: if Ccₙ = λₙcₙ, then bₙ = X^Tcₙ / ||X^Tcₙ|| is the n-th eigenvector of S. Complexity: O(N²D) for C vs O(D²N) for S — with N=10,000 and D=1,048,576, the difference is a factor of ~100x.

**2. Variance captured:** V_100 = Σᵢ₌₁^{100} λᵢ / Σⱼ₌₁^D λⱼ. For natural images, the first 100 components typically capture 80-95% of variance (image eigenvalues decay rapidly). To measure: compute the cumulative eigenvalue ratio and plot the "elbow" of the curve.

**3. Problem without centering:** Without centering, the first principal component will capture the direction of the data mean (the "average color" of images) rather than the direction of maximum variance. All components will be biased. In the worst case, the mean consumes almost all variance and the remaining components are irrelevant. Centering: X ← X - μ (subtract the pixel-wise mean image).

---

### Answer 8

**Error:** The deepest conceptual error is the absence of a numerical stability guarantee: `Phi_star @ S_N @ Phi_star` for Phi_star of shape (K,) and S_N of shape (K,K) produces a scalar in NumPy (which is numerically correct for a valid SPD matrix S_N). However, if S_N loses positive definiteness due to accumulated floating-point errors across many posterior updates, `var_epistemic` can become negative, leading to an invalid (negative) predictive variance.

The corrected version:

```python
def bayesian_linear_regression_predict(Phi_star, m_N, S_N, sigma2):
    """
    Phi_star: (K,)  — feature vector for new point
    m_N: (K,)       — posterior mean
    S_N: (K, K)     — posterior covariance (must be SPD)
    sigma2: scalar  — noise variance
    """
    mean_pred = Phi_star @ m_N
    # Epistemic variance: phi^T S_N phi (positive scalar if S_N is SPD)
    var_epistemic = Phi_star @ S_N @ Phi_star
    # Total variance: epistemic (decreases with data) + aleatoric (constant)
    var_pred = var_epistemic + sigma2
    # Guard against numerical errors making S_N indefinite
    var_pred = max(var_pred, sigma2)
    return mean_pred, var_pred
```

Conceptual note: the variance cannot be less than sigma2 (the irreducible aleatoric component). This guard ensures the output is physically meaningful even when S_N suffers from floating-point drift.

---

### Answer 9

**Model answer:**

**1. Cause:** Poor initialization (likely random with all centroids near the same cluster). Component 1 "won" the early competition because its initial responsibilities were higher, which fed more weight in the M-step, increasing its likelihood further, and so on (dominant component collapse). EM converged to a suboptimal local maximum.

**2. Solutions:**
- Initialize with K-means: K-means centroids give a reasonable starting point for μ_k.
- Run multiple initializations (5-20) with different random seeds; keep the one with the highest final log-likelihood.
- Use K-means++ for initialization: distributes initial centroids by maximizing pairwise separation.
- Add covariance regularization: Σ_k ← Σ_k + εI to prevent singularities and collapse.

**3. Selecting K:** Compute BIC for K ∈ {1, 2, 3, 4, 5}: BIC(K) = k_params·ln(N) - 2·L(K), where k_params = K·(D + D(D+1)/2 + 1) - 1 for full covariance. Choose K minimizing BIC. Alternatively, use marginal likelihood (Bayes factor) comparing models, or visual inspection of the elbow in the log-likelihood vs K curve.

---

### Answer 10

**Model answer:**

**Derivation:**

MAP estimation minimizes the negative log-posterior:
-log p(θ|X,y) = -log p(y|X,θ) - log p(θ) + const

For linear regression with Gaussian noise y ~ N(Φθ, σ²I):
-log p(y|X,θ) = (1/2σ²)||y - Φθ||² + const₁

For prior θ ~ N(0, b²I):
-log p(θ) = (1/2b²)||θ||² + const₂

Adding: θ_MAP = argmin (1/2σ²)||y - Φθ||² + (1/2b²)||θ||²
= argmin ||y - Φθ||² + λ||θ||² where λ = σ²/b²

This is exactly regularized least squares with L2 penalty (Ridge regression).

**When to use full Bayesian inference vs MAP:**

MAP gives a point estimate — computationally simple but does not quantify uncertainty over θ. If θ_MAP lies in a high-curvature region (many parameters with similar likelihoods), the point estimate can produce overconfident predictions.

Full Bayesian inference maintains the posterior distribution p(θ|X,y) = N(m_N, S_N) and predicts via:
p(y*|x*) = N(φ^T m_N, φ^T S_N φ + σ²)

The epistemic term φ^T S_N φ quantifies disagreement about θ in the direction of x*. It decreases with more data in that region.

**In production:** Use full Bayesian inference when the cost of a confidently wrong prediction is high (medicine, financial decisions, systems where the model must know when it does not know). MAP is sufficient when the data volume is large and residual uncertainty is irrelevant to the application.

---

### Answer 11

**Evaluation:**

**Where it is right:** The RBF kernel has universal approximation capacity — with enough data and the right bandwidth, it can approximate any continuous function. The linear kernel can only classify with hyperplanes, which fails for data with nonlinear decision boundaries.

**Where it fails — cases where the linear kernel is better:**

1. **High dimensionality with linearly separable data:** In very high-dimensional spaces (text with TF-IDF in 100,000 features), data are frequently linearly separable. The linear kernel is O(N) in the dual representation vs O(N²) for RBF. Faster, fewer hyperparameters to tune.

2. **Large N:** The RBF kernel requires storing and evaluating the Gram matrix K ∈ ℝ^{N×N}. For N > 10,000, this is O(N²) memory and time. The linear kernel can be implemented directly with w = Σ αₙyₙxₙ without materializing K.

3. **Implicit regularization of the linear kernel:** In high-dimensional spaces, the linear kernel has appropriate inductive bias (parsimony). RBF with very small ℓ can perfectly overfit.

4. **Interpretability:** The weights w of the linear kernel are interpretable: each component wᵢ indicates the importance of feature i. With RBF, the decision depends on distances to support vectors — difficult to interpret.

**Practical rule:** Always try the linear kernel first. If the data has obvious nonlinear structure and N is manageable (< 10,000), switch to RBF with CV over C and ℓ.

---

### Answer 12

**Model answer:**

**1. Eckart-Young → PCA connection:**

The centered data matrix X ∈ ℝ^{N×D} can be decomposed as X = UΣV^T. The sample covariance S = (1/N)X^TX = (1/N)VΣ²V^T. The variance in the direction of eigenvector bₘ (= column m of V) is λₘ = σₘ²/N (the corresponding eigenvalue).

By Eckart-Young, the best rank-M approximation of X in Frobenius norm is X̄_M = U_MΣ_MV_M^T. The error ||X - X̄_M||_F² = Σⱼ₌ₘ₊₁^D σⱼ² = N · J_M where J_M is the lost variance. Minimizing the Frobenius reconstruction error = minimizing lost variance = maximizing retained variance. Both formulations converge to exactly the same solution: the M eigenvectors with the largest eigenvalues.

**2. LoRA connection:**

LoRA parametrizes ΔW = BA with B ∈ ℝ^{d×r} and A ∈ ℝ^{r×k}. This is a rank-r approximation of the full update. The empirical justification is that gradients during fine-tuning have low-rank structure: the gradient matrix ∇W_L has rapidly decaying singular values, meaning the "semantic content" of fine-tuning fits in a subspace of dimension r << min(d,k). Eckart-Young says the best rank-r compression of ΔW is exactly the truncated SVD decomposition — which LoRA approximates.

**3. Limitation:**

Eckart-Young minimizes reconstruction error in Frobenius norm, which weights all matrix elements equally. But in a transformer, not all weights are equally important for the task. The gradients relevant for fine-tuning may be concentrated in specific dimensions that are not the ones with the largest norm. Additionally, Eckart-Young is optimal for approximating W itself, but LoRA updates only the delta ΔW — there is no guarantee that the optimal rank of ΔW matches that of W. In practice, LoRA's rank r is a hyperparameter chosen by CV, not formally derived from Eckart-Young.

---

### Answer 13

**Model answer:**

**1. Recommended technique: QLoRA**

With 1 A100 GPU (40GB) and a 7B parameter model:
- Full fine-tuning of 7B parameters in fp32 requires ~112GB just for weights (28 bytes per parameter with optimizer states). Impossible on 40GB.
- LoRA: quantizes the base model to fp16/bf16 (~14GB), adds low-rank adapters. The base model is frozen; only B and A are trained for each layer. With rank r=16 on all attention matrices of a 7B, approximately 20M trainable parameters are added (~0.3% of total).
- QLoRA: quantizes the base model to 4-bit NF4 (~4GB) with double quantization, loads LoRA adapters in fp16. Allows training the model in ~8GB total, leaving room for batch and gradients.

From Eckart-Young's perspective: the ΔW of fine-tuning for style adaptation (not factual knowledge) should have low rank — the change is a gentle rotation in the representation space, not a complete rewrite of the knowledge base.

With 3,000 examples, the fine-tuning is small and the risk of catastrophic forgetting is low when using LoRA (base model is frozen).

**2. Rank r as hyperparameter:**

r controls how many "degrees of freedom" the fine-tuning has to modify each layer. r=1: only a single adjustment vector per layer (like tuning the bias of a projection). r=64: can capture more complex changes but requires more data to avoid overfitting. For 3,000 examples and style adaptation, r ∈ {8, 16} is appropriate. If the singular values of the trained ΔW decay rapidly (verify by computing SVD of ΔW post-training), it confirms the chosen r was sufficient.

**3. Loss function:**

Cross-entropy over the vocabulary: L = -(1/T)Σₜ log p(token_t | tokens_<t ; θ). This is MLE: maximize the probability of the response tokens. For instruction fine-tuning, compute loss only over response tokens (not instruction tokens), so the model learns to complete responses, not to copy instructions.

**4. Avoiding overfitting with 3,000 examples:**

- **L2 regularization (weight decay):** In AdamW, weight_decay = λ acts as a Gaussian prior over LoRA parameters — equivalent to MAP. Typical value: 0.01-0.1.
- **Early stopping:** Monitor validation loss on a 10% holdout. Stop when validation loss starts increasing (overfitting beginning). With 3,000 examples, the validation set could be 300 pairs.
- **Dropout** on LoRA adapters (lora_dropout ∈ 0.05-0.1).
- **Data augmentation**: paraphrase instructions or responses to increase effective dataset diversity.

---

### Answer 14

**Model answer:**

**1. Hypotheses using the book's language:**

- **Distributional shift:** p_train(x) ≠ p_prod(x). The validation set was drawn from the same distribution as the training set, so validation loss is good. But production queries come from real users with a different distribution (different topics, lengths, question formality).
- **High epistemic uncertainty in uncovered regions:** In Bayesian regression, Var_epistemic = φ^T(x*)S_Nφ(x*) is high for x* outside the training distribution. The model extrapolates in those regions — equivalent to making predictions with high uncertainty without reporting it.
- **Incomplete coverage of the fine-tuning dataset:** If certain topics or query types of the client are not represented in the 3,000 pairs, the model will fall back to the base model's behavior (which may not be aligned with the desired slang or style).

**2. Diagnosing distributional shift:**

- Extract embeddings (e.g., last transformer layer) from 500 production queries and 500 training set examples.
- Apply PCA or UMAP on the combined embeddings.
- If the point clouds overlap: no shift. If they are separated: there is shift in the representation space.
- Statistical test: Maximum Mean Discrepancy (MMD) between the two distributions. A high MMD (> threshold calibrated on the training set) confirms shift.
- Alternatively: train a binary classifier (train vs production) on the embeddings. If it achieves accuracy > 70%, the distributions are distinguishable.

**3. Metrics and visualizations:**

- PCA/UMAP 2D of embeddings: direct visualization of the shift.
- Distribution of query lengths in train vs production (histogram).
- Vocabulary distribution: top-K most frequent tokens in production vs training.
- Perplexity of the model on production queries vs training set: high perplexity on production indicates out-of-distribution queries.
- Attention heatmap in final layers: if problematic queries have irregular attention patterns (very uniform or very concentrated), it may indicate inputs with no representation in the fine-tuning.

**4. Solutions ordered by implementation cost:**

1. (Low cost) **Curate and expand the fine-tuning dataset** with examples of the topics where the model fails. Run a second fine-tuning round.
2. (Medium cost) **Retrieval-Augmented Generation (RAG):** for queries on specific topics, retrieve relevant documents from the client's corpus and include them in the context. No retraining required.
3. (Medium cost) **Confidence filter in production:** use perplexity or token distribution entropy to detect out-of-distribution queries. If confidence is low, return a fallback message or route to a human operator.
4. (High cost) **Recollect production data** (with consent), annotate it, and run a new fine-tuning round with the real production distribution. This is the correct MLOps cycle.
