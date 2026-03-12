# Exam: Machine Learning for Absolute Beginners
**Progressive difficulty · 14 questions**

---

## Section 1 — Fundamentals (comprehension)

**1.** Which of the following best describes the difference between Machine Learning and Data Mining?

A. Data Mining analyzes both input and output to improve future predictions
B. Machine Learning only analyzes inputs to find patterns without self-learning
C. Machine Learning analyzes input and output to improve predictions; Data Mining analyzes inputs without self-learning
D. They are synonymous — no meaningful difference exists

---

**2.** True or False: In k-Nearest Neighbors, using an even value of k is recommended to avoid bias in classification.

---

**3.** What problem does the following scenario describe? A model has an MAE of $3,000 on the training set and an MAE of $140,000 on the test set.

A. Underfitting (high bias)
B. Overfitting (high variance)
C. The model is perfectly calibrated
D. An error in data scrubbing

---

**4.** What is the purpose of the Kernel Trick in SVM?

A. Reduce the number of features in the dataset
B. Project data to a higher dimension to make non-linearly separable classes separable
C. Increase the C value to harden the margin
D. Apply normalization before training

---

**5.** Which of the following is NOT a type of ensemble modeling?

A. Bagging
B. Stacking
C. Binning
D. Boosting

---

## Section 2 — Practical Application

**6.** You have an e-commerce customer dataset with 50,000 rows and the following columns: age (numeric), country (text), product_purchased (text), amount_spent (numeric), churn (binary: 1/0). You want to predict whether a customer will churn. Describe the scrubbing steps needed before training the model.

---

**7.** Explain the difference between Random Forests and Gradient Boosting in terms of: (a) how trees are built, (b) training speed, and (c) sensitivity to outliers. When would you use each?

---

**8.** In the book's exercise (Melbourne Housing), the initial model had `max_depth=30` and produced train MAE of $27k vs test MAE of $168k. Reducing `max_depth` to 5 raised the train MAE to $135k. Why can raising training error be an improvement? What concept explains this?

---

**9.** A client asks you to build a loan approval/rejection system. The legal team requires that each rejection be explainable to the customer. Would you use a deep neural network or a decision tree? Justify your answer considering the black-box dilemma.

---

## Section 3 — Synthesis

**10.** Describe the Bias-Variance tradeoff. Why can't both bias and variance be low simultaneously? Include in your answer the concept of regularization and how it affects each term.

---

**11.** Compare these three algorithms for a classification task on a dataset with 500 rows and 8 features: Logistic Regression, k-NN, and Random Forests. Which would you recommend and why? Consider: transparency, speed, overfitting risk, and dataset size.

---

**12.** The book describes a 6-step workflow for building a model in Python. Describe each step and explain why the order matters — what happens if steps are done out of order?

---

## Section 4 — Applied to the Platform

**13.** In an LLM fine-tuning platform, clients upload JSONL datasets to train models. How do the Bias-Variance and Model Optimization concepts from the book map to a client's experience on the platform? Describe a concrete scenario where the client sees underfitting and how they should correct it.

---

**14.** The platform lets clients configure hyperparameters before launching a fine-tuning job (learning rate, epochs, LoRA rank). Design a "hyperparameter optimization" feature based on the Grid Search and RandomizedSearch concepts from the book. What trade-offs would each option have in the context of a SaaS platform (GPU cost, wait time, user experience)?

---

## Answers

**1.** C — Machine Learning analyzes both input and output to improve future predictions. Data Mining only analyzes inputs to discover patterns without self-learning.

**2.** False — An odd k value is recommended to avoid ties when classifying by majority vote.

**3.** B — Overfitting (high variance). The model learned the training set's patterns too well but fails to generalize to new data. The large gap between train and test MAE is the classic indicator.

**4.** B — The Kernel Trick projects data to a higher dimension (e.g., 2D → 3D) where a linear boundary can separate classes that were not linearly separable in the original space.

**5.** C — Binning is a data scrubbing technique (converting continuous values into categories), not an ensemble modeling method.

**6.** Scrubbing steps:
- Feature selection: evaluate whether `product_purchased` contributes signal or has excessive cardinality (may require grouping or removal)
- One-hot encoding: convert `country` and `product_purchased` into binary columns with `pd.get_dummies()`
- Missing values: check for nulls in each column; fill with median for `age` and `amount_spent` (continuous), or drop rows if few are missing
- Target variable is `churn` (y=1/0); remaining columns are features (X)
- Standardization: apply before training if using k-NN or SVM; not required for decision trees/random forests

**7.**
- (a) Random Forests: builds trees in parallel on random data samples, limiting features per split (bootstrap sampling). Gradient Boosting: builds trees sequentially, each one correcting the previous tree's errors with weights.
- (b) Random Forests: faster (parallel). Gradient Boosting: slower (sequential).
- (c) Random Forests: more robust to outliers — the vote from 100+ trees dilutes their impact. Gradient Boosting: more sensitive to outliers because each tree is forced to learn from previous errors, including those caused by outliers.
- Use Random Forests when: the dataset has many outliers, a fast benchmark model is needed, or training time is a constraint. Use Gradient Boosting when: maximum accuracy is needed and the dataset has consistent patterns.

**8.** Raising training MAE is an improvement because it indicates the model stopped memorizing the training set and learned more generalizable patterns. The concept that explains this is the Bias-Variance tradeoff: reducing max_depth increases bias (the model is less flexible) but decreases variance (it generalizes better). The gap between train and test MAE narrows, which is the actual goal: predicting well on unseen data, not on data the model already saw.

**9.** Decision tree. The black-box dilemma of neural networks makes it impossible to explain the reasoning behind each decision — you cannot reconstruct the path of variables that led to a loan rejection. A decision tree, in contrast, is fully transparent: you can show the customer exactly which variables and thresholds led to the rejection ("Income < $X and Age < Y → Rejected"). The legal requirement for explainability is non-negotiable, and decision trees are the only option that guarantees it here.

**10.** The Bias-Variance tradeoff states that reducing one term tends to increase the other. A simple model (few features, low depth) has high bias (doesn't capture all patterns) but low variance (predicts consistently). A complex model has low bias (captures all training patterns) but high variance (overfits, fails on new data). Regularization penalizes model complexity, pushing toward a balance point: it reduces variance (less overfitting) at the cost of a slight increase in bias. The regularization hyperparameter controls how much penalty is applied.

**11.** Recommendation: Random Forests. Justification:
- 500 rows is a small-to-medium dataset; k-NN and Logistic Regression both work at this scale, but Random Forests is more robust to overfitting
- k-NN on 500 rows can be slow (O(n) per prediction) and sensitive to irrelevant features
- Logistic Regression is the most interpretable but only works well when the relationship is approximately linear
- Random Forests handles mixed variables (numeric and categorical), requires no standardization, is robust to outliers, and with 100-150 trees produces a strong baseline quickly
- If the client needs interpretability, use a simple Decision Tree — though with higher overfitting risk

**12.** The 6 steps and why order matters:
1. Import libraries: must come first so the rest of the code has access to functions
2. Import dataset: must exist before it can be processed
3. Scrubbing: must happen before the split — doing it after can cause data leakage if transformations are computed on test data. Dropna, one-hot encoding should be computed only on training data
4. Train/test split: must come after scrubbing but before training
5. Configure and instantiate the algorithm: must be defined before calling `model.fit()`
6. Evaluate: must be the last step, using data the model has never seen

Critical wrong order: scrubbing after the split can cause data leakage. Evaluating with training data completely invalidates the evaluation.

**13.** Underfitting scenario in fine-tuning:
- Client uploads a dataset of 500 examples to fine-tune a support ticket classification model
- Configures 1 epoch and learning_rate=0.00001 (very conservative)
- Result: the loss curve doesn't converge, the model shows high perplexity on the validation set — equivalent to a high training MAE with a small gap from test MAE
- This is underfitting / high bias: the model didn't learn enough from the dataset
- Fix: increase epochs (from 1 to 3-5), slightly increase learning_rate, or use a higher LoRA rank to give the model more learning capacity
- The platform should display the loss curve in real time so the client can visually identify whether the model is converging

**14.** Grid Search vs RandomizedSearch on a SaaS platform:
- Grid Search: the client defines ranges (e.g., epochs=[3,5,10], lr=[0.0001,0.001,0.01]) and the platform launches N×M×K jobs. Pros: systematic and exhaustive. Cons: potentially very high GPU cost (3×3×3 = 27 jobs), long wait times, excessive for budget-constrained users
- RandomizedSearch: the client defines ranges and a max number of attempts (e.g., 5 jobs). The platform samples random combinations. Pros: controllable cost, acceptable results in less time, better UX. Cons: may miss the optimal combination
- Platform recommendation: offer RandomizedSearch by default with a configurable job limit (e.g., 3-5 jobs), and Grid Search as an advanced option with a cost estimate before confirmation. Display the best result with its hyperparameters so the client can manually iterate from that point.
