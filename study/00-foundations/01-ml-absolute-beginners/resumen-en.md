# Summary: Machine Learning for Absolute Beginners
**Author:** Oliver Theobald · Third Edition · 2021
**Pages read:** 1-160 (chapters 1-18, full book)

---

## About

A practical, math-light introduction to Machine Learning. Covers the full workflow of an ML project — from data cleaning to model evaluation — and the most widely used algorithms, with concrete examples and chapter quizzes. Language: Python with Scikit-learn. The book's capstone is a real house price prediction model built end-to-end using the Melbourne Housing dataset.

---

## Key Concepts

- **Machine Learning**: subfield of AI where machines learn from data instead of following explicitly programmed rules. The human defines the algorithm and hyperparameters; the machine finds the patterns.
- **Supervised Learning**: data with known input (X) and output (y). The model learns the X→y relationship. Examples: linear regression, logistic regression, k-NN, SVM, neural networks.
- **Unsupervised Learning**: only inputs (X), no known output. The model discovers hidden patterns. Key example: k-means clustering.
- **Semi-supervised Learning**: mix of labeled and unlabeled data. Trains on labeled examples and uses the model to label the rest.
- **Reinforcement Learning**: model learns by trial and error, accumulating rewards and penalties. Example: Q-learning, video games, self-driving cars.
- **Feature (variable)**: each column in a dataset. X = independent variables (inputs), y = dependent variable (output).
- **Hyperparameter**: algorithm configuration set before training that controls how it learns — not the model's internal parameters (which the model learns itself).
- **Training / Test data**: dataset split for training (70-80%) and evaluation (20-30%). Rule: never test with the same data used for training.
- **Cross-validation (k-fold)**: split data into k buckets, use one as test at each round, rotate. Maximizes use of available data. Recommended for small datasets.
- **Data Scrubbing**: cleaning the dataset before training. The most time-consuming task in any ML project.
- **One-hot Encoding**: convert categorical variables into binary columns (0/1) so algorithms can process them. "red/blue/green" → 3 binary columns.
- **Normalization**: rescale features to a fixed range like [0,1]. Useful when variable magnitudes distort the model.
- **Standardization**: convert to normal distribution with mean 0 and standard deviation 1. Recommended for SVM, PCA, and k-NN.
- **Linear Regression**: predicts continuous values (y = bx + a). The model fits a line (hyperplane) that minimizes residual error.
- **Multiple Linear Regression**: multiple independent variables → `y = a + b₁x₁ + b₂x₂ + ...` Watch out for multicollinearity between variables.
- **Logistic Regression**: classifies into discrete categories using the sigmoid function (S-curve). Outputs probabilities between 0 and 1. Best for binary classification.
- **k-Nearest Neighbors (k-NN)**: classifies a new point based on the majority class of its k nearest neighbors. Simple but computationally expensive on large datasets (O(n) per prediction).
- **k-Means Clustering**: unsupervised algorithm that divides data into k groups based on similarity. Euclidean distance: `d = √((x₂-x₁)² + (y₂-y₁)²)`. Initial k estimate: `√(n/2)`.
- **Bias**: error caused by overly simplistic model assumptions (underfitting). Large systematic gap between predictions and reality.
- **Variance**: scatter of predictions when applied to new data (overfitting). Model memorized the training set but fails to generalize.
- **Regularization**: hyperparameter that penalizes model complexity to combat overfitting.
- **SVM (Support Vector Machine)**: classification algorithm that finds the hyperplane with the maximum margin between classes. Margin = distance from boundary to nearest data point × 2.
- **Kernel Trick**: projects data to a higher dimension (e.g., 2D→3D) so that a linear hyperplane can separate classes that are not linearly separable in the original space.
- **Soft Margin (low C)**: more tolerance for misclassification → better generalization, less overfitting.
- **Hard Margin (high C)**: fewer training errors → risk of overfitting.
- **Activation Function**: threshold that determines whether a node "fires" and passes output to the next layer. Binary (perceptron: ≥0 → 1), sigmoid (0-1), or tanh (-1 to 1).
- **Backpropagation**: weight adjustment process that flows in reverse from the output layer to the input, iteratively minimizing the cost value (prediction error).
- **Black-box Dilemma**: neural networks produce accurate predictions but reveal no insight into how individual variables influence the result.
- **Perceptron**: most basic neural network unit (Frank Rosenblatt, 1950s). Produces binary output (0 or 1). Weakness: small weight changes can flip output dramatically.
- **MLP (Multilayer Perceptron)**: network with multiple layers (input, hidden, output). Aggregates models into one. Slower than logistic regression, faster than SVM.
- **Deep Learning**: networks with 5-10+ hidden layers. Capable of decomposing complex patterns (images, text, video). CNN, RNN, RNTN, Deep Belief Networks.
- **Decision Tree**: hierarchical tree that recursively splits data into homogeneous subsets. Transparent and interpretable. Prone to overfitting via the greedy algorithm.
- **Entropy / Information Gain**: measure of disorder at a node. Goal: select the variable that minimizes entropy at the next split. Formula: `(-p₁ log p₁ - p₂ log p₂) / log2`.
- **ID3**: greedy algorithm (J.R. Quinlan) that picks the variable with lowest entropy at each split. "Greedy" = optimizes locally, not globally.
- **Bagging**: grows multiple decision trees on random bootstrap samples and combines by voting (classification) or averaging (regression). Reduces variance.
- **Random Forests**: like bagging, but also caps the number of variables per split → less correlated trees → more robust. Trained in parallel. 100-150 trees is a good starting point.
- **Gradient Boosting**: builds trees sequentially — each tree corrects the previous one's errors. Very accurate but slow and sensitive to outliers.
- **Ensemble Modeling**: combining multiple models produces a prediction that outperforms any single model. Types: bagging, boosting, bucket of models, stacking.
- **Stacking**: level-0 models run in parallel, their outputs feed into a level-1 blender. Won Netflix Prize (BellKor's Pragmatic Chaos, 2009).
- **Grid Search (GridSearchCV)**: exhaustive search over all hyperparameter combinations. Systematic but exponentially slow.
- **RandomizedSearchCV**: samples random hyperparameter combinations per round. Faster; number of iterations is controllable.
- **MAE (Mean Absolute Error)**: average absolute difference between predictions and actual values. If test MAE is much higher than train MAE → overfitting.

---

## Main Chapters

### Ch. 2 — What is Machine Learning?
ML = learning from data, not from explicit commands. Key distinction from Data Mining: ML analyzes both input AND output to improve future predictions; Data Mining only analyzes inputs to discover existing patterns without self-learning.

### Ch. 3 — Machine Learning Categories
Four categories:
- **Supervised**: labeled data (X, y). Goal: predict y from X.
- **Unsupervised**: no labels. Goal: find hidden structure in X.
- **Semi-supervised**: mostly unlabeled, some labeled. Train on labeled, extend to unlabeled.
- **Reinforcement**: agent learns via rewards and penalties. No dataset — the model generates its own data through interaction.

### Ch. 4 — The ML Toolbox
Three compartments:
1. **Data**: structured (CSV tables) or unstructured (images, audio, text)
2. **Infrastructure**: Python + Jupyter + NumPy + Pandas + Scikit-learn for beginners; TensorFlow/PyTorch + GPU for advanced
3. **Algorithms**: shallow ML (Scikit-learn) vs deep learning (TensorFlow/PyTorch)

GPU note: Andrew Ng (Stanford, 2009) showed GPU clusters could do in one day what a CPU cluster took weeks to compute. GPUs handle the parallel matrix multiplications that ML training requires.

### Ch. 5 — Data Scrubbing
The most time-consuming step. Pipeline order matters:

1. **Feature selection**: remove irrelevant or redundant columns first (reduces the surface area for the rest of the steps)
2. **Row compression**: merge similar rows if applicable
3. **One-hot encoding**: convert text variables to binary columns
4. **Binning**: bucket continuous values into categories when exact magnitude doesn't matter (e.g., age → "18-25", "26-35")
5. **Normalization / Standardization**: uniform variable scale
6. **Missing data**: fill with mode (categorical), median (continuous), or drop rows

Rule: drop irrelevant columns *before* dropping NaN rows — otherwise you might lose a row because of a missing value in a column you were going to delete anyway.

### Ch. 6 — Setting Up Your Data
- Split: 70/30 or 80/20. Always randomize before splitting.
- Small datasets: use k-fold cross-validation to maximize data usage.
- Minimum data rule: 10× the number of features.
- Algorithm choice by dataset size:
  - < 10k rows → clustering, dimensionality reduction
  - < 100k rows → regression, classification
  - > 100k rows → neural networks

### Ch. 7 — Linear Regression
"Hello World" of supervised ML. Formula: `y = bx + a`. The model fits a line (hyperplane) that minimizes residual error (the sum of squared distances from each data point to the line).

Multiple regression: `y = a + b₁x₁ + b₂x₂ + ...`

Key problem to avoid: **multicollinearity** — two independent variables that are strongly correlated with each other. If x₁ and x₂ both encode "income" in different units, their coefficients cancel each other out and the model becomes unreliable. Solution: remove one of them.

### Ch. 8 — Logistic Regression
For predicting discrete classes, not continuous values. Uses the **sigmoid function**:

```
y = 1 / (1 + e^(-x))
```

Output is always between 0 and 1. Default cutoff: 0.5 (above → class 1, below → class 0). Best for binary classification. For multi-class problems, use decision trees or SVM instead.

### Ch. 9 — k-Nearest Neighbors
Classifies a new data point by looking at its k nearest neighbors and taking a majority vote. Distance measured with Euclidean distance. Rules:
- Use odd k to avoid ties
- Requires standardization before training (scale-sensitive)
- Avoid non-critical binary variables (they add noise)
- Slow on large datasets: every prediction requires computing distance to all n training points → O(n) per prediction

### Ch. 10 — k-Means Clustering
Unsupervised algorithm. Algorithm steps:
1. Choose k (number of clusters)
2. Randomly place k centroids
3. Assign each data point to its nearest centroid (Euclidean distance)
4. Recalculate each centroid as the mean of its assigned points
5. Repeat steps 3-4 until centroids stop moving (convergence)

Choosing k:
- Quick estimate: `k ≈ √(n/2)`
- Better: plot SSE (sum of squared errors) vs k → use the "elbow" point where adding more clusters stops reducing SSE significantly (scree plot / elbow method)

### Ch. 11 — Bias & Variance
The central tradeoff of all ML:

| | High Bias | Low Bias |
|---|---|---|
| **High Variance** | Worst case | Overfitting |
| **Low Variance** | Underfitting | Ideal (hard to achieve) |

- **Underfitting (high bias)**: model too simple, can't capture the pattern. Train error is high, test error is high. Fix: more features, more complex model.
- **Overfitting (high variance)**: model too complex, memorized the training data. Train error is low, test error is much higher. Fix: regularization, less complexity, more data.
- **Regularization**: penalizes model complexity. Increases bias slightly but significantly reduces variance. The regularization hyperparameter controls the strength of the penalty.

### Ch. 12 — SVM (Support Vector Machines)
Finds the hyperplane that creates the maximum margin between classes. The "support vectors" are the data points closest to the boundary — they define the margin.

- **Margin** = distance from boundary to nearest point × 2
- **Soft margin (low C)**: allows some misclassifications → better generalization
- **Hard margin (high C)**: forces correct classification of training data → overfitting risk
- **Kernel Trick**: when data is not linearly separable in 2D, project it to 3D (or higher) where a linear plane can separate the classes

SVM caveats:
- Requires standardization (feature scale-sensitive)
- Slow on datasets with low feature-to-row ratio
- Excellent for small/medium datasets with high dimensionality (many features relative to rows)

### Ch. 13 — Artificial Neural Networks
Inspired by brain neurons. Structure: nodes (neurons) connected by edges (axons). Each edge has a **weight**. The sum of weighted inputs is passed through an **activation function** to decide if the node fires.

Node computation:
```
sum = x1*w1 + x2*w2 + x3*w3
output = activation_function(sum)
```

**Perceptron example** (from the book):
- Input 1: x1 = 24, weight w1 = 0.5 → contribution: 12
- Input 2: x2 = 16, weight w2 = -1.0 → contribution: -16
- Sum: 12 + (-16) = -4 → activation function (≥0 = 1, else 0) → output: **0** (did not fire)
- Adjust weights: w2 becomes -0.5 → contribution: -8
- New sum: 12 + (-8) = 4 → output: **1** (fires)

**Backpropagation**: after each forward pass, the error (cost value) is measured. The algorithm then flows in reverse, adjusting each weight proportionally to its contribution to the error. This repeats until the cost converges to a minimum.

**Activation function types**:
- **Perceptron**: binary step (0 or 1). Weakness: tiny weight changes can flip output.
- **Sigmoid neuron**: output between 0 and 1. More stable for small weight adjustments.
- **Tanh**: output between -1 and 1. Can represent negative relationships.

**Black-box dilemma**: two networks with different architectures can produce the same output, making it impossible to trace which variables drove the prediction. Decision trees are the transparent alternative.

**Network architecture**:
- **Input layer**: one node per feature
- **Hidden layer(s)**: intermediate processing. More hidden layers = more capacity to find complex patterns.
- **Output layer**: one node per class (classification) or one node (regression)

**Deep Learning**: 5-10+ hidden layers. "Deep" refers to the depth of layers, not the number of neurons. Object recognition systems (self-driving cars) use 150+ layers. Applications: image recognition (CNN), speech/text (RNN), time series (RNN), classification (MLP, Deep Belief Networks).

### Ch. 14 — Decision Trees
Tree structure: root node → branches (splits) → leaf nodes. Terminal node = leaf with no more splits.

Building the tree: at each node, the algorithm selects the variable that minimizes **entropy** (disorder) at the next level. This is the **ID3 algorithm** (Iterative Dichotomizer 3, J.R. Quinlan). Process: **recursive partitioning** — repeat until stopping criterion is met (< 3-5 items per leaf, or all items belong to one class).

**Entropy formula**: `(-p₁ log p₁ - p₂ log p₂) / log2`

**Worked example** from the book (10 employees, predict promotion):
| Variable | Entropy | Result |
|---|---|---|
| Exceeded KPIs | 0 bits | Perfect split — two homogeneous groups |
| Aged < 30 | 0.6895 bits | One homogeneous group |
| Leadership Capability | 0.9508 bits | Both groups mixed |

Winner: **Exceeded KPIs** (entropy = 0). The tree splits on this variable first and terminates immediately — no further splits needed.

**Overfitting in decision trees**: the greedy algorithm optimizes locally at each split without considering global impact. A slightly worse first split might produce a globally better model. Like a kid eating the best cupcake first without thinking about the overall meal.

**Bagging**: grows N trees on random bootstrap samples of the data, combines predictions by voting (classification) or averaging (regression). Reduces variance by exposing different trees to different data.

**Random Forests**: extends bagging by also capping the number of variables considered at each split. This forces trees to use different variables, making them less correlated with each other. Less correlation = more independent errors = more reliable average. Trained in parallel. Start with 100-150 trees.

**Gradient Boosting**: sequential ensemble. Each new tree focuses on the examples the previous trees got wrong. Errors from round N are weighted higher in round N+1. Very accurate but: (1) slow because trees are sequential, (2) tends to overfit with many outliers (it obsessively tries to correct every error).

Random Forest vs Gradient Boosting:
- Outliers: Random Forest wins (voting dilutes impact)
- Accuracy: Gradient Boosting wins (more focused learning)
- Speed: Random Forest wins (parallel)

### Ch. 15 — Ensemble Modeling
Core idea: aggregate multiple models to reduce the risk of any single model being wrong. The analogy is polling multiple doctors instead of trusting one diagnosis.

- **Classification**: combine via voting (majority wins)
- **Regression**: combine via numeric averaging

Four ensemble techniques:
1. **Bagging**: parallel, homogeneous (same algorithm, different data samples). Reduces variance.
2. **Boosting** (Gradient Boosting, AdaBoost): sequential, homogeneous. Reduces bias. AdaBoost specifically targets classification problems.
3. **Bucket of Models**: trains multiple different algorithms on the same data, picks the best on test data. Heterogeneous.
4. **Stacking**: all models run in parallel (level-0), their outputs are fed to a meta-learner blender (level-1) that produces the final prediction. Neural network + decision tree is a classic heterogeneous stack: the neural network handles cases with complete data; the decision tree handles missing values.

Stacking won the Netflix Prize (2006-2009): team BellKor's Pragmatic Chaos used linear stacking of hundreds of different models to improve recommendation accuracy by 10.06%.

### Ch. 16 — Development Environment
Recommended setup: **Anaconda** (includes Jupyter, NumPy, Pandas, Scikit-learn) + **Jupyter Notebook** (web-based code editor at `localhost:8888`).

Key Pandas commands:
```python
import pandas as pd

df = pd.read_csv('~/Downloads/dataset.csv')  # load CSV
df.head()                                     # preview first 5 rows
df.head(10)                                   # preview first N rows
df.iloc[100]                                  # get row at index 100
df.columns                                    # list all column names
```

Note: Python indexes from 0. `df.iloc[100]` returns the 101st row.

### Ch. 17 — Building a Model in Python
Full end-to-end gradient boosting model on the Melbourne Housing dataset (34,857 rows, 21 variables, predicting house price).

**6-step workflow:**

**Step 1 — Import libraries**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
```

**Step 2 — Import dataset**
```python
df = pd.read_csv('~/Downloads/Melbourne_housing_FULL.csv')
```

**Step 3 — Scrub dataset**
```python
# Remove irrelevant columns
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']   # note: misspelled in source file
del df['Longtitude']  # note: misspelled in source file
del df['Regionname']
del df['Propertycount']

# Drop rows with any missing values
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Suburb', 'CouncilArea', 'Type'])

# Assign X and y
X = df.drop('Price', axis=1)
y = df['Price']
```

**Step 4 — Split the dataset**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True
)
```

**Step 5 — Configure algorithm and hyperparameters**
```python
model = ensemble.GradientBoostingRegressor(
    n_estimators=150,      # number of decision trees
    learning_rate=0.1,     # shrinks contribution of each tree
    max_depth=30,          # max layers per tree (WARNING: causes overfitting)
    min_samples_split=4,   # min samples required to create a new branch
    min_samples_leaf=6,    # min samples required in each leaf
    max_features=0.6,      # fraction of features considered per split
    loss='huber'           # error function (robust to outliers)
)

model.fit(X_train, y_train)
```

Hyperparameter reference:
| Param | Effect |
|---|---|
| `n_estimators` | More trees = more accuracy (up to a point) + slower |
| `learning_rate` | Lower = each tree contributes less → better generalization |
| `max_depth` | Higher = more complex trees → overfitting risk |
| `min_samples_split` | Higher = harder to create new branches → simpler trees |
| `min_samples_leaf` | Higher = each leaf needs more samples → less overfitting |
| `max_features` | Float (0.6) = 60% of features randomly selected per split |
| `loss` | `huber` = combination of `ls` and `lad`, robust to outliers |

**Step 6 — Evaluate results**
```python
mae_train = mean_absolute_error(y_train, model.predict(X_train))
mae_test  = mean_absolute_error(y_test,  model.predict(X_test))

print("Training MAE: %.2f" % mae_train)   # → $27,834.12
print("Test MAE:     %.2f" % mae_test)    # → $168,262.14
```

Result analysis: training error ($27k) is much lower than test error ($168k) → **overfitting**. The model memorized training patterns but fails to generalize. Root cause: `max_depth=30` made each tree too complex.

### Ch. 18 — Model Optimization
Starting from the overfitted model (max_depth=30), two optimizations:

**Optimization 1 — Reduce max_depth from 30 to 5**
```
Training MAE: $135,283.69   (was $27k — increased)
Test MAE:     not reported  (gap with training narrowed)
```
Training error went up, but the model generalizes better. This is correct — higher training error often means less overfitting.

**Optimization 2 — Increase n_estimators from 150 to 250**
```
Training MAE: $124,469.48
Test MAE:     $161,602.45   (was $168k — improved)
```
More trees improved both metrics. The train/test gap also narrowed.

**Grid Search — exhaustive hyperparameter search:**
```python
from sklearn.model_selection import GridSearchCV

model = ensemble.GradientBoostingRegressor()

hyperparameters = {
    'n_estimators':    [200, 300],
    'max_depth':       [4, 6],
    'min_samples_split': [3, 4],
    'min_samples_leaf':  [5, 6],
    'learning_rate':   [0.01, 0.02],
    'max_features':    [0.8, 0.9],
    'loss':            ['ls', 'lad', 'huber']
}

grid = GridSearchCV(model, hyperparameters, n_jobs=4)
grid.fit(X_train, y_train)

print(grid.best_params_)   # returns optimal combination

mae_train = mean_absolute_error(y_train, grid.predict(X_train))
mae_test  = mean_absolute_error(y_test,  grid.predict(X_test))
```

Grid search limitation: 2×2×2×2×2×2×3 = 192 combinations × cross-validation folds = very slow. Strategy: run coarse grid first (powers of 10: 0.01, 0.1, 1, 10), identify the best region, then run a fine-grained grid around that region.

**RandomizedSearchCV**: instead of testing all combinations, samples random values from each range. Faster and allows you to control exactly how many trials to run via `n_iter`.

Optimization principles:
- Change one hyperparameter at a time
- Feature selection: adding/removing one variable at a time and measuring MAE impact is often more effective than exhaustive grid search
- The gap between train and test MAE is the key signal — minimize the gap, not just the test error

---

## Key Takeaways

1. **The workflow is always the same:** raw data → scrubbing → train/test split → choose algorithm → train → evaluate → tune hyperparameters. Memorize this loop.

2. **Scrubbing is 80% of the real work.** Algorithms are trivial to call with Scikit-learn. The art is in preparing the data correctly. Drop irrelevant columns before dropping NaN rows.

3. **Choose the algorithm based on the problem:** predict a continuous number → linear regression; classify into categories → logistic regression / k-NN / SVM / decision trees; group without labels → k-means; complex patterns with many features → ANN / deep learning; maximum accuracy on tabular data → gradient boosting.

4. **Bias-Variance is the central tradeoff of all ML.** A high train/test MAE gap always signals overfitting. Reducing model complexity (max_depth, regularization) or adding data are the primary fixes.

5. **Transparency has an accuracy cost.** Decision trees are interpretable but fragile. Neural networks and ensemble models are more accurate but black-box. Choose based on the use case — if a legal team needs to explain rejections, choose a decision tree.

6. **In production, ensembles win almost every time.** Random forests or gradient boosting outperform individual models in the vast majority of cases. The Netflix Prize was won by stacking hundreds of models.

7. **Hyperparameter optimization is iterative.** Change one at a time and measure the impact before changing another. Grid search is exhaustive but exponentially slow; RandomizedSearch is more practical for initial exploration.

---

## Platform Connection

| Book concept | Direct application in the fine-tuning platform |
|---|---|
| Data Scrubbing | The validation and sanitization pipeline for client JSONL datasets before fine-tuning. One-hot encoding, null handling, normalization — all of this runs before the job starts. |
| Train/Test Split | In fine-tuning: the client dataset split between training and evaluation data. Controls the quality of the resulting model. |
| Hyperparameters | Parameters the user configures when launching a job: learning rate, epochs, LoRA rank. Exactly the hyperparameters from this book, just in an LLM context. |
| GPU infrastructure | The chapter explains why GPUs are necessary for neural networks. The platform uses A100/H100 — exactly this. |
| MAE / error metrics | In the platform: perplexity, loss curves in MLflow. Equivalent to the book's MAE — used to evaluate whether the model improved. |
| Bias-Variance tradeoff | In fine-tuning: underfitting when epochs are too few or learning rate too low; overfitting when the model memorizes the training dataset. Validation perplexity is exactly the test MAE from this book. |
| ANN / Deep Learning | The conceptual foundation of all LLMs. Backpropagation, activation functions, layers — these are the exact mechanisms operating inside the transformers you fine-tune. |
| Gradient Boosting | Conceptual analogue to iterative fine-tuning: each step improves the model relative to the previous error. The `learning_rate` in GBR and in LoRA fine-tuning serve the same role. |
| Ensemble Modeling | Your platform can offer multi-model evaluation (comparing results from multiple fine-tuned checkpoints) as a value-added service. |
| Grid Search / RandomizedSearch | Equivalent to hyperparameter tuning experiments in MLflow. Clients should be able to launch multiple jobs with different configs and compare results side by side. |
| Model Optimization (Ch. 18) | The loop the platform should expose to users: adjust one hyperparameter, retrain, compare loss, iterate. The gap between train loss and val loss is the key signal to surface in the UI. |
| k-Means Clustering | Concrete use case for clients: customer segmentation without prior labels (e.g., grouping users by behavior for targeted prompting). |
