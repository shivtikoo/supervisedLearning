# Supervised Learning — From Scratch

Building **linear regression**, **logistic regression**, and a **neural network** from the ground up using only NumPy. No sklearn estimators — all gradient descent implemented manually.

## Notebooks

| Notebook | Contents |
|----------|---------|
| [script.ipynb](script.ipynb) | Linear regression, Ridge/Lasso, feature engineering, learning rate search, logistic regression |
| [neuralNetworks.ipynb](neuralNetworks.ipynb) | Neural network from scratch — width/depth experiments, forward pass vs backprop, comparison with logistic regression |

## Contents

| Section | Topics |
|---------|--------|
| 1. Regression | Linear, Ridge (L2), Lasso (L1) |
| 2. Feature Engineering | Hand-crafted features, polynomial expansion |
| 3. Learning Rate Optimisation | Grid search over α |
| 4. Classification | Binary logistic regression |

**Datasets:** Diabetes · California Housing · Breast Cancer WI · Iris

---

## Datasets

| Task | Dataset | Load via | Shape |
|------|---------|----------|-------|
| Regression | Diabetes | `sklearn.datasets.load_diabetes` | 442 × 10 |
| Regression | California Housing | `sklearn.datasets.fetch_california_housing` | 20640 × 8 |
| Classification | Breast Cancer WI | `sklearn.datasets.load_breast_cancer` | 569 × 30 |
| Classification | Iris (binary) | `sklearn.datasets.load_iris` (classes 0 & 1) | 100 × 4 |

**Note:** Diabetes features are pre-standardized (mean ≈ 0, std ≈ 0.047). California Housing is raw — `StandardScaler` applied before training.

---

## Results

### 1. Linear Regression (plain gradient descent)

| Dataset | MSE | R² |
|---------|-----|-----|
| Diabetes | 2886.9824 | 0.4551 |
| California Housing | 0.5562 | 0.5756 |

### 2. Regularized Regression (λ = 0.1)

| Model | Diabetes MSE | Diabetes R² | California MSE | California R² |
|-------|:-----------:|:-----------:|:--------------:|:-------------:|
| Linear (baseline) | 2886.98 | 0.4551 | 0.5562 | 0.5756 |
| Ridge (L2) | 2899.05 | 0.4528 | 0.5559 | 0.5758 |
| Lasso (L1) | 2900.14 | 0.4526 | 0.5559 | 0.5758 |

### 3. Feature Engineering (California Housing)

| Feature Set | R² | Δ |
|-------------|-----|---|
| Baseline (8 features) | 0.5757 | — |
| + 3 engineered features (11 total) | 0.6337 | **+0.0580** |
| + polynomial degree-2 (77 total) | 0.4485 | **−0.1272** |

### 4. Learning Rate Grid Search (engineered California, 10K epochs)

| α | R² | Notes |
|---|-----|-------|
| 0.0001 | 0.0292 | Far too small — never converged |
| 0.0005 | 0.5767 | Converging but slow |
| 0.001 | 0.6016 | Reasonable |
| 0.005 | 0.6308 | Good |
| 0.01 | 0.6337 | Good |
| 0.05 | **0.6347** | **Best** — plateau starts here |
| 0.1 | 0.6347 | Same as 0.05 |

### 5. Logistic Regression (Binary Classification)

| Dataset | Accuracy | TN | FP | FN | TP |
|---------|---------|----|----|----|----|
| Breast Cancer WI | **98.25%** | 42 | 1 | 1 | 70 |
| Iris (setosa vs versicolor) | **100%** | 12 | 0 | 0 | 8 |

---

## Key Findings

### Regularization barely helped — why, and what theory says

**What we got:** Plain linear regression matched or beat Ridge and Lasso on both datasets. Max improvement was Δ 0.002 R².

**What theory says regularization requires to work:**
- The model must be **overfitting** — training error significantly lower than test error
- **High-dimensional data:** many features relative to samples (p >> n)
- **Correlated features:** multicollinearity inflates coefficient variance — L2 stabilizes this
- **Noisy/sparse data:** many irrelevant features — L1 zeros them out

**Why it didn't work here:**
1. **Diabetes is too small and already clean.** 442 samples, 10 pre-decorrelated features. No overfitting to correct.
2. **California is too large relative to model complexity.** 20K samples, 8 features — a linear model has far more data than parameters, so variance is already low.
3. **λ = 0.1 was not tuned.** The right λ matters as much as the right α. Without a sweep, we can't conclude regularization doesn't help — only that this λ doesn't.
4. **Regularization has the biggest effect on high-variance models.** High-degree polynomials, deep networks, or hundreds of features — not on clean 8–10 feature linear models.

**When you would see a real difference:** polynomial regression (77+ features), noisy datasets, or datasets where n < p.

### Ridge vs Lasso

At λ = 0.1, they're interchangeable on these datasets. Lasso's feature-zeroing effect needs larger λ to activate. The key distinction:
- **Ridge wins** when all features contribute — it shrinks weights proportionally
- **Lasso wins** when features are sparse/irrelevant — it drives weak weights to exactly zero

### Feature engineering > regularization on clean data

| Approach | R² gain on California |
|----------|----------------------|
| Regularization (Ridge/Lasso) | ≤ 0.0002 |
| 3 hand-crafted features | +0.0580 |

Hand-crafted features gave 290× more improvement than regularization. This is the expected priority order: better features > hyperparameter tuning > regularization on clean tabular data.

### Polynomial features hurt (−12.7 pp R²) — but that's a convergence bug, not a model flaw

11 features → 77 features (degree-2). The optimizer needs far more gradient steps. At 100K epochs with `lr=0.01` it still hasn't converged. This is an optimization failure, not a generalization failure — with `lr=0.05` or a closed-form solver, polynomial features would very likely help.

### Learning rate is the highest-leverage hyperparameter

The range from worst to best α spans R² 0.03 to 0.63 — a 20× difference in explained variance. Get this right before tuning anything else. The plateau at α ≥ 0.05 is typical: beyond that, the gradient direction is the bottleneck, not the step size.

### Logistic regression: near-perfect on both datasets

- **Breast Cancer (98.25%):** 30 cell-nucleus morphology features are highly discriminative. Only 2 errors on 114 test samples.
- **Iris binary (100%):** Setosa and versicolor are linearly separable in 4D feature space — gradient descent finds the boundary cleanly.

---

## Regularization

Standard loss minimizes prediction error only. Regularization adds a penalty on weights:
- **Ridge (L2):** `Loss = MSE + λ·Σwᵢ²` — shrinks weights, keeps all features
- **Lasso (L1):** `Loss = MSE + λ·Σ|wᵢ|` — drives weak weights to zero, selects features

---

## What to explore next

- **λ sweep:** grid-search λ like we did for α — find where Lasso actually zeros out weights
- **Polynomial + higher α:** re-run polynomial features with `lr=0.05` or normal equations to verify the convergence hypothesis
- **Multi-class logistic regression:** OvR or softmax for the full 3-class Iris problem
- **Regularized logistic regression:** apply L1/L2 penalty to the classifier
- **Bias-variance decomposition:** plot train vs test error curves to visualize where overfitting starts
