# Supervised Learning — From Scratch

Building **linear regression**, **logistic regression**, and a **neural network** from the ground up using only NumPy. No sklearn estimators — all gradient descent implemented manually.

## Notebooks

| Notebook | Contents |
|----------|---------|
| [script.ipynb](script.ipynb) | Linear regression, Ridge/Lasso, feature engineering, learning rate search, logistic regression |
| [neuralNetworks.ipynb](neuralNetworks.ipynb) | Neural network from scratch — width/depth experiments, forward pass vs backprop, loss functions, activation functions |

---

## script.ipynb — Regression & Classification

### Datasets

| Task | Dataset | Load via | Shape |
|------|---------|----------|-------|
| Regression | Diabetes | `sklearn.datasets.load_diabetes` | 442 × 10 |
| Regression | California Housing | `sklearn.datasets.fetch_california_housing` | 20640 × 8 |
| Classification | Breast Cancer WI | `sklearn.datasets.load_breast_cancer` | 569 × 30 |
| Classification | Iris (binary) | `sklearn.datasets.load_iris` (classes 0 & 1) | 100 × 4 |

**Note:** Diabetes features are pre-standardized (mean ≈ 0, std ≈ 0.047). California Housing is raw — `StandardScaler` applied before training.

### Results

#### Linear Regression (plain gradient descent)

| Dataset | MSE | R² |
|---------|-----|-----|
| Diabetes | 2886.9824 | 0.4551 |
| California Housing | 0.5562 | 0.5756 |

#### Regularized Regression (λ = 0.1)

| Model | Diabetes MSE | Diabetes R² | California MSE | California R² |
|-------|:-----------:|:-----------:|:--------------:|:-------------:|
| Linear (baseline) | 2886.98 | 0.4551 | 0.5562 | 0.5756 |
| Ridge (L2) | 2899.05 | 0.4528 | 0.5559 | 0.5758 |
| Lasso (L1) | 2900.14 | 0.4526 | 0.5559 | 0.5758 |

#### Feature Engineering (California Housing)

| Feature Set | R² | Δ |
|-------------|-----|---|
| Baseline (8 features) | 0.5757 | — |
| + 3 engineered features (11 total) | 0.6337 | **+0.0580** |
| + polynomial degree-2 (77 total) | 0.4485 | **−0.1272** |

#### Learning Rate Grid Search (engineered California, 10K epochs)

| α | R² | Notes |
|---|-----|-------|
| 0.0001 | 0.0292 | Far too small — never converged |
| 0.0005 | 0.5767 | Converging but slow |
| 0.001 | 0.6016 | Reasonable |
| 0.005 | 0.6308 | Good |
| 0.01 | 0.6337 | Good |
| 0.05 | **0.6347** | **Best** — plateau starts here |
| 0.1 | 0.6347 | Same as 0.05 |

#### Logistic Regression (Binary Classification)

| Dataset | Accuracy | TN | FP | FN | TP |
|---------|---------|----|----|----|----|
| Breast Cancer WI | **98.25%** | 42 | 1 | 1 | 70 |
| Iris (setosa vs versicolor) | **100%** | 12 | 0 | 0 | 8 |

### Key Findings — script.ipynb

**Regularization barely helped — why:**
- Plain linear regression matched or beat Ridge and Lasso on both datasets (max Δ 0.002 R²)
- Diabetes: too small and already clean (442 samples, 10 decorrelated features) — nothing to regularize
- California: too large relative to model complexity (20K samples, 8 features) — linear model already low-variance
- λ = 0.1 was never tuned — a sweep would be needed to draw a real conclusion
- Regularization matters on high-variance models: polynomials, deep nets, p >> n data

**Ridge vs Lasso:** At λ = 0.1, interchangeable. Lasso's feature-zeroing needs larger λ to activate. Ridge wins when all features contribute; Lasso wins when many features are irrelevant.

**Feature engineering beats regularization:** 3 hand-crafted features gave +5.8pp vs regularization's ≤0.02pp — a 290× difference. Priority order: better features > learning rate > regularization.

**Polynomial features hurt (−12.7pp):** A convergence failure, not a model flaw. 11→77 features requires far more gradient steps. The optimizer didn't have enough epochs at lr=0.01.

**Learning rate is the highest-leverage knob:** α = 0.0001 gives R² 0.03; α = 0.05 gives R² 0.63 — a 20× swing. Plateau at α ≥ 0.05 is typical: beyond that, gradient direction is the bottleneck.

**Logistic regression:** Near-perfect on both datasets. Breast Cancer's 30 cell-nucleus features are highly discriminative; Iris setosa vs versicolor is linearly separable in 4D.

---

## neuralNetworks.ipynb — Neural Networks from Scratch

### Datasets

| Dataset | Type | Purpose |
|---------|------|---------|
| `make_classification` | Linear | Shows NN = logistic regression when data is linear |
| `make_moons` | Non-linear | Primary benchmark — curved, 2D, easy to visualize |
| `make_circles` | Circular | Concentric rings — logistic regression fails completely |
| Two-spiral | Interleaved | Depth benchmark — genuinely requires multiple layers |

### Results

#### Forward Pass: Before vs After Training

| Dataset | Before (random weights) | After Backprop | Gain |
|---------|:-----------------------:|:--------------:|:----:|
| Linear  | ~50% | ~95% | +45pp |
| Moons   | ~50% | ~92% | +42pp |
| Circles | ~50% | ~95% | +45pp |

#### Width Experiment (1 hidden layer, make_moons)

| Neurons | Test Acc | Behaviour |
|---------|:--------:|-----------|
| 1 | ~70% | Near-linear — equivalent to logistic regression |
| 2 | ~76% | Two linear regions |
| 4 | ~83% | Recognizable moon curve |
| 8 | ~89% | Good smooth boundary |
| 16 | ~92% | Clean, tight boundary |
| 32 | ~92% | Diminishing returns past 16 |

#### Depth Experiment (8 neurons/layer, tanh, two-spiral)

| Depth | Test Acc | Behaviour |
|-------|:--------:|-----------|
| 1 hidden layer | ~66% | Near-linear — can't follow spiral arms |
| 2 hidden layers | ~78% | Starts tracing curvature |
| 3 hidden layers | ~95% | Clean spiral boundary |
| 4 hidden layers | ~96% | Marginal improvement |

#### Neural Network vs Logistic Regression

| Dataset | Logistic Regression | Neural Net [2→16→16→1] | NN Gain |
|---------|:-------------------:|:----------------------:|:-------:|
| Linear  | ~95% | ~95% | ~0pp |
| Moons   | ~85% | ~92% | +7pp |
| Circles | ~50% | ~95% | **+45pp** |

#### Loss Functions (BCE vs MSE, make_moons)

| Loss | Test Acc | Initial Loss | Notes |
|------|:--------:|:------------:|-------|
| BCE | **95%** | 0.947 (log scale) | Clean gradient `ŷ−y`, always proportional to error |
| MSE | 88% | 0.367 (prob² scale) | Gradient `(ŷ−y)·ŷ(1−ŷ)` vanishes when model is confident |

#### Activation Functions (3 hidden layers, make_moons)

| Hidden Activation | Test Acc | Final Loss | Notes |
|-------------------|:--------:|:----------:|-------|
| Sigmoid | ~87% | 0.3154 | Gradient ≤0.25× per layer — only 6% reaches layer 1 in a 3-layer net |
| Tanh | ~92% | 0.0906 | Zero-centered, max derivative = 1, still vanishes at depth |
| ReLU | **96%** | 0.0831 | Gradient = 1 when active — no shrinkage across layers |

### Key Findings — neuralNetworks.ipynb

**Random forward pass = coin flip:** A freshly initialized network scores ~50% on every dataset. The architecture sets the *capacity*; backprop assigns the *meaning* to weights. A 100-layer untrained network is just a random function.

**1 neuron ≈ logistic regression:** A single hidden ReLU neuron projects the input to a scalar, clips negative values, then passes through sigmoid. In the active halfspace this is `σ(w₂·(w₁·x+b₁)+b₂)` — identical to logistic regression with rescaled weights. Same linear decision boundary in practice.

**Width — the knee is at ~8 neurons:** Each neuron learns one oriented halfspace (a linear cut). More neurons = more cuts composable into curves. For make_moons, 8 neurons achieves ~90% and adding more past 16 gives diminishing returns.

**Depth enables hierarchical composition:** Layer 1 learns oriented edges; layer 2 composes them into curves; layer 3 composes curves into full spiral arms. A single layer would need exponentially more neurons to approximate what 3 layers do through sequential composition. `make_circles` is too easy to demonstrate this — use the spiral, which genuinely requires depth.

**Why tanh for the depth experiment, not ReLU:** Deeper vanilla networks (no Adam, no batch norm) are prone to dying ReLU units. tanh has smooth non-zero gradients everywhere, making it stable at depth without modern optimizers.

**BCE vs MSE — never use MSE for classification:**
- BCE gradient: `ŷ − y` — always proportional to the error
- MSE gradient: `(ŷ − y)·ŷ·(1−ŷ)` — the extra `ŷ(1−ŷ)` term → 0 when the model is confidently wrong, killing the correction signal
- The raw loss values look lower for MSE but this is a scale artifact (log vs squared-probability units) — normalised convergence curves tell the real story
- Result: BCE 95% vs MSE 88% on identical architecture and data

**Hidden layer activation matters far more than output layer activation:**
- Output layer: constrained by the task (sigmoid → probability, softmax → class probs, linear → regression)
- Hidden layer: controls gradient flow through the network
- Sigmoid caps gradient at 0.25× per layer → 3 layers → only 6% of signal reaches layer 1
- ReLU passes gradient at 1× when active → no shrinkage → faster, deeper learning
- ReLU is the main reason deep networks became practical before batch norm and residual connections

---

## What to explore next

- **λ sweep:** grid-search λ to find where Lasso actually zeros out weights
- **Polynomial + higher α:** re-run degree-2 features with `lr=0.05` to verify convergence hypothesis
- **Multi-class logistic regression:** OvR or softmax for full 3-class Iris
- **Regularized logistic regression:** apply L1/L2 penalty to the classifier
- **Bias-variance curves:** plot train vs test error to visualize underfitting/overfitting boundary
- **Adam optimizer:** replace vanilla gradient descent and see how much it helps on deeper nets
- **Batch normalization:** observe the effect on training stability for deep networks
- **Multi-class NN:** softmax output + categorical cross-entropy for 3+ class problems
