# Supervised Learning — From Scratch

Implementing linear and logistic regression from scratch. Manually tuning learning rate `α` and regularization strength `λ` to observe their effect on model performance.

## Regularization
Standard loss just minimizes prediction error. Regularization adds a penalty on the weights:
- **Ridge (L2):** `Loss = MSE + λ·Σw²` — shrinks weights, keeps all features
- **Lasso (L1):** `Loss = MSE + λ·Σ|w|` — drives weak weights to zero, selects features

`λ` is set by hand. Higher = simpler model. Experiment is finding the sweet spotand if regularization actually helps the model 

## Datasets
| Model       | Dataset            | Load via                                    |
|-------------|--------------------|--------------------------------------------|
| Regression  | Diabetes           | `sklearn.datasets.load_diabetes`           |
| Regression  | California Housing | `sklearn.datasets.fetch_california_housing`|
| Logistic    | Breast Cancer WI   | `sklearn.datasets.load_breast_cancer`      |
| Logistic    | Iris               | `sklearn.datasets.load_iris`               |


## What to tune
- `α` (learning rate): controls step size during gradient descent
- `λ` (regularization): controls how hard weights are penalized
- Type: Ridge vs Lasso — does L1 actually zero out features here and if always useful?
