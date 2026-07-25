# Fashion-MNIST MLP Classification & Perceptron Convergence Study

**CS3807 – Deep Learning Laboratory · Experiment 2**
Shiv Nadar University Chennai · B.Tech AI & Data Science

A from-scratch implementation of a multi-layer perceptron for Fashion-MNIST classification with automated hyperparameter search, plus a single-layer perceptron study on logic gates demonstrating the linear-separability limit (XOR non-convergence).

---

## Contents

| File | Description |
|---|---|
| `Experiment_2_MLP_FashionMNIST.ipynb` | Full notebook: data pipeline, baseline MLP, hyperparameter search, optimized model, evaluation, and the perceptron/XOR study |
| `Experiment_2_Report.tex` / `.pdf` | Full lab report with results, figures, and discussion |
| `figures/` | All generated plots, saved as 600 DPI EPS (produced on notebook execution) |

## What's Implemented

### 1. MLP on Fashion-MNIST
- Data pipeline: load → flatten (28×28 → 784) → normalize [0,1] → one-hot encode
- Baseline architecture: `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)` (109,386 params)
- Full evaluation suite: accuracy, macro-precision/recall/F1, confusion matrix, classification report

### 2. Automated Hyperparameter Search
- `RandomizedSearchCV` + SciKeras `KerasClassifier` over an 8-dimensional search space (hidden layers, neurons, learning rate, batch size, epochs, optimizer, activation, dropout)
- 25 sampled candidates × 3-fold CV = 75 fits (documented deviation from exhaustive grid search — see report §5 for the runtime justification)
- Best model retrained from scratch on the full training set and evaluated independently on the held-out test set

### 3. Perceptron / Linear Separability Study
- Single-layer perceptron implemented from scratch (NumPy only, no `sklearn.linear_model.Perceptron`)
- Trained on OR, AND, NOT (converge) and XOR (does not converge)
- Weight history logged after every update; decision boundary plotted after every update
- Empirical + theoretical analysis of why XOR breaks the perceptron convergence guarantee

## Results Summary

| Model | Test Accuracy | Macro F1 | Training Time |
|---|---|---|---|
| Baseline MLP | 87.13% | 0.8707 | 108.3s |
| Optimized MLP (search) | 87.08% | 0.8716 | 153.7s |

The hyperparameter search's CV estimate (88.28%) did not fully hold up on the test set — see report §10 for discussion. Full per-class metrics, plots, and the perceptron/XOR results are in the report.

## Requirements

Designed to run on **Google Colab** (GPU runtime recommended).

```bash
pip install scikit-learn<1.6 scikeras>=0.13 kagglehub
```

> `scikeras` currently requires `scikit-learn < 1.6` (breaking change in `__sklearn_tags__`). Install the pinned versions above, then **restart the runtime** before running the rest of the notebook.

Core stack: `tensorflow`, `scikit-learn`, `scikeras`, `numpy`, `pandas`, `matplotlib`, `seaborn`.

## Running

1. Open `Experiment_2_MLP_FashionMNIST.ipynb` in Google Colab.
2. Run the setup cell, then **Runtime → Restart session**.
3. Run all cells top to bottom. Total runtime ≈ 90–100 minutes, dominated by the hyperparameter search (~79 min).
4. Generated figures are saved to `figures/*.eps` (600 DPI) as the notebook executes.

## Notes

- Random seed fixed (`SEED = 42`) for reproducibility of weight initialization and data splits; exact metrics can still drift slightly run-to-run due to GPU non-determinism in TensorFlow.
- All plots use Times New Roman, 13–15pt, with bold axis labels/legends per lab formatting requirements.
- See the full report (`Experiment_2_Report.pdf`) for detailed results, all figures with interpretation, and discussion.

## Author

Prepared for CS3807 – Deep Learning Laboratory, AY 2026–27.
