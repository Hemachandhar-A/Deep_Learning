# Experiment 1 — Single Layer Perceptron for Banknote Authentication

**Course:** CS3807 – Deep Learning Laboratory
**Institution:** Shiv Nadar University Chennai, B.Tech AI & Data Science, Semester V
**Repository:** https://github.com/Hemachandhar-A/Deep_Learning.git

A from-scratch implementation of a Single Layer Perceptron (plain NumPy, no ML framework) trained on the UCI Banknote Authentication dataset, plus a full exploratory data analysis, evaluation suite, and a set of additional experiments (learning-rate sweep, XOR limitation, feature normalization effect, and per-update logic-gate training for OR/AND/NOT/XOR).

---

## 1. What's in this repository

| File / Folder | Description |
|---|---|
| `Experiment_1_Perceptron.ipynb` | The main deliverable — a single Jupyter notebook containing all code, outputs, and inline results for this experiment. |
| `data_banknote_authentication.txt` | The raw UCI Banknote Authentication dataset (1372 rows, no header). |
| `figures/` | All 15 generated plots, exported as vector `.eps` files at 600 dpi (per the lab's formatting rules). |
| `Experiment_1.tex` | The LaTeX lab report, referencing the same figures and results produced by the notebook. |
| `Experiment_1.pdf` | Compiled PDF of the lab report. |
| `README.md` | This file. |

---

## 2. Dataset

- **Source:** [UCI Machine Learning Repository — Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
- **Instances:** 1372
- **Features (4, all continuous):** `variance`, `skewness`, `curtosis`, `entropy` — all derived from a wavelet transform of banknote images
- **Target:** `class` — `0` = Authentic, `1` = Forged
- **File format:** comma-separated, **no header row** (the notebook assigns column names on load)

The dataset file must sit in the **same directory** as the notebook, since it's loaded with a relative path:

```python
df = pd.read_csv("data_banknote_authentication.txt", header=None, names=COLUMN_NAMES)
```

---

## 3. What the notebook does

The notebook is organized to mirror the lab manual's task structure, plus additional tasks:

1. **Setup** — imports, a fixed random seed, and one shared Matplotlib style (Times New Roman, bold axes/legends, 600 dpi EPS export) used by every plot.
2. **Task 1 — Dataset Exploration:** shape, dtypes, missing-value check, class balance, descriptive statistics.
3. **Task 2 — EDA:** feature histograms, correlation heatmap, scatter plots, boxplots (all clubbed into multi-panel figures).
4. **Task 3 — Preprocessing:** feature standardization (fit on train only) and an 80/20 stratified train/test split.
5. **Task 4 — Perceptron (from scratch):** a single `Perceptron` class — weight/bias initialization, step activation, forward pass, and the classic perceptron learning rule.
6. **Task 5 — Training:** per-epoch training log (misclassified count, weights, bias) plus training-dynamics plots.
7. **Task 6 — Evaluation:** accuracy, precision, recall, F1-score, and a confusion matrix on the held-out test set.
8. **Task 7 — Learning Rate Study:** the same model retrained at η = 0.001, 0.01, and 0.1, compared side by side.
9. **Optional — Decision Boundary:** a 2-feature (variance, skewness) perceptron with its separating hyperplane plotted directly.
10. **Additional Task — Scikit-learn comparison:** a sanity check against `sklearn.linear_model.Perceptron` on the same split.
11. **Additional Task — Step vs. Sigmoid:** a visual side-by-side of the two activation shapes.
12. **Additional Task — XOR:** the perceptron trained on the XOR truth table, demonstrating non-convergence.
13. **Additional Task — Normalization effect:** raw vs. standardized features compared under identical hyperparameters (with a documented caveat about shared RNG state across sequential runs — see the report's Discussion section).
14. **Additional Task — Logic Gates (OR / AND / NOT / XOR):** weights and decision boundary plotted after **every individual weight update** (not just per epoch) for each gate, including a worked explanation of the periodic weight-cycling pattern that prevents XOR from converging.
15. **Performance Summary Table:** a consolidated table of the final model's configuration and metrics.

Every figure the notebook produces is saved automatically to `figures/<name>.eps` at 600 dpi as it's generated — no manual export step is needed.

---

## 4. How to run it

### 4.1 Prerequisites

- Python 3.10+ (tested with 3.12)
- Jupyter (Notebook, JupyterLab, or VS Code's Jupyter extension)

### 4.2 Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel
```

> If you're using `pip` on a system that requires it, add `--break-system-packages` to the command above.

### 4.3 Get the code and data

```bash
git clone https://github.com/Hemachandhar-A/Deep_Learning.git
cd Deep_Learning          # or the specific experiment folder, if the repo has multiple experiments
```

Make sure `data_banknote_authentication.txt` is in the same folder as `Experiment_1_Perceptron.ipynb` before running — the notebook loads it with a relative path and will raise a `FileNotFoundError` otherwise.

### 4.4 Run the notebook

**Option A — Jupyter Notebook / JupyterLab:**

```bash
jupyter notebook Experiment_1_Perceptron.ipynb
```

Then use **Run → Run All Cells** (or run cells top to bottom with `Shift+Enter`).

**Option B — From the command line (headless execution):**

```bash
jupyter nbconvert --to notebook --execute --inplace Experiment_1_Perceptron.ipynb
```

This re-runs every cell and overwrites the notebook in place with fresh outputs — useful for regenerating all figures and printed results after a code change.

**Option C — VS Code:** open the `.ipynb` file directly and click **Run All** at the top of the notebook.

### 4.5 Expected output

- All intermediate results (dataset summary, training logs, evaluation metrics, comparison tables) print inline as the notebook runs.
- 15 figures are written to `./figures/` as `.eps` files, each named to match its section (e.g. `06_confusion_matrix.eps`, `15_xor_gate_updates.eps`).
- No internet connection is required — everything runs on the local dataset file.

### 4.6 Runtime

The full notebook runs in well under a minute on a standard laptop CPU; there is no GPU dependency anywhere in this experiment.

---

## 5. Reproducibility notes

- A fixed seed (`RANDOM_STATE = 42`) is used for the train/test split and for weight initialization.
- The main training loop shuffles each epoch using a single shared `numpy.random.default_rng` object. Because several models are trained sequentially in one notebook run (main model, learning-rate sweep, decision-boundary model, XOR model, normalization-comparison models), later models consume the RNG stream at a different point than earlier ones. This is called out explicitly in the report's Discussion section — if you need a completely independent, order-invariant comparison between two specific runs, give each one its own freshly seeded `default_rng` instance instead of sharing the global one.
- The logic-gate task (OR/AND/NOT/XOR) uses its own dedicated helper, `train_with_update_history`, seeded independently per gate.

---

## 6. Regenerating the LaTeX report

The `Experiment_1.tex` report embeds the same figures produced by the notebook (converted from `.eps` to `.pdf` for inclusion, since `pdflatex` cannot embed EPS directly). If you regenerate the figures, convert them before recompiling:

```bash
for f in figures/*.eps; do
  gs -q -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress \
     -sOutputFile="tex/figures/$(basename "${f%.eps}").pdf" "$f"
done

pdflatex -interaction=nonstopmode Experiment_1.tex
pdflatex -interaction=nonstopmode Experiment_1.tex   # run twice for stable cross-references
```

Ghostscript (`gs`) is required for the EPS → PDF conversion step.

---

## 7. License / Attribution

- Dataset: UCI Machine Learning Repository — Banknote Authentication (Volkan and Lohweg, donated 2013).
- Code and report: prepared for CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai.
