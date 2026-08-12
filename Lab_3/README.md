# CIFAR-10 CNN Image Classification & Kernel/Architecture Study

**CS3807 – Deep Learning Laboratory · Experiment 3**
Shiv Nadar University Chennai · B.Tech AI & Data Science

A from-scratch-to-Keras implementation of a Convolutional Neural Network for CIFAR-10 image classification, built up from first principles: classical convolution kernels, hand-traceable numerical examples, and systematic studies of kernel size, stride, padding, pooling, activation, and filter count, before training and evaluating the final CNN classifier.

---

## 1. What's in this repository

| File / Folder | Description |
|---|---|
| `Experiment_3_CNN_Image_Classification.ipynb` | Full notebook: classical kernels, worked examples, data pipeline, architecture studies, main CNN training, and full evaluation |
| `Experiment_3_CNN_Image_Classification.tex` / `.pdf` | Full lab report with results, figures, and discussion |
| `figures/` | All 20 generated plots, saved as EPS (produced on notebook execution) |

---

## 2. Dataset

- **Dataset:** CIFAR-10
- **Training images:** 50,000 (further split 90/10 into 45,000 training and 5,000 validation)
- **Testing images:** 10,000
- **Classes (10):** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Image size:** 32×32×3 (RGB)
- **Preprocessing:** pixel values normalized from [0, 255] to [0, 1]

The dataset is loaded directly via `tensorflow.keras.datasets.cifar10`, so no manual download is required.

---

## 3. What the notebook does

The notebook is organized to mirror the lab manual's task structure, plus additional exercises:

1. **Part A — Classical Convolution Kernels:** the ten canonical kernels from the manual (blur, sharpen, emboss, edge-detection variants, etc.) applied to a fixed 512×512 grayscale reference image and clubbed into a single annotated grid (CIFAR-10's 32×32 images are too small to visibly show these effects).
2. **Part B — Worked Numerical Examples:** hand-traceable arithmetic for convolution, max pooling, and parameter counting, matching the manual's formulas.
3. **Part C — Task 1: Dataset Loading and Exploration:** CIFAR-10 load, sample-image grid, dataset dimensions, and class-distribution plot.
4. **Part D — Task 2 & 3: Kernel Size, Stride, and Padding Studies:** a from-scratch convolution layer compared across kernel sizes 3×3, 5×5, 7×7, and across stride ∈ {1, 2} and padding ∈ {valid, same}, with output-size effects visualized.
5. **Part E — CNN Architecture and Parameter Budget:** the main model's layer-by-layer parameter breakdown (`Input → Conv → ReLU → MaxPool → Conv → ReLU → MaxPool → Flatten → Dense → Softmax`).
6. **Part F — Task 4: Feature-Map and Filter Visualization (Untrained):** first- and second-layer feature maps and Conv1 filter weights visualized on the freshly-initialized (untrained) model, to isolate the effect of architecture from the effect of learning.
7. **Part H — Task 6: Main CNN Training:** the main model trained with the Adam optimizer for 20 epochs at batch size 32 (545,098 trainable parameters), with training/validation accuracy and loss curves.
8. **Part I — Task 5: Max Pooling vs. Average Pooling:** an identical architecture retrained with average pooling in place of max pooling, compared side by side.
9. **Part J — Additional Exercise 3: ReLU vs. Sigmoid Activation:** the same architecture retrained with sigmoid activations, demonstrating the vanishing-gradient failure mode against the ReLU baseline.
10. **Part K — Additional Exercise 5: 16 vs. 64 Convolution Filters:** filter-count variants trained and compared against the 32/64-filter main model on accuracy, parameter count, and training time.
11. **Part L — Additional Exercises 1, 2, and 4:** output-size and parameter-count arithmetic worked by hand and checked against the layers actually instantiated.
12. **Part M — Task 7: Full Evaluation of the Main Model:** accuracy, weighted precision/recall/F1, full per-class classification report, confusion matrix, per-class metrics chart, misclassified-example gallery, and trained-model feature maps/filters (Conv1 and Conv2) for direct before/after comparison against Part F.
13. **Part N — Results Summary:** a consolidated table and chart comparing every trained variant (main, average-pool, sigmoid, 16-filter, 64-filter) on train/val/test accuracy, precision, recall, F1, training time, and parameter count.

Every figure the notebook produces is saved automatically to `figures/<name>.eps` as it's generated — no manual export step is needed.

---

## 4. Results Summary

| Model | Train Acc | Val Acc | Test Acc | Precision | Recall | F1 | Time (s) | Params |
|---|---|---|---|---|---|---|---|---|
| Main (Max, ReLU, 32–64f) | 0.9586 | 0.6652 | 0.6584 | 0.6611 | 0.6584 | 0.6582 | 123.8 | 545,098 |
| Avg-pool variant | 0.9310 | 0.6378 | 0.6411 | 0.6532 | 0.6411 | 0.6433 | 114.9 | 545,098 |
| Sigmoid variant | 0.0988 | 0.0946 | 0.1000 | 0.0100 | 0.1000 | 0.0182 | 133.5 | 545,098 |
| 16-filter variant | 0.8767 | 0.6198 | 0.6145 | 0.6093 | 0.6145 | 0.6096 | 111.6 | 135,258 |
| 64-filter variant | 0.9692 | 0.6480 | 0.6633 | 0.6740 | 0.6633 | 0.6651 | 125.2 | 564,426 |

The sigmoid variant's collapse to ~10% accuracy (chance level for 10 classes) is a textbook vanishing-gradient failure — see the report's Discussion section for the explanation. Full per-class metrics, the confusion matrix, and all other plots are in the report.

---

## 5. Requirements

Designed to run on **Google Colab** (GPU runtime recommended) or any local environment with TensorFlow/Keras.

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```

Core stack: `tensorflow`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`.

---

## 6. Running

1. Open `Experiment_3_CNN_Image_Classification.ipynb` in Google Colab (or Jupyter/VS Code locally).
2. Run all cells top to bottom. CIFAR-10 downloads automatically on first run via `keras.datasets.cifar10`.
3. Generated figures are saved to `figures/*.eps` as the notebook executes.

> A GPU runtime is strongly recommended — six separate CNNs are trained in this notebook (main, average-pool, sigmoid, 16-filter, and 64-filter variants, plus the untrained-model visualization pass), each for 20 epochs.

---

## 7. Reproducibility notes

- A fixed random seed is used for weight initialization and data splits; exact metrics can still drift slightly run-to-run due to GPU non-determinism in TensorFlow.
- All five trained variants (main, average-pool, sigmoid, 16-filter, 64-filter) share the same architecture, optimizer (Adam), epoch count (20), and batch size (32) except for the single factor being studied, so the comparison in the Results Summary isolates that factor.
- The untrained-model visualizations in Part F use the same random initialization that seeds the main model's training run in Part H, so the "before" and "after" feature maps in the report are a direct before/after comparison of the same layer, not two independently-initialized models.
- All plots use consistent formatting (per lab formatting requirements) and are exported as vector EPS files for lossless inclusion in the report.

---

## 8. Regenerating the LaTeX report

The `Experiment_3_CNN_Image_Classification.tex` report embeds the same figures produced by the notebook (converted from `.eps` to `.pdf` for inclusion, since `pdflatex` cannot embed EPS directly). If you regenerate the figures, convert them before recompiling:

```bash
for f in figures/*.eps; do
  gs -q -dBATCH -dNOPAUSE -dEPSCrop -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress \
     -sOutputFile="tex/figures/$(basename "${f%.eps}").pdf" "$f"
done

pdflatex -interaction=nonstopmode Experiment_3_CNN_Image_Classification.tex
pdflatex -interaction=nonstopmode Experiment_3_CNN_Image_Classification.tex   # run twice for stable cross-references
```

Ghostscript (`gs`) is required for the EPS → PDF conversion step. Some of the raw feature-map EPS exports are very large (tens to hundreds of MB, since matplotlib embeds each pixel of every feature map as vector paths); the `/prepress` Ghostscript setting keeps the resulting PDFs compact without any visible quality loss.

---

## 9. License / Attribution

- Dataset: CIFAR-10 (Krizhevsky, 2009), distributed via `tensorflow.keras.datasets`.
- Code and report: prepared for CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai.
