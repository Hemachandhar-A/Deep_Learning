"""Solar filament segmentation pipeline (GONG H-alpha, MAGFiLO / Kaggle 2026).

Module map (document IDs in each module docstring refer to the team's
"Spec & Context" (S) and "Architecture & Pipeline Decisions" (A) documents):

    rle.py         RLE encoding, submission CSV writer and checker      (S4)
    pq_metric.py   replica of the host's Panoptic Quality scorer         (S4, A4.5)
    data.py        annotation parsing, rasterisation, cache, datasets    (A3, A4.8)
    folds.py       grouped, stratified 5-fold split                      (A4.1)
    disc.py        solar-disc detection recipe                           (A3)
    postprocess.py probability map -> disjoint filament masks            (A5.2)
    losses.py      BCE/focal + Dice/Tversky                              (A4.2)
    model.py       smp model factory                                     (A2)
    config.py      YAML/JSON config loading                              (A4.8)
"""

__version__ = "0.1.0"
