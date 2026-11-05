"""Host scoring functions, transcribed from the public Kaggle notebook
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook  (Version 6 of 6, Apache 2.0 licence as stated on its page).

Only the scoring cells (In [1] imports, In [3] helpers, In [4] get_pq_score) are copied; docstrings shortened, logic unchanged.
Plot functions are left out.  Read on 2026-10-12 from the notebook's rendered page.

Copyright and licence (added for the public repository; Gate 1b review B6, Q11): the notebook page carries no
copyright line, neither in its body nor in its metadata (checked on 2026-10-13); its metadata states "This Notebook
has been released under the Apache 2.0 open source license."  Licence text: third_party/LICENSE-Apache-2.0.txt;
see also the NOTICE file of this repository.

Changed by the authors of this repository: transcribed from the rendered page, docstrings shortened, plotting
functions omitted, helper `get_counts` added; the logic of the scoring functions is unchanged.
"""
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util


def get_overlap_matrices(gt_layers, pred_layers):
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        intersection / union,
    )
    dice_matrix = torch.where(
        union == 0,
        torch.tensor(0., device=gt_layers.device),
        2 * intersection / (gt_areas + pred_areas),
    )
    return iou_matrix, dice_matrix


def fp_count_hit(hit_matrix):
    gt_matches_per_pred = hit_matrix.sum(dim=0)
    return (gt_matches_per_pred == 0).sum().item()


def fn_count_hit(hit_matrix):
    pred_matches_per_gt = hit_matrix.sum(dim=1)
    return (pred_matches_per_gt == 0).sum().item()


def rles_to_layers(rles, height=2048, width=2048):
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def process_entry(annotator_image, gt_annotator_image_ids, pred_image_ids, gt_df, pred_df):
    image_id = annotator_image.split("-", maxsplit=1)[1]
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
    n_gt = len(gt_rles)
    n_pred = len(pred_rles)
    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
    return iou_matrix, dice_matrix, n_gt, n_pred


def get_overlap_df(gt_df, pred_df):
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    annotator_images = gt_annotator_image_ids.unique()
    overlap_df = pd.DataFrame({"annotator_image": annotator_images})
    results = overlap_df["annotator_image"].apply(
        process_entry,
        gt_annotator_image_ids=gt_annotator_image_ids,
        pred_image_ids=pred_image_ids,
        gt_df=gt_df,
        pred_df=pred_df,
    )
    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df["n_pred"] = zip(*results)
    return overlap_df


def get_pq_score(overlap_df):
    iou_threshold = 0.5
    tp_iou_scores = []
    fp_count = 0
    fn_count = 0
    for row in overlap_df.itertuples(index=False):
        iou_matrix = row.iou_matrix
        n_gt = row.n_gt
        n_pred = row.n_pred
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > iou_threshold
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += fp_count_hit(hit_matrix)
        fn_count += fn_count_hit(hit_matrix)
    tp_count = len(tp_iou_scores)
    denominator = tp_count + 0.5 * fp_count + 0.5 * fn_count
    if denominator > 0:
        pq_score = sum(tp_iou_scores) / denominator
    else:
        pq_score = 0.0
    return pq_score


def get_counts(overlap_df):
    """Not in the notebook: same loop as get_pq_score, returning (sum of TP IoUs, TP, FP, FN) for comparison."""
    tp = []
    fp = fn = 0
    for row in overlap_df.itertuples(index=False):
        if row.n_gt == 0:
            fp += row.n_pred
            continue
        if row.n_pred == 0:
            fn += row.n_gt
            continue
        hit = row.iou_matrix > 0.5
        tp.extend(row.iou_matrix[hit].tolist())
        fp += fp_count_hit(hit)
        fn += fn_count_hit(hit)
    return sum(tp), len(tp), fp, fn
