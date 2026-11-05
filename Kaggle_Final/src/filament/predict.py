"""Whole-image prediction with optional flips (Arch section 2 inference pipeline, decision D10, A4.8 item 1).

Used by ``train.py`` (held-out maps of a fold, no flips) and by ``infer.py`` (test images, the configured flip set).

Flips are of the *same* single image (D10): the network sees a mirrored copy, its sigmoid map is mirrored back, and the
maps of all flips are averaged.  This uses only the image itself, which the host requires (S6).  A flip is written as a
string: ``""`` identity, ``"h"`` mirror left-right, ``"v"`` mirror up-down, ``"hv"`` both.

NaN guard (finding N1, review 17 4.2 point 3): under mixed precision one trained model produced NaN logits on a patch of an image
(a half-precision overflow inside the network; float32 has none).  A forward pass whose logits are not all finite is now run
again without autocast (float32) and the float32 logits are used; the number of such passes and of non-finite logits that are
still left after the retry are counted (``guard_counts()``, and the optional ``stats`` argument of ``predict_probabilities``).
When every logit is finite, which is the normal case, the guard does nothing and the output is byte for byte what it was before.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from .data import normalise

VALID_FLIPS = ("", "h", "v", "hv")

# Cumulative counts of the NaN guard since the process started (or since ``reset_guard_counts``): ``passes`` = forward passes
# checked (one per batch and flip), ``recomputed`` = passes whose autocast logits were not all finite and that were run again in
# float32, ``nonfinite_left`` = non-finite logit values still present after that (0 when float32 repairs the batch).
_GUARD_KEYS = ("passes", "recomputed", "nonfinite_left")
_GUARD = {k: 0 for k in _GUARD_KEYS}


def guard_counts() -> dict:
    """A copy of the cumulative guard counts (``passes``, ``recomputed``, ``nonfinite_left``)."""
    return dict(_GUARD)


def reset_guard_counts() -> None:
    for k in _GUARD_KEYS:
        _GUARD[k] = 0


def _flip(x: torch.Tensor, how: str) -> torch.Tensor:
    dims = []
    if "h" in how:
        dims.append(-1)  # width
    if "v" in how:
        dims.append(-2)  # height
    return torch.flip(x, dims) if dims else x


def _guarded_logits(model, xt, how: str, use_amp: bool, dev_type: str, stats: dict | None):
    """Logits of one (possibly flipped) batch; non-finite autocast logits are recomputed in float32 (see the module docstring)."""
    with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
        logits = model(_flip(xt, how))
    recomputed, left = 0, 0
    if not bool(torch.isfinite(logits).all()):
        if use_amp:
            recomputed = 1
            with torch.autocast(device_type=dev_type, enabled=False):
                logits = model(_flip(xt, how))
        left = int((~torch.isfinite(logits)).sum())
        warnings.warn(
            "predict_probabilities: non-finite logits%s; %d non-finite logit value(s) left" % (
                " under autocast, the pass was recomputed in float32" if recomputed else " (autocast is off, nothing to recompute)", left))
    for d in (_GUARD, stats):
        if d is not None:
            d["passes"] = d.get("passes", 0) + 1
            d["recomputed"] = d.get("recomputed", 0) + recomputed
            d["nonfinite_left"] = d.get("nonfinite_left", 0) + left
    return logits


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    images_u8,
    mean: float,
    std: float,
    device: str,
    amp: bool = True,
    batch_size: int = 4,
    flips: tuple[str, ...] | list[str] = ("",),
    out_dtype=np.float16,
    stats: dict | None = None,
) -> np.ndarray:
    """Sigmoid maps ``(N, H, W)`` at the resolution of ``images_u8`` (the arm's resolution), averaged over ``flips``.

    Averaging is done on probabilities in float32; the result is cast to ``out_dtype`` (float16 is what is stored).
    ``stats``, if given, is a dict to which this call adds the NaN-guard counts (``passes``, ``recomputed``, ``nonfinite_left``).
    """
    bad = [f for f in flips if f not in VALID_FLIPS]
    if bad:
        raise ValueError("unknown flips %s; use %s" % (bad, VALID_FLIPS))
    if len(flips) == 0:
        raise ValueError("at least one flip (use '' for the identity)")
    model.eval()
    use_amp = bool(amp and device.startswith("cuda"))
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    out = []
    for i in range(0, len(images_u8), batch_size):
        x = np.stack([normalise(np.asarray(a), mean, std) for a in images_u8[i : i + batch_size]])[:, None]
        xt = torch.from_numpy(x).to(device)
        acc = None
        for how in flips:
            logits = _guarded_logits(model, xt, how, use_amp, dev_type, stats)
            p = _flip(torch.sigmoid(logits.float()), how)  # a flip is its own inverse
            acc = p if acc is None else acc + p
        out.append((acc / len(flips))[:, 0].cpu().numpy().astype(out_dtype))
    return np.concatenate(out) if out else np.zeros((0,) + tuple(np.asarray(images_u8).shape[1:]), out_dtype)
