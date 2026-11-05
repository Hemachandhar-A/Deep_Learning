"""Losses (Arch section 4.2, decision D5; E4 arms).

Default: BCE with logits against the soft target + soft Dice, weights 1 : 1.

    soft Dice = 1 - (2 sum(p y) + 1) / (sum(p) + sum(y) + 1),   sums over the whole batch (A4.2)

``p`` is the sigmoid of the logits, ``y`` the soft target in [0, 1] (the mean of the annotator versions, P7).  Why Dice
at all: annotated pixels are a median 0.30% of a version's pixels (S5), so plain BCE is dominated by background.

Switchable options for experiment E4 (not defaults, constants below are plain untested choices, see DECISIONS.md):

* ``bce="focal"``     BCE multiplied by ``|y - p| ** gamma`` (the usual focal modulation, written for soft targets);
* ``region="tversky"`` Tversky index with weights ``alpha`` on false positives and ``beta`` on false negatives;
  alpha = beta = 0.5 is the neutral setting (no direction assumed, D4).

All reductions are done in float32 so that mixed precision does not change the loss arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    bce: str = "bce"  # "bce" | "focal"
    region: str = "dice"  # "dice" | "tversky"
    w_bce: float = 1.0
    w_region: float = 1.0
    smooth: float = 1.0  # A4.2: "the smoothing constant 1 is a plain choice, untested"
    focal_gamma: float = 2.0  # open choice, E4 only
    tversky_alpha: float = 0.5  # weight of false positives; open choice, E4 only
    tversky_beta: float = 0.5  # weight of false negatives; open choice, E4 only

    @classmethod
    def from_dict(cls, d: dict | None) -> "LossConfig":
        return cls(**(d or {}))


def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy with logits against a soft target."""
    return F.binary_cross_entropy_with_logits(logits.float(), target.float())


def focal_bce_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    z, y = logits.float(), target.float()
    ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    return ((y - torch.sigmoid(z)).abs().pow(gamma) * ce).mean()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """1 - (2 sum(p y) + s) / (sum(p) + sum(y) + s), summed over the whole batch (A4.2)."""
    p, y = torch.sigmoid(logits.float()), target.float()
    return 1.0 - (2.0 * (p * y).sum() + smooth) / (p.sum() + y.sum() + smooth)


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, beta: float = 0.5,
                 smooth: float = 1.0) -> torch.Tensor:
    p, y = torch.sigmoid(logits.float()), target.float()
    tp = (p * y).sum()
    fp = (p * (1.0 - y)).sum()
    fn = ((1.0 - p) * y).sum()
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


class CombinedLoss(torch.nn.Module):
    """``w_bce * (BCE or focal) + w_region * (soft Dice or Tversky)``; ``forward`` returns (total, parts)."""

    def __init__(self, cfg: LossConfig = LossConfig()):
        super().__init__()
        if cfg.bce not in ("bce", "focal"):
            raise ValueError("bce must be 'bce' or 'focal', got %r" % cfg.bce)
        if cfg.region not in ("dice", "tversky"):
            raise ValueError("region must be 'dice' or 'tversky', got %r" % cfg.region)
        self.cfg = cfg

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        c = self.cfg
        a = bce_loss(logits, target) if c.bce == "bce" else focal_bce_loss(logits, target, c.focal_gamma)
        if c.region == "dice":
            b = soft_dice_loss(logits, target, c.smooth)
        else:
            b = tversky_loss(logits, target, c.tversky_alpha, c.tversky_beta, c.smooth)
        total = c.w_bce * a + c.w_region * b
        return total, {"bce": a.detach(), "region": b.detach()}
