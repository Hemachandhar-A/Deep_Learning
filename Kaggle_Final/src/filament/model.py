"""Model factory (Arch section 2, decision D1, experiment E7).

Start here: ``smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, classes=1)``, 24.4 M
parameters, one logit map out.  Architecture, encoder and weights are configuration, so that the E7 arms need no code.

Only SMP's CNN encoders are used (K5): no SegFormer / MiT encoders (non-commercial source licence).  The ImageNet
weights are downloaded by SMP 0.5.0 from the Hugging Face repository ``smp-hub/resnet34.imagenet`` (original weights:
download.pytorch.org/models/resnet34-333f7ec4.pth); that repository's licence field is "other" (A K5), which is why the
source and the revision are named in the README and recorded in every run's metrics.

One input channel: SMP sums the first convolution's pretrained weights over the three RGB channels, which equals a
convolution over three identical gray channels (A2, derived); tests/test_model_loss.py verifies this on the real weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

FORBIDDEN_ENCODER_PREFIXES = ("mit_b", "segformer")  # NVIDIA non-commercial source licence (A1, D12, K5)
FORBIDDEN_ARCHS = ("Segformer",)  # SMP ships it; parked by D12


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "Unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"  # None = random initialisation (E7 arm)
    in_channels: int = 1
    classes: int = 1

    @classmethod
    def from_dict(cls, d: dict | None) -> "ModelConfig":
        return cls(**(d or {}))


def build_model(cfg: ModelConfig = ModelConfig()) -> torch.nn.Module:
    """Create the network.  The output is a logit map of the same height and width as the input."""
    import segmentation_models_pytorch as smp  # not preinstalled on Kaggle: pip install segmentation-models-pytorch==0.5.0

    if cfg.encoder.lower().startswith(FORBIDDEN_ENCODER_PREFIXES):
        raise ValueError("encoder %r is excluded by the project rules (licence, A K5)" % cfg.encoder)
    if cfg.arch in FORBIDDEN_ARCHS:
        raise ValueError("architecture %r is parked by decision D12" % cfg.arch)
    if not hasattr(smp, cfg.arch):
        raise ValueError("segmentation_models_pytorch has no architecture %r" % cfg.arch)
    return getattr(smp, cfg.arch)(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pretrained_weights_info(encoder: str = "resnet34", weights: str = "imagenet") -> dict | None:
    """Hugging Face repository and *pinned revision* that SMP loads for ``encoder`` / ``weights`` (for metrics, K5).

    SMP 0.5.0 hard-codes both in ``encoders/resnet.py`` (resnet34 imagenet: ``smp-hub/resnet34.imagenet`` at revision
    7a57b34f723329ff020b3f8bc41771163c519d0c).  Returns None for random initialisation or an unknown pair.
    """
    if weights is None:
        return None
    try:
        import segmentation_models_pytorch as smp

        return dict(smp.encoders.encoders[encoder]["pretrained_settings"][weights])
    except Exception:  # a different SMP version may lay this out differently; never break a run for a log field
        return None


def pretrained_weights_revision(encoder: str = "resnet34", weights: str = "imagenet") -> str | None:
    info = pretrained_weights_info(encoder, weights)
    return info.get("revision") if info else None
