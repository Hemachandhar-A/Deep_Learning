"""Acceptance test 10 (E0): model and loss (Arch sections 1, 2 and 4.2)."""
import numpy as np
import pytest
import torch

from filament import losses, model


# ---------------------------------------------------------------- losses (hand-checkable numbers)
def test_soft_dice_matches_the_formula_of_arch_4_2():
    logits = torch.tensor([[[[2.0, -2.0], [0.0, 1.0]]]])
    y = torch.tensor([[[[1.0, 0.0], [0.5, 1 / 3]]]])
    p = torch.sigmoid(logits)
    expect = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    assert losses.soft_dice_loss(logits, y).item() == pytest.approx(expect.item(), abs=1e-7)


def test_soft_dice_is_summed_over_the_batch_not_averaged_per_image():
    # one image perfect and large, one image empty with a false positive: batch sum != mean of per-image Dice
    big = torch.full((1, 1, 8, 8), 20.0)  # p ~ 1 everywhere
    y_big = torch.ones(1, 1, 8, 8)
    small_z = torch.full((1, 1, 8, 8), -20.0)
    small_z[0, 0, 0, 0] = 20.0
    y_small = torch.zeros(1, 1, 8, 8)
    both = losses.soft_dice_loss(torch.cat([big, small_z]), torch.cat([y_big, y_small])).item()
    per_image = np.mean([losses.soft_dice_loss(big, y_big).item(), losses.soft_dice_loss(small_z, y_small).item()])
    assert both < 0.01 < per_image  # the batch-summed value is dominated by the large perfect image


def test_perfect_and_worst_predictions():
    y = torch.zeros(1, 1, 8, 8)
    y[0, 0, 2:5, 2:5] = 1.0
    good = (y * 2 - 1) * 15.0
    assert losses.soft_dice_loss(good, y).item() < 1e-3
    assert losses.bce_loss(good, y).item() < 1e-5
    assert losses.soft_dice_loss(-good, y).item() > 0.8


def test_bce_uses_soft_targets():
    z = torch.zeros(1, 1, 2, 2)  # p = 0.5
    y = torch.tensor([[[[0.0, 1.0], [0.5, 1 / 3]]]])
    assert losses.bce_loss(z, y).item() == pytest.approx(np.log(2.0), abs=1e-6)  # BCE(0.5, any y) = ln 2


def test_tversky_with_neutral_weights_equals_dice_without_smoothing():
    torch.manual_seed(0)
    z = torch.randn(2, 1, 16, 16)
    y = torch.rand(2, 1, 16, 16)
    d = losses.soft_dice_loss(z, y, smooth=0.0)
    t = losses.tversky_loss(z, y, alpha=0.5, beta=0.5, smooth=0.0)
    assert t.item() == pytest.approx(d.item(), abs=1e-6)
    # beta > alpha punishes missed filament pixels more
    fn_heavy = losses.tversky_loss(torch.full((1, 1, 4, 4), -20.0), torch.ones(1, 1, 4, 4), alpha=0.3, beta=0.7, smooth=0.0)
    assert fn_heavy.item() == pytest.approx(1.0, abs=1e-5)


def test_focal_reduces_to_bce_at_gamma_zero_and_downweights_easy_pixels():
    torch.manual_seed(0)
    z, y = torch.randn(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    assert losses.focal_bce_loss(z, y, 0.0).item() == pytest.approx(losses.bce_loss(z, y).item(), abs=1e-6)
    assert losses.focal_bce_loss(z, y, 2.0).item() < losses.bce_loss(z, y).item()


def test_combined_loss_weights_and_validation():
    torch.manual_seed(0)
    z, y = torch.randn(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
    total, parts = losses.CombinedLoss(losses.LossConfig(w_bce=1.0, w_region=1.0))(z, y)
    assert total.item() == pytest.approx((parts["bce"] + parts["region"]).item(), abs=1e-6)
    total2, _ = losses.CombinedLoss(losses.LossConfig(w_bce=2.0, w_region=0.0))(z, y)
    assert total2.item() == pytest.approx(2 * parts["bce"].item(), abs=1e-6)
    with pytest.raises(ValueError):
        losses.CombinedLoss(losses.LossConfig(region="nope"))


def test_loss_is_float32_under_autocast():
    z = torch.randn(1, 1, 8, 8).half()
    y = torch.rand(1, 1, 8, 8)
    total, _ = losses.CombinedLoss()(z, y)
    assert total.dtype == torch.float32


# ---------------------------------------------------------------- model
def test_unet_resnet34_has_24_4_million_parameters():
    m = model.build_model(model.ModelConfig(encoder_weights=None))
    n = model.count_parameters(m)
    print("\n[E0-10] smp.Unet(resnet34, in_channels=1, classes=1): %.3f M parameters (document: 24.4 M)" % (n / 1e6))
    assert abs(n / 1e6 - 24.4) < 0.05


def test_forward_backward_and_loss_decreases_when_fitting_8_synthetic_images():
    torch.manual_seed(0)
    m = model.build_model(model.ModelConfig(encoder_weights=None))
    x = torch.randn(8, 1, 64, 64)
    y = torch.zeros(8, 1, 64, 64)
    for i in range(8):
        y[i, 0, 8 + 3 * i : 20 + 3 * i, 10:50] = 1.0 if i % 2 == 0 else 0.5
    out = m(x)
    assert out.shape == (8, 1, 64, 64)
    loss_fn = losses.CombinedLoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    history = []
    for _ in range(25):
        opt.zero_grad()
        loss, _ = loss_fn(m(x), y)
        loss.backward()
        opt.step()
        history.append(loss.item())
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    print("[E0-10] synthetic 8-image fit: loss %.3f -> %.3f in 25 steps" % (history[0], history[-1]))
    assert history[-1] < 0.6 * history[0]


def test_forbidden_encoders_and_unknown_architectures_are_rejected():
    with pytest.raises(ValueError):
        model.build_model(model.ModelConfig(encoder="mit_b0", encoder_weights=None))
    with pytest.raises(ValueError):  # smp ships it, but D12 parks it
        model.build_model(model.ModelConfig(arch="Segformer", encoder="resnet34", encoder_weights=None))
    with pytest.raises(ValueError):
        model.build_model(model.ModelConfig(arch="NoSuchNet", encoder="resnet34", encoder_weights=None))


@pytest.mark.net
def test_imagenet_weights_load_and_one_channel_equals_three_identical_channels():
    """The ImageNet weights load; 1 gray channel equals 3 identical channels through the encoder (A2, derived claim)."""
    m1 = model.build_model(model.ModelConfig(encoder_weights="imagenet", in_channels=1)).eval()
    m3 = model.build_model(model.ModelConfig(encoder_weights="imagenet", in_channels=3)).eval()
    w1 = m1.encoder.conv1.weight.detach()
    w3 = m3.encoder.conv1.weight.detach()
    assert w1.shape == (64, 1, 7, 7) and w3.shape == (64, 3, 7, 7)
    assert torch.allclose(w1, w3.sum(dim=1, keepdim=True), atol=1e-6)  # SMP sums the pretrained conv1 weights
    # the rest of the encoder is identical, so the features agree
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        f1 = m1.encoder(x)
        f3 = m3.encoder(x.repeat(1, 3, 1, 1))
    err = max((a - b).abs().max().item() for a, b in zip(f1, f3))
    print("\n[E0-10] ImageNet weights loaded; conv1 = sum of pretrained RGB weights; "
          "max feature difference 1-ch vs 3 identical channels: %.2e" % err)
    assert err < 1e-4
    assert model.count_parameters(m1) == model.count_parameters(
        model.build_model(model.ModelConfig(encoder_weights=None)))
    info = model.pretrained_weights_info("resnet34", "imagenet")
    print("[E0-10] weights SMP loads: %s at pinned revision %s" % (info["repo_id"], info["revision"]))
    assert info == {"repo_id": "smp-hub/resnet34.imagenet", "revision": "7a57b34f723329ff020b3f8bc41771163c519d0c"}
    assert model.pretrained_weights_info("resnet34", None) is None
