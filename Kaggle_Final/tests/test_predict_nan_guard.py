"""The NaN guard of predict.py (finding N1): bit-identity on clean input, and a forced non-finite pass that is repaired and counted."""
import numpy as np
import pytest
import torch

from filament import predict
from filament.data import normalise


class Conv(torch.nn.Module):
    """A small fixed network (no randomness after construction)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn(self.conv.weight.shape, generator=g))
            self.conv.bias.copy_(torch.randn(self.conv.bias.shape, generator=g))

    def forward(self, x):
        return self.conv(x) * 3.0


class Flaky(Conv):
    """NaN on a 3 x 3 corner patch whenever autocast is on for the input's device type; finite (and equal to Conv) otherwise: the shape
    of finding N1 (an overflow in half precision that float32 does not have)."""

    def forward(self, x):
        y = super().forward(x)
        if torch.is_autocast_enabled(x.device.type):
            y = y.clone()
            y[..., :3, :3] = float("nan")
        return y


class AlwaysNaN(Conv):
    """Non-finite in float32 as well: nothing can repair it, the guard must only count it."""

    def forward(self, x):
        y = super().forward(x).clone()
        y[..., :2, :2] = float("nan")
        return y


def reference_predict(model, images_u8, mean, std, device, amp=True, batch_size=4, flips=("",), out_dtype=np.float16):
    """The body of predict_probabilities as it was before the guard (copied from 188284c), for the bit-identity tests."""
    model.eval()
    use_amp = bool(amp and device.startswith("cuda"))
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    out = []
    with torch.no_grad():
        for i in range(0, len(images_u8), batch_size):
            x = np.stack([normalise(np.asarray(a), mean, std) for a in images_u8[i : i + batch_size]])[:, None]
            xt = torch.from_numpy(x).to(device)
            acc = None
            for how in flips:
                with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp):
                    logits = model(predict._flip(xt, how))
                p = predict._flip(torch.sigmoid(logits.float()), how)
                acc = p if acc is None else acc + p
            out.append((acc / len(flips))[:, 0].cpu().numpy().astype(out_dtype))
    return np.concatenate(out)


def _images(n=6, side=24, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (n, side, side), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _fresh_counts():
    predict.reset_guard_counts()
    yield
    predict.reset_guard_counts()


@pytest.mark.parametrize("flips", [("",), ("", "h", "v", "hv")])
@pytest.mark.parametrize("out_dtype", [np.float16, np.float32])
def test_clean_input_gives_byte_identical_output_on_cpu(flips, out_dtype):
    model, imgs = Conv(), _images()
    stats = {}
    new = predict.predict_probabilities(model, imgs, 0.3, 0.2, "cpu", amp=True, batch_size=4, flips=flips, out_dtype=out_dtype, stats=stats)
    old = reference_predict(model, imgs, 0.3, 0.2, "cpu", amp=True, batch_size=4, flips=flips, out_dtype=out_dtype)
    assert new.dtype == old.dtype and new.shape == old.shape
    assert new.tobytes() == old.tobytes()  # byte for byte
    n_passes = len(flips) * 2  # six images in batches of four: two batches
    assert stats == {"passes": n_passes, "recomputed": 0, "nonfinite_left": 0}
    assert predict.guard_counts() == stats


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("flips", [("",), ("", "h", "v", "hv")])
def test_clean_input_gives_byte_identical_output_with_autocast_on_cuda(flips):
    model, imgs = Conv().to("cuda"), _images(n=9, side=32)
    stats = {}
    new = predict.predict_probabilities(model, imgs, 0.3, 0.2, "cuda", amp=True, batch_size=4, flips=flips, stats=stats)
    old = reference_predict(model, imgs, 0.3, 0.2, "cuda", amp=True, batch_size=4, flips=flips)
    assert new.tobytes() == old.tobytes()
    assert stats["recomputed"] == 0 and stats["nonfinite_left"] == 0 and stats["passes"] == len(flips) * 3


def test_forced_nonfinite_pass_is_recomputed_in_float32_and_counted():
    """_guarded_logits with autocast on (CPU autocast, float16): the Flaky network gives NaN under autocast, finite without."""
    model, x = Flaky(), torch.randn(2, 1, 16, 16)
    stats = {}
    with pytest.warns(UserWarning, match="recomputed in float32"):
        logits = predict._guarded_logits(model, x, "", True, "cpu", stats)
    assert torch.isfinite(logits).all()
    assert logits.dtype == torch.float32
    assert torch.equal(logits, Conv.forward(model, x))  # the float32 result, bit for bit
    assert stats == {"passes": 1, "recomputed": 1, "nonfinite_left": 0}
    assert predict.guard_counts() == stats


def test_a_clean_pass_under_autocast_is_not_recomputed():
    model, x = Conv(), torch.randn(2, 1, 16, 16)
    stats = {}
    logits = predict._guarded_logits(model, x, "h", True, "cpu", stats)
    with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True):
        ref = model(predict._flip(x, "h"))
    assert logits.dtype == ref.dtype and torch.equal(logits, ref)
    assert stats == {"passes": 1, "recomputed": 0, "nonfinite_left": 0}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_forced_nan_through_predict_probabilities_on_cuda_is_repaired():
    imgs = _images(n=5, side=32)
    flaky, clean = Flaky().to("cuda"), Conv().to("cuda")
    stats = {}
    with pytest.warns(UserWarning):
        out = predict.predict_probabilities(flaky, imgs, 0.3, 0.2, "cuda", amp=True, batch_size=4, flips=("", "h"), stats=stats)
    assert np.isfinite(out).all()
    ref = reference_predict(clean, imgs, 0.3, 0.2, "cuda", amp=False, batch_size=4, flips=("", "h"))  # float32 path of the same weights
    assert out.tobytes() == ref.tobytes()
    assert stats == {"passes": 4, "recomputed": 4, "nonfinite_left": 0}  # two batches x two flips, all recomputed


def test_nonfinite_in_float32_is_counted_not_hidden():
    """When float32 is non-finite too (AlwaysNaN, amp off on CPU) there is nothing to recompute: the NaN pixels stay and are counted."""
    model, imgs = AlwaysNaN(), _images(n=4, side=16)
    stats = {}
    with pytest.warns(UserWarning, match="nothing to recompute"):
        out = predict.predict_probabilities(model, imgs, 0.3, 0.2, "cpu", amp=True, batch_size=4, stats=stats)
    assert stats == {"passes": 1, "recomputed": 0, "nonfinite_left": 4 * 2 * 2}  # 4 images x a 2 x 2 patch
    assert int(np.isnan(out).sum()) == 4 * 2 * 2  # untouched: the maps still carry them, and the count says so


def test_guard_counts_accumulate_and_reset():
    model, imgs = Conv(), _images(n=4, side=16)
    predict.predict_probabilities(model, imgs, 0.3, 0.2, "cpu", batch_size=4)
    predict.predict_probabilities(model, imgs, 0.3, 0.2, "cpu", batch_size=2)
    assert predict.guard_counts() == {"passes": 3, "recomputed": 0, "nonfinite_left": 0}
    predict.reset_guard_counts()
    assert predict.guard_counts() == {"passes": 0, "recomputed": 0, "nonfinite_left": 0}
