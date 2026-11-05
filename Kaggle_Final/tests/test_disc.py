"""Acceptance test 8 (E0): disc recipe (Arch section 3)."""
from collections import Counter

import numpy as np
import pytest

from filament import disc


def _synthetic_sun(side=2048, cx=1010.0, cy=1030.0, R=890.0, limb=True, halo=True, seed=0):
    """Disc with limb darkening, a faint outer halo (what Otsu alone picks up), noise and a stray bright blob."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side].astype(np.float32)
    r = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy) / R
    img = np.full((side, side), 4.0, np.float32)
    if halo:
        img[r < 1.04] = 60.0  # dim ring just outside the limb
    inside = r < 1.0
    prof = 140.0 - (60.0 * np.clip(r, 0, 1) ** 4 if limb else 0.0)
    img[inside] = prof[inside]
    img += rng.normal(0, 3.0, img.shape)
    img[40:90, 40:90] = 200.0  # a bright artefact in the corner, disconnected from the disc
    return np.clip(img, 0, 255).astype(np.uint8)


def test_fit_disc_recovers_centre_and_radius():
    for cx, cy, R in [(1010.0, 1030.0, 890.0), (1024.0, 1024.0, 930.0), (1000.0, 1060.0, 870.0)]:
        d = disc.fit_disc(_synthetic_sun(cx=cx, cy=cy, R=R))
        assert abs(d.cx - cx) < 4 and abs(d.cy - cy) < 4  # one 512-px pixel = 4 px on the 2048 grid
        # the 0.65 x median rule puts the edge where limb-darkened intensity falls below it, close to the limb
        assert abs(d.radius - R) < 10, (d, R)


def test_artefact_and_halo_do_not_move_the_fit():
    clean = disc.fit_disc(_synthetic_sun(halo=False))
    noisy = disc.fit_disc(_synthetic_sun(halo=True))
    assert abs(clean.cx - noisy.cx) < 2 and abs(clean.radius - noisy.radius) < 4


def test_radial_map_and_scaling():
    d = disc.Disc(cx=10.0, cy=20.0, radius=5.0)
    r = disc.radial_map((40, 40), d)
    assert r.shape == (40, 40) and r.dtype == np.float32
    assert r[20, 10] == pytest.approx(np.hypot(0.5, 0.5) / 5.0, abs=1e-6)  # pixel centres are at +0.5
    s = disc.scale_disc(d, 2.0)
    assert (s.cx, s.cy, s.radius) == (20.0, 40.0, 10.0)


def test_input_validation():
    with pytest.raises(ValueError):
        disc.fit_disc(np.zeros((2048, 2048), np.uint8))  # no disc at all
    with pytest.raises(ValueError):
        disc.fit_disc(np.zeros((100, 100), np.uint8))
    rgb = disc.draw_overlay(_synthetic_sun(side=512), disc.Disc(256, 256, 200), out_side=256)
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8


@pytest.mark.data
def test_radius_statistics_over_training_images(layout, annotations):
    """Arch section 3: bright-disc radius over the 707 training images has median 891.4 px (p5 880.2, p95 933.6)."""
    from filament import data

    radii, centre_off = {}, {}
    for name in annotations.image_names:
        d = disc.fit_disc(data.read_gray(data.image_path(layout.train_images, name)))
        radii[name] = d.radius
        centre_off[name] = (d.cx - 1024.0, d.cy - 1024.0)
    r = np.array(list(radii.values()))
    med, p5, p95 = np.median(r), *np.percentile(r, [5, 95])
    print("\n[E0-8] disc radius over %d images: median %.1f  p5 %.1f  p95 %.1f  (document: 891.4, 880.2, 933.6)"
          % (len(r), med, p5, p95))
    print("  per instrument: median radius | std | median centre offset from image centre (x, y)")
    for s in data.INSTRUMENT_CODES:
        names = [n for n in radii if data.instrument_of(n) == s]
        rr = np.array([radii[n] for n in names])
        off = np.array([centre_off[n] for n in names])
        print("  %s n=%3d | %6.1f | std %5.1f | (%+.1f, %+.1f)" % (s, len(rr), np.median(rr), rr.std(), *np.median(off, axis=0)))
    over = {t: sorted(n for n, v in radii.items() if v > t) for t in (920, 940)}
    print("[E0-8] fits with radius > 920 px: %d images; > 940 px: %d images (by instrument: %s); maximum %.1f px"
          % (len(over[920]), len(over[940]), dict(sorted(Counter(data.instrument_of(n) for n in over[940]).items())),
             r.max()))
    print("[E0-8] the bulk is 880-900 px; the large radii are fits that follow a scattered-light halo "
          "(see OPEN_QUESTIONS Q1 and the overlays); the recipe is NOT changed")
    assert abs(med - 891.4) < 0.15 and abs(p5 - 880.2) < 0.3 and abs(p95 - 933.6) < 0.3
