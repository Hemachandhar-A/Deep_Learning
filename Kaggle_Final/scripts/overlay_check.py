"""Write PNG overlays of the fitted solar disc for a visual check (Arch section 3; acceptance test 8).

    python scripts/overlay_check.py --data-root <folder> --out <folder for PNGs> [--per-instrument 10]

The check has not been done yet (A3): Bh and Th have larger radius spread (27.6 and 30.9 px) than the other sites.  For
each of Bh and Th this picks the images with the largest radius (where a mis-fit is most likely) and a seeded random
sample, writes one PNG per image (green circle = fitted bright disc, red cross = centre) and a contact sheet.
Overlays are derived from competition images: keep them out of the public repository.
"""
import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import cv2
import numpy as np

from filament import data, disc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-instrument", type=int, default=10)
    ap.add_argument("--instruments", nargs="+", default=["Bh", "Th"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    layout = data.find_data_layout(args.data_root)
    names = data.list_image_names(layout.train_images)
    rng = np.random.default_rng(args.seed)
    tiles, lines = [], []
    n_big = args.per_instrument // 2
    for inst in args.instruments:
        pool = [n for n in names if data.instrument_of(n) == inst]
        fits = {n: disc.fit_disc(data.read_gray(data.image_path(layout.train_images, n))) for n in pool}
        by_radius = sorted(pool, key=lambda n: -fits[n].radius)
        chosen = by_radius[:n_big]  # largest radii: the most suspicious fits
        rest = [n for n in pool if n not in chosen]
        chosen += list(rng.choice(rest, size=args.per_instrument - n_big, replace=False))
        med = float(np.median([f.radius for f in fits.values()]))
        lines.append("%s: %d images, median radius %.1f px" % (inst, len(pool), med))
        for n in chosen:
            img = data.read_gray(data.image_path(layout.train_images, n))
            d = fits[n]
            rgb = disc.draw_overlay(img, d, out_side=1024)
            cv2.imwrite(str(args.out / ("%s_r%03d.png" % (n, round(d.radius)))), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            tile = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, "%s r=%.0f c=(%.0f,%.0f)" % (n, d.radius, d.cx, d.cy), (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
            lines.append("  %s  radius %.1f  centre (%.1f, %.1f)" % (n, d.radius, d.cx, d.cy))
    cols = 4
    rows = [np.hstack(tiles[i : i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i : i + cols])))
            for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    cv2.imwrite(str(args.out / "contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("\n".join(lines))
    print("wrote %d overlays and contact_sheet.png to %s" % (len(tiles), args.out))


if __name__ == "__main__":
    main()
