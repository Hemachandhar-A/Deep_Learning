"""One training-step benchmark process for the two-GPU probe (development prompt section 9; Arch section 4.8 item 6).

    python kaggle/probe_step_benchmark.py [--steps 20] [--size 512] [--batch 8]

Runs ``steps`` real optimiser steps (forward, backward, Adam, mixed precision) of the project's U-Net on random data on
the *visible* GPU and prints one JSON line with the mean seconds per step.  ``probe_environment.py`` starts it alone and
then twice at once with ``CUDA_VISIBLE_DEVICES=0`` and ``=1`` to see whether two training processes on two cards run
side by side at full speed.  Random data and random weights: it measures speed, nothing else.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from filament.losses import CombinedLoss  # noqa: E402
from filament.model import ModelConfig, build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    model = build_model(ModelConfig(encoder_weights=None)).to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = CombinedLoss()
    x = torch.randn(a.batch, 1, a.size, a.size, device=dev)
    y = (torch.rand(a.batch, 1, a.size, a.size, device=dev) > 0.99).float()
    times = []
    for i in range(a.warmup + a.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(x)
        loss, _ = loss_fn(z, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        if i >= a.warmup:
            times.append(time.perf_counter() - t0)
    print(json.dumps({"gpu": torch.cuda.get_device_name(0), "steps": a.steps, "size": a.size, "batch": a.batch,
                      "seconds_per_step": sum(times) / len(times), "peak_gib": torch.cuda.max_memory_allocated() / 2**30}))


if __name__ == "__main__":
    main()
