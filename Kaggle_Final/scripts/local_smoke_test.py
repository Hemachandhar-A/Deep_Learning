"""GPU memory/speed smoke test for the filament-segmentation candidates.
Needs: python, torch with CUDA, segmentation-models-pytorch.   Run:  python gpu_smoke_test.py
Reports PEAK ALLOCATED GPU memory of real training steps (forward + backward + AdamW step).
The CUDA context itself (~0.3-0.5 GB) is not included in torch's number; compare with nvidia-smi.

(This is the script of Arch section 4.6, unchanged, so that every card is measured the same way; it is kept in the
repository as scripts/local_smoke_test.py.  Inputs and targets are random numbers and the models start from random
weights: it measures memory and step time only, no data loading.)"""
import time
import torch
import segmentation_models_pytorch as smp

def run(arch, enc, size, batch, amp):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = getattr(smp, arch)(encoder_name=enc, encoder_weights=None, in_channels=1, classes=1).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    x = torch.randn(batch, 1, size, size, device='cuda')
    y = (torch.rand(batch, 1, size, size, device='cuda') > 0.99).float()
    try:
        for i in range(5):
            if i == 2:
                torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        return 'OK', torch.cuda.max_memory_allocated() / 2**30, (time.time() - t0) / 3
    except torch.cuda.OutOfMemoryError:
        return 'OOM', torch.cuda.max_memory_allocated() / 2**30, float('nan')
    finally:
        del model, opt, x, y
        torch.cuda.empty_cache()

if __name__ == '__main__':
    p = torch.cuda.get_device_properties(0)
    print('GPU:', p.name, '| total memory %.2f GiB' % (p.total_memory / 2**30), '| torch', torch.__version__, '| smp', smp.__version__)
    configs = [  # (architecture, encoder, image side in px, batch, mixed precision)
        ('Unet', 'resnet34', 512, 4, True), ('Unet', 'resnet34', 512, 8, True),
        ('Unet', 'resnet34', 1024, 1, True), ('Unet', 'resnet34', 1024, 2, True), ('Unet', 'resnet34', 1024, 1, False),
        ('FPN', 'resnet34', 1024, 2, True), ('UnetPlusPlus', 'resnet34', 512, 4, True), ('UnetPlusPlus', 'resnet34', 1024, 1, True),
        ('Unet', 'resnet50', 1024, 1, True),
    ]
    print('%-13s %-9s %5s %5s %4s | %-4s %12s %12s' % ('arch', 'encoder', 'size', 'batch', 'amp', 'res', 'peak GiB', 'sec/step'))
    for c in configs:
        try:
            r = run(*c)
        except Exception as e:  # keep going if an architecture is unavailable
            r = ('ERR ' + type(e).__name__, float('nan'), float('nan'))
        print('%-13s %-9s %5d %5d %4s | %-4s %12.2f %12.2f' % (c[0], c[1], c[2], c[3], 'yes' if c[4] else 'no', *r))
