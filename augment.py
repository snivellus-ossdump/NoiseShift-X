"""Mixed Poisson-Gaussian noise augmentation (the invariance idea).

Heteroscedastic model on linear-ish intensities in [0,1]:
    y = x + N(0, a*x + b)      (a = signal-dependent / shot, b = signal-independent / read)
Applied ON TOP of the real noisy input with probability p, so the network sees more noise
diversity than any single camera provides. Parameters are sampled per patch (log-uniform).
NOTE: sRGB images are gamma-encoded, so this is an approximation; state this in the report.
"""
import numpy as np

def add_poisson_gaussian(noisy_u8, rng, a_range=(1e-4, 5e-3), b_range=(1e-5, 2e-3), gauss_only=False):
    # Work in float intensities in [0, 1].
    x = noisy_u8.astype(np.float32) / 255.0
    # Draw the noise parameters log-uniformly for this patch: a = shot-noise coefficient (signal-dependent),
    # b = read-noise variance (signal-independent). gauss_only=True is the control experiment (a = 0).
    a = 0.0 if gauss_only else float(np.exp(rng.uniform(np.log(a_range[0]), np.log(a_range[1]))))   # gauss_only = control: signal-independent noise
    b = float(np.exp(rng.uniform(np.log(b_range[0]), np.log(b_range[1]))))
    # Per-pixel standard deviation: variance = a * intensity + b, so brighter pixels get more noise
    # when a > 0 (Poisson-Gaussian model; a raw-sensor model, only an approximation for sRGB).
    sigma = np.sqrt(np.clip(a * x + b, 0, None))
    # Add zero-mean Gaussian noise with that per-pixel std ON TOP of the already noisy real patch.
    y = x + rng.standard_normal(x.shape).astype(np.float32) * sigma
    # Back to uint8 in [0, 255]; also return the sampled (a, b), which the unit test uses.
    return np.clip(y * 255.0, 0, 255).astype(np.uint8), (a, b)

def maybe_augment(noisy_u8, rng, p=0.5, **kw):
    # With probability p the patch receives extra synthetic noise; otherwise it is left unchanged.
    if rng.random() < p:
        out, _ = add_poisson_gaussian(noisy_u8, rng, **kw)
        return out
    return noisy_u8
