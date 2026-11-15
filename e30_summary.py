"""30-epoch check on S6: mean +- std over the seeds that are finished.   python e30_summary.py"""
import numpy as np
from aggregate_all import seeds, ms, fmt, paired

# Seeds 0, 1, 2 of the 30-epoch S6 runs (checkpoint names end in _e30, _e30_s1, _e30_s2).
b, a = seeds("base_hold_{k}_e30", "S6"), seeds("aug_hold_{k}_e30", "S6")
bs, as_ = seeds("base_hold_{k}_e30", "S6", "ssim"), seeds("aug_hold_{k}_e30", "S6", "ssim")
print("S6 held out, 30 epochs")
print("baseline PSNR :", fmt(ms(b)), "  per seed:", {k: round(v, 3) for k, v in b.items()})
print("augmented PSNR:", fmt(ms(a)), "  per seed:", {k: round(v, 3) for k, v in a.items()})
print("paired gain   :", fmt(ms(paired(b, a))), "  per seed:", {k: round(v, 3) for k, v in paired(b, a).items()})
print("baseline SSIM :", fmt(ms(bs), 4), "   augmented SSIM:", fmt(ms(as_), 4))
