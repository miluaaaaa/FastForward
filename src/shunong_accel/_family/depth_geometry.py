"""Frozen input policies for the 32B depth study; no labels enter geometry."""
import math

VARIANTS = ['square256', 'aspect256', 'square320', 'aspect320']


def canvas(height, width, variant):
    assert height > 0 and width > 0 and variant in VARIANTS
    side = 256 if variant.endswith('256') else 320
    if variant.startswith('square'):
        return side, side
    pairs = [(8, 8), (6, 10), (4, 16), (2, 32)] if side == 256 else [(10, 10), (8, 12), (6, 16), (4, 24), (2, 50)]
    pairs = sorted(set(pairs + [(w, h) for h, w in pairs]))
    h, w = min(pairs, key=lambda p: (abs(math.log((p[0] / p[1]) / (height / width))), -p[0]*p[1], p))
    return h * 32, w * 32


def fit(height, width, target):
    h, w = target
    ratio = min(h / height, w / width)
    nh, nw = min(h, max(1, round(height * ratio))), min(w, max(1, round(width * ratio)))
    return nh, nw, (h - nh)//2, (w - nw)//2
