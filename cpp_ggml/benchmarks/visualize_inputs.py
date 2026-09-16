#!/usr/bin/env python3
"""Visualize the raw multiview INPUT tensors (benchmarks/mv/<img>/image.bin).

image.bin is [V,3,224,224] float32, ImageNet-normalized. This script
un-normalizes it and tiles the V views into a single PNG so the input can be
viewed directly (the bin itself is a raw tensor, not an image file).

Usage: python3 benchmarks/visualize_inputs.py
Writes benchmarks/mv/<img>/preview.png for each image.
"""
import glob
import os

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")
MV = os.path.join(ROOT, "benchmarks", "mv")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def to_image(t):
    x = t * STD + MEAN
    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (0, 2, 3, 1))  # [V,H,W,3]
    return (x * 255).astype(np.uint8)


def main():
    for ibin in glob.glob(os.path.join(MV, "*", "image.bin")):
        img_dir = os.path.dirname(ibin)
        name = os.path.basename(img_dir)
        raw = np.fromfile(ibin, dtype=np.float32)
        V = 6
        per = V * 3 * 224 * 224
        if raw.size % per != 0:
            V = raw.size // (3 * 224 * 224)
            per = V * 3 * 224 * 224
        arr = raw[:per].reshape(V, 3, 224, 224)
        imgs = to_image(arr)
        # tile into 2 rows x (ceil(V/2)) cols
        cols = (V + 1) // 2
        rows = 2
        canvas = np.zeros((rows * 224, cols * 224, 3), dtype=np.uint8)
        for i in range(V):
            r, c = divmod(i, cols)
            canvas[r * 224:(r + 1) * 224, c * 224:(c + 1) * 224] = imgs[i]
        out = os.path.join(img_dir, "preview.png")
        Image.fromarray(canvas).save(out)
        print(f"wrote {out}  (V={V} multiview input)")


if __name__ == "__main__":
    main()
