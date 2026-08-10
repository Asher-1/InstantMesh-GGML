#!/usr/bin/env python3
"""Focused, honest ggml 512 vs 2048 comparison (the 'positive fix').
Clean render (matplotlib per-face, zero self-noise) + native texture crops.
"""
import os, sys
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from texmap_compare_clean import load_mesh, face_colors, render

RES = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
variants = {
    "ggml 512": "/tmp/ggml_verify/mesh.obj",
    "ggml 2048 (positive fix)": "/tmp/ggml_base2/mesh.obj",
}

fig, axs = plt.subplots(2, 2, figsize=(9, 9))
for i, (name, path) in enumerate(variants.items()):
    V, F, uv, tex = load_mesh(path)
    fc = face_colors(V, F, uv, tex)
    axs[0, i].imshow(render(V, F, fc)); axs[0, i].axis("off")
    axs[0, i].set_title(f"{name}  —  clean render", fontsize=12)
    # texture crop
    h, w = tex.shape[:2]
    c = int(w * 0.35)
    crop = tex[c:c + int(w * 0.3), c:c + int(w * 0.3)]
    axs[1, i].imshow(crop); axs[1, i].axis("off")
    axs[1, i].set_title(f"{name} — texture crop", fontsize=12)
fig.suptitle("Positive fix: higher bake resolution (snowflake was a renderer artifact)", fontsize=13)
fig.tight_layout()
out = os.path.join(RES, "tex_512_vs_2048_clean.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
