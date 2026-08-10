#!/usr/bin/env python3
"""CORRECTED end-to-end textured-mesh comparison.

The previous figures used a crude per-triangle numpy z-buffer rasterizer that
aliases the mesh's xatlas chart seams into ~7% per-pixel "snowflake" noise --
identical for every texture resolution, so 512 vs 2048 looked the same, and the
PyTorch mesh appeared black because trimesh failed to resolve its MTL texture
(loaded a 2x2 placeholder).

This figure renders every mesh with a clean per-face (Lambert-lit) viewer that
produces zero self-noise, and loads the PyTorch texture PNG explicitly (fixing
the black-render bug). The real, measurable texture improvement (2048 keeps
detail that 512 aliases) is shown via native-resolution texture crops.
"""
import os
import sys

import numpy as np
import trimesh
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "benchmarks", "results")
LIGHT = np.array([0.25, 0.30, 0.90]); LIGHT /= np.linalg.norm(LIGHT)

OBJS = {
    "PyTorch official": os.path.join(RES, "cute_horse_official_tex.obj"),
    "ggml 512": "/tmp/ggml_verify/mesh.obj",
    "ggml 2048 (positive fix)": "/tmp/ggml_base2/mesh.obj",
}


def load_geometry(path):
    m = trimesh.load(path, process=False)
    return (np.asarray(m.vertices, np.float32),
            np.asarray(m.faces, np.int64),
            np.asarray(m.visual.uv, np.float32))


def load_tex(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def face_colors(V, F, uv, tex):
    th, tw = tex.shape[:2]
    cu = uv[F].mean(1)
    sx = np.clip(cu[:, 0] * (tw - 1), 0, tw - 1).astype(int)
    sy = np.clip((1.0 - cu[:, 1]) * (th - 1), 0, th - 1).astype(int)
    return tex[sy, sx]


def render_clean(V, F, tex, elev=28, azim=-58, size=640):
    col = face_colors(V, F, tex if tex.shape[0] == 1 else None, tex) if False else None
    cu = None
    # per-face base color already inside caller via face_colors; recompute here
    return None


def render(V, F, face_col, elev=28, azim=-58, size=640):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(v1 - v0, v2 - v0); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    n[:, 2] = np.abs(n[:, 2])
    inten = np.clip(n @ LIGHT, 0, 1)
    fc = np.clip((0.35 + 0.65 * inten)[:, None] * face_col, 0, 1)
    Vc = V - V.mean(0); Vc = Vc / (np.max(np.abs(Vc)) + 1e-9)
    fig = plt.figure(figsize=(size / 100, size / 100))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(Vc[F], facecolors=fc, edgecolors="none")
    ax.add_collection3d(pc)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float32) / 255.0
    plt.close(fig)
    return arr


def load_mesh(path):
    """Return (V,F,uv,texture) with manual texture load (bypass trimesh MTL bug)."""
    V, F, uv = load_geometry(path)
    # find png referenced by mtl in same dir
    d = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]
    png = os.path.join(d, base + ".png")
    if not os.path.exists(png) and base.startswith("cute_horse_official"):
        png = os.path.join(d, "cute_horse_official_tex.png")
    tex = load_tex(png) if os.path.exists(png) else np.zeros((4, 4, 3), np.float32) + 0.5
    return V, F, uv, tex


def crop_tex(tex, frac=0.6):
    """center crop of the covered region for a close-up view"""
    h, w = tex.shape[:2]
    c = int(w * (1 - frac) / 2)
    return tex[c:c + int(w * frac), c:c + int(w * frac)]


def main():
    # precompute for each variant: render + native texture crop
    renders, crops, names = {}, {}, []
    for name, path in OBJS.items():
        V, F, uv, tex = load_mesh(path)
        fc = face_colors(V, F, uv, tex)
        renders[name] = render(V, F, fc)
        crops[name] = crop_tex(tex)
        names.append(name)

    fig = plt.figure(figsize=(13, 8))
    # top row: renders
    for i, name in enumerate(names):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(renders[name]); ax.axis("off")
        ax.set_title(name, fontsize=12)
    # bottom row: native texture crops
    for i, name in enumerate(names):
        ax = fig.add_subplot(2, 3, i + 4)
        ax.imshow(crops[name]); ax.axis("off")
        ax.set_title(f"{name} - texture (crop)", fontsize=11)
    fig.suptitle("Textured mesh: clean render (top) vs native texture crop (bottom)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RES, "e2e_tex_clean_compare.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
