#!/usr/bin/env python3
"""Render the reconstructed meshes into comparison images.

For each benchmarked input image it builds a 3(backend) x 3(precision) grid of
rendered meshes and saves it to benchmarks/results/render_<image>.png. The
f32-CUDA cell is the numerical reference (validated to match PyTorch within
parity tolerance), so it doubles as the "PyTorch-equivalent" baseline.

Meshes are the standard .obj files produced by run_bench.sh; .sdf.bin files are
only precision fingerprints and are ignored here (they live in results/sdf/).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..")            # cpp_ggml/
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # repo root (examples/)
RES = os.path.join(ROOT, "benchmarks", "results")
EX = os.path.join(REPO_ROOT, "examples")

BACKENDS = ["cpu", "cuda", "vulkan"]
PRECS = ["f32", "f16", "q8"]
BACKEND_LABEL = {"cpu": "CPU", "cuda": "CUDA", "vulkan": "Vulkan"}
SOURCE = {
    "blue_cat": "examples/blue_cat.png",
    "cute_horse": "examples/cute_horse.jpg",
    "fox": "examples/fox.jpg",
    "robot": "examples/robot.jpg",
}
LIGHT = np.array([0.25, 0.30, 0.90]); LIGHT /= np.linalg.norm(LIGHT)
VIEW = dict(elev=28, azim=-58)


def load_mesh(img, b, p):
    """Parse the OBJ (vertices may carry r g b) -> verts, faces, vcol [0,1]."""
    path = os.path.join(RES, f"{img}__{b}__{p}.obj")
    if not os.path.exists(path):
        return None
    verts, faces, vcol = [], [], []
    try:
        with open(path) as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                if t[0] == "v":
                    verts.append([float(t[1]), float(t[2]), float(t[3])])
                    if len(t) >= 7:
                        vcol.append([min(1.0, max(0.0, float(t[4]))),
                                     min(1.0, max(0.0, float(t[5]))),
                                     min(1.0, max(0.0, float(t[6])))])
                    else:
                        vcol.append([0.6, 0.6, 0.6])
                elif t[0] == "f":
                    # OBJ faces are 1-based; may carry v/vt/vn forms
                    idx = [int(x.split("/")[0]) - 1 for x in t[1:4]]
                    faces.append(idx)
    except Exception:
        return None
    if not verts or not faces:
        return None
    return np.asarray(verts, np.float32), np.asarray(faces, np.int64), np.asarray(vcol, np.float32)


def normalize(verts):
    v = verts - verts.mean(axis=0)
    v = v / (np.max(np.abs(v)) + 1e-9)
    return v


def face_colors(verts, faces, vcol):
    """Per-face color = mean of its 3 vertex colors, shaded by normal."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    back = n[:, 2] < 0
    n[back] *= -1
    intensity = np.clip(n @ LIGHT, 0.0, 1.0)[:, None]
    base = (vcol[faces[:, 0]] + vcol[faces[:, 1]] + vcol[faces[:, 2]]) / 3.0
    return np.clip(base * (0.30 + 0.70 * intensity), 0.0, 1.0)


def plot_mesh(ax, data):
    if data is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, 0.5, "n/a", ha="center", va="center", fontsize=9)
        return
    verts, faces, vcol = data
    verts = normalize(verts)
    tris = verts[faces]
    poly = Poly3DCollection(tris, facecolors=face_colors(verts, faces, vcol),
                            edgecolors="none", shade=False)
    ax.add_collection3d(poly)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1)); ax.set_proj_type("persp")
    ax.view_init(**VIEW)
    ax.set_axis_off()


def render_grid(img):
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.02, wspace=0.02)
    for ri, b in enumerate(BACKENDS):
        for ci, p in enumerate(PRECS):
            ax = fig.add_subplot(gs[ri, ci], projection="3d")
            plot_mesh(ax, load_mesh(img, b, p))
            if ri == 0:
                ax.set_title(f"{p.upper()}", fontsize=14)
            if ci == 0:
                ax.text2D(-0.18, 0.5, BACKEND_LABEL[b], fontsize=13,
                          transform=ax.transAxes, rotation=90, va="center", ha="center")

    fig.suptitle(f"InstantMesh reconstruction — {img}\nrows=backend, cols=quantization "
                 "(f32-CUDA = PyTorch-equivalent reference)", fontsize=15)

    # source image thumbnail
    src = os.path.join(REPO_ROOT, SOURCE.get(img, ""))
    if os.path.exists(src):
        img_ax = fig.add_axes([0.015, 0.80, 0.11, 0.16])
        img_ax.imshow(np.asarray(Image.open(src).convert("RGB")))
        img_ax.set_axis_off()
        img_ax.set_title("input", fontsize=9)

    out = os.path.join(RES, f"render_{img}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def load_pytorch(img):
    """Load the PyTorch-CUDA reference mesh (pytorch_ref/<img>__pytorch.obj)."""
    path = os.path.join(ROOT, "benchmarks", "pytorch_ref", f"{img}__pytorch.obj")
    if not os.path.exists(path):
        return None
    return load_obj(path)


def load_obj(path):
    """Shared OBJ parser -> verts, faces, vcol."""
    verts, faces, vcol = [], [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
                if len(t) >= 7:
                    vcol.append([min(1.0, max(0.0, float(t[4]))),
                                 min(1.0, max(0.0, float(t[5]))),
                                 min(1.0, max(0.0, float(t[6])))])
                else:
                    vcol.append([0.6, 0.6, 0.6])
            elif t[0] == "f":
                faces.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    return (np.asarray(verts, np.float32), np.asarray(faces, np.int64),
            np.asarray(vcol, np.float32))


def render_pytorch_compare(img):
    """Side-by-side: input image | PyTorch-CUDA | ggml CUDA f16 (both colored)."""
    fig = plt.figure(figsize=(16, 6))
    src = os.path.join(REPO_ROOT, SOURCE.get(img, ""))
    ax_in = fig.add_subplot(1, 3, 1)
    ax_in.imshow(np.asarray(Image.open(src).convert("RGB")))
    ax_in.set_axis_off()
    ax_in.set_title("input", fontsize=14)
    labels = ["PyTorch-CUDA reference", "ggml CUDA (f16)"]
    datas = [load_pytorch(img), load_mesh(img, "cuda", "f16")]
    for i, (d, lab) in enumerate(zip(datas, labels), start=2):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        plot_mesh(ax, d)
        ax.set_title(lab, fontsize=13)
    out = os.path.join(RES, f"pytorch_vs_ggml_{img}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    imgs = sorted({f.split("__")[0] for f in os.listdir(RES) if f.endswith(".obj")})
    if not imgs:
        print("no .obj meshes found — run benchmarks/run_bench.sh first")
        return
    for img in imgs:
        render_grid(img)
        render_pytorch_compare(img)


if __name__ == "__main__":
    main()
