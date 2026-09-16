#!/usr/bin/env python3
"""Visualizations for the model-compression + multi-backend benchmark report.

Generates into results/:
  precision_loss_heatmap.png  SDF grid error (f16/q8 vs f32) per image (precision loss)
  perf_bar.png                inference time (ms) by backend, grouped by precision
  perf_fps.png                throughput (fps) by backend, grouped by precision
  latency_line.png            inference-time line chart vs precision per backend
  energy_bar.png              energy efficiency (W/fps) by backend, grouped by precision

Reads results/full_bench.csv (bench_full.py) and the *.sdf.bin dumps.
Usage: python3 benchmarks/visualize.py [image]
"""
import os
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "benchmarks", "results")
IMG = sys.argv[1] if len(sys.argv) > 1 else "cute_horse"
PRECS = ["f32", "f16", "q8"]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_sdf(img, pr):
    return np.fromfile(os.path.join(RES, f"{img}__cpu__{pr}.sdf.bin"), dtype=np.float32)


def plot_precision_heatmap(images):
    # rows = precision, cols = image; cell = mean_abs SDF error vs f32
    m = np.zeros((2, len(images)))
    for j, im in enumerate(images):
        f32 = load_sdf(im, "f32")
        for i, pr in enumerate(["f16", "q8"]):
            d = load_sdf(im, pr) - f32
            m[i, j] = np.abs(d).mean()
    fig, ax = plt.subplots(figsize=(max(6, len(images) * 1.6), 2.6))
    im = ax.imshow(m, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(images))); ax.set_xticklabels(images)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["f16 vs f32", "q8 vs f32"])
    for i in range(2):
        for j in range(len(images)):
            ax.text(j, i, f"{m[i,j]:.4f}", ha="center", va="center", color="white", fontsize=10)
    ax.set_title("Precision-loss heatmap: mean |SDF error| vs f32 (grid-res 88)\n"
                 "consistency across images (low variance = controllable deviation)")
    fig.colorbar(im, ax=ax, label="mean |ΔSDF|")
    fig.tight_layout()
    p = os.path.join(RES, "precision_loss_heatmap.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print("wrote", p)


def plot_perf(rows):
    backends = ["pytorch-cuda", "ggml-cuda", "ggml-vulkan", "ggml-cpu"]
    data = {b: {pr: [] for pr in PRECS} for b in backends}
    for r in rows:
        if r["precision"] in PRECS and r["backend"] in data:
            data[r["backend"]][r["precision"]].append(float(r["time_ms"]))
    for b in data:
        for pr in data[b]:
            data[b][pr] = np.mean(data[b][pr]) if data[b][pr] else np.nan

    x = np.arange(len(backends)); width = 0.26
    colors = {"f32": "#7f8c8d", "f16": "#3498db", "q8": "#9b59b6"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, pr in enumerate(PRECS):
        vals = [data[b][pr] for b in backends]
        ax.bar(x + (i - 1) * width, vals, width, label=pr, color=colors[pr])
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(j + (i - 1) * width, v + 8, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(backends)
    ax.set_ylabel("full-pipeline inference time (ms)")
    ax.set_ylim(0, max([v for b in backends for v in [data[b][pr] for pr in PRECS]
                        if not np.isnan(v)] or [100]) * 1.12)
    ax.set_title(f"Inference performance by backend x precision ({IMG}, grid-res=88)")
    ax.legend(title="precision"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RES, "perf_bar.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print("wrote", p)

    # FPS
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, pr in enumerate(PRECS):
        vals = [1000.0 / data[b][pr] if not np.isnan(data[b][pr]) else 0 for b in backends]
        ax.bar(x + (i - 1) * width, vals, width, label=pr, color=colors[pr])
        for j, v in enumerate(vals):
            if v: ax.text(j + (i - 1) * width, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(backends)
    ax.set_ylabel("throughput (fps)"); ax.set_title(f"Throughput by backend x precision ({IMG})")
    ax.legend(title="precision"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RES, "perf_fps.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print("wrote", p)


def plot_latency_line(rows):
    backends = ["pytorch-cuda", "ggml-cuda", "ggml-vulkan", "ggml-cpu"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    markers = {"pytorch-cuda": "o", "ggml-cuda": "s", "ggml-vulkan": "^", "ggml-cpu": "D"}
    for b in backends:
        vals = []
        for pr in PRECS:
            s = [float(r["time_ms"]) for r in rows if r["backend"] == b and r["precision"] == pr]
            vals.append(np.mean(s) if s else np.nan)
        ax.plot([1, 2, 3], vals, marker=markers[b], label=b, linewidth=2)
        for i, v in enumerate(vals):
            if not np.isnan(v): ax.text(1 + i, v, f"{v:.0f}", fontsize=7, ha="center", va="bottom")
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(PRECS)
    ax.set_xlabel("model precision"); ax.set_ylabel("inference time (ms)")
    ax.set_title(f"Inference-time lines vs precision per backend ({IMG})")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RES, "latency_line.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print("wrote", p)


def plot_energy(rows):
    backends = ["pytorch-cuda", "ggml-cuda", "ggml-vulkan", "ggml-cpu"]
    x = np.arange(len(backends)); width = 0.26
    colors = {"f32": "#7f8c8d", "f16": "#3498db", "q8": "#9b59b6"}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, pr in enumerate(PRECS):
        vals = []
        for b in backends:
            s = [float(r["wat_per_fps"]) for r in rows
                 if r["backend"] == b and r["precision"] == pr]
            vals.append(np.mean(s) if s else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=pr, color=colors[pr])
        for j, v in enumerate(vals):
            if not np.isnan(v): ax.text(j + (i - 1) * width, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(backends)
    ax.set_ylabel("energy efficiency (W / fps, lower is better)")
    ax.set_title(f"Energy efficiency by backend x precision ({IMG})")
    ax.legend(title="precision"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RES, "energy_bar.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print("wrote", p)


def main():
    rows = read_csv(os.path.join(RES, "full_bench.csv"))
    if not rows:
        print("no full_bench.csv — run bench_full.py first"); sys.exit(1)
    images = ["blue_cat", "cute_horse", "fox", "robot"]
    plot_precision_heatmap(images)
    plot_perf(rows)
    plot_latency_line(rows)
    plot_energy(rows)


if __name__ == "__main__":
    main()
