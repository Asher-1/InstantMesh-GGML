#!/usr/bin/env python3
"""Performance comparison: ggml (per-stage CUDA) + ggml (full CPU/CUDA/Vulkan
wall) vs the PyTorch reference on the same hardware.

Reads:
  results/stage_times.csv   ggml per-stage CUDA inference timings (dino/transformer/synthesizer/extract)
  results/times.csv         ggml full-pipeline wall times (cpu/cuda/vulkan x f32/f16/q8)
  pytorch_ref/times.csv     PyTorch reference (forward_planes + extract_mesh) on CUDA

Writes a grouped bar chart and a markdown table to results/.
"""
import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "benchmarks", "results")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    stages = read_csv(os.path.join(RES, "stage_times.csv"))      # ggml cuda per-stage
    walls = read_csv(os.path.join(RES, "times.csv"))             # ggml full pipeline
    pt = read_csv(os.path.join(ROOT, "benchmarks", "pytorch_ref", "times.csv"))

    # ---- per-image mean ggml CUDA inference time by precision -----------------
    precs = ["f32", "f16", "q8"]
    images = sorted({r["image"] for r in stages})
    ggml_cuda = {p: {r["image"]: float(r["total_s"]) for r in stages if r["precision"] == p}
                 for p in precs}
    pytorch = {r["image"]: float(r["wall_s"]) for r in pt if r["backend"] == "pytorch"}

    # ---- chart: ggml CUDA (f32/f16/q8) vs PyTorch, per image -----------------
    x = np.arange(len(images))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"f32": "#7f8c8d", "f16": "#3498db", "q8": "#9b59b6"}
    for i, p in enumerate(precs):
        ax.bar(x + (i - 1) * width, [ggml_cuda[p][im] for im in images],
               width, label=f"ggml CUDA {p}", color=colors[p])
    pt_vals = [pytorch.get(im, 0) for im in images]
    ax.bar(x + 2 * width, pt_vals, width, label="PyTorch CUDA", color="#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(images)
    ax.set_ylabel("inference time (s)"); ax.set_ylim(0, 9.5)
    ax.set_title("InstantMesh inference time, same CUDA GPU (grid_res=88)\n"
                 "ggml per-stage inference vs PyTorch forward_planes+extract_mesh")
    ax.legend(ncol=4, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, p in enumerate(precs):
        for j, im in enumerate(images):
            ax.text(x[j] + (i - 1) * width, ggml_cuda[p][im] + 0.12,
                    f"{ggml_cuda[p][im]:.2f}", ha="center", fontsize=7)
    for j, im in enumerate(images):
        ax.text(x[j] + 2 * width, pt_vals[j] + 0.12, f"{pt_vals[j]:.2f}",
                ha="center", fontsize=7)
    fig.tight_layout()
    chart = os.path.join(RES, "perf_ggml_vs_pytorch.png")
    fig.savefig(chart, dpi=120)
    plt.close(fig)

    # ---- stage breakdown (ggml CUDA f16) -------------------------------------
    f16 = [r for r in stages if r["precision"] == "f16"]
    labels = ["dino", "transformer", "synthesizer", "extract"]
    keys = ["dino_s", "transformer_s", "synthesizer_s", "extract_s"]
    means = [np.mean([float(r[k]) for r in f16]) for k in keys]
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.bar(labels, means, color="#2c3e50")
    for i, v in enumerate(means):
        ax2.text(i, v + 0.05, f"{v:.2f}s", ha="center", fontsize=9)
    ax2.set_ylabel("mean time (s)")
    ax2.set_title("ggml CUDA f16 stage breakdown (mean over 4 images)")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    stage_chart = os.path.join(RES, "perf_ggml_stage_breakdown.png")
    fig2.savefig(stage_chart, dpi=120)
    plt.close(fig2)

    # ---- markdown table -------------------------------------------------------
    pt_mean = np.mean(list(pytorch.values())) if pytorch else 0
    lines = ["### ggml (CUDA, per-stage inference) vs PyTorch (CUDA) — wall seconds\n",
             "| image | ggml f32 | ggml f16 | ggml q8 | PyTorch | ggml f16 / PyTorch |",
             "|-------|----------|----------|---------|---------|--------------------|"]
    for im in images:
        g = [f"{ggml_cuda[p][im]:.2f}" for p in precs]
        ptv = pytorch.get(im, float("nan"))
        ratio = f"{ggml_cuda['f16'][im] / ptv:.1f}x" if ptv else "n/a"
        lines.append(f"| {im} | {g[0]} | {g[1]} | {g[2]} | {ptv:.2f} | {ratio} |")
    lines.append("")
    lines.append("Mean ggml CUDA f16 total inference: "
                 f"{np.mean([ggml_cuda['f16'][im] for im in images]):.2f}s vs PyTorch mean {pt_mean:.2f}s.")
    lines.append("")

    # ---- full-pipeline wall comparison (all backends) -------------------------
    lines.append("### ggml full-pipeline wall time (incl. model load + mesh export) — seconds\n")
    lines.append("| image | backend | f32 | f16 | q8 |")
    lines.append("|-------|---------|-----|-----|----|")
    for im in images:
        for b in ["cpu", "cuda", "vulkan"]:
            row = {r["precision"]: float(r["wall_s"]) for r in walls
                   if r["image"] == im and r["backend"] == b}
            lines.append(f"| {im} | {b} | {row.get('f32', float('nan')):.2f} | "
                         f"{row.get('f16', float('nan')):.2f} | {row.get('q8', float('nan')):.2f} |")
        lines.append(f"| {im} | PyTorch | -- | -- | -- | (CUDA {pytorch.get(im, 0):.2f}s)")
    report = "\n".join(lines)
    with open(os.path.join(RES, "PERFORMANCE.md"), "w") as f:
        f.write(report + "\n")
    print(report)
    print("\nwrote", chart, stage_chart, os.path.join(RES, "PERFORMANCE.md"))


if __name__ == "__main__":
    main()
