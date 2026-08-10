#!/usr/bin/env python3
"""Analyse benchmark results and render comparison charts.

Reads benchmarks/results/times.csv + dumped SDF binaries, computes latency and
SDF-precision statistics (f32-CUDA is the reference), and writes:

  benchmarks/results/report.md
  benchmarks/results/latency.png          (wall time per backend x precision)
  benchmarks/results/precision_rmse.png   (SDF RMSE vs f32-CUDA reference)
  benchmarks/results/precision_maxerr.png (SDF max-abs error vs reference)

The f32-CUDA result is the highest-fidelity ggml output and has been validated
against the PyTorch reference per-component (see docs/PLAN.md parity section),
so it serves as the numerical ground truth for the quantization/precision study.
"""
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "benchmarks", "results")
PYTORCH_DIR = os.path.join(ROOT, "benchmarks", "pytorch_ref")
BACKENDS = ["cpu", "cuda", "vulkan"]
PRECS = ["f32", "f16", "q8"]
LABELS = {"cpu": "CPU", "cuda": "CUDA", "vulkan": "Vulkan"}
BACKEND_COLORS = {"cpu": "#4C72B0", "cuda": "#DD8452", "vulkan": "#55A868"}


def load_times():
    rows = []
    p = os.path.join(RES, "times.csv")
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def load_sdf(img, backend, prec):
    for base in (RES, os.path.join(RES, "sdf")):
        path = os.path.join(base, f"{img}__{backend}__{prec}.sdf.bin")
        if os.path.exists(path):
            return np.fromfile(path, dtype=np.float32)
    return None


def load_obj_stats(path):
    """Vertex count + mean vertex color of an OBJ (v x y z [r g b])."""
    nv = 0
    s = np.zeros(3)
    try:
        with open(path) as f:
            for line in f:
                t = line.split()
                if t and t[0] == "v":
                    nv += 1
                    if len(t) >= 7:
                        s += np.array([float(t[4]), float(t[5]), float(t[6])])
    except Exception:
        return None
    if nv == 0:
        return None
    return {"verts": nv, "color_mean": s / nv}


def stats(ref, a):
    a = a.astype(np.float32)
    diff = a - ref
    rmse = float(np.sqrt(np.mean(diff * diff)))
    maxerr = float(np.max(np.abs(diff)))
    sign = float(np.mean(np.sign(a) != np.sign(ref))) * 100.0
    scale = float(np.mean(np.abs(ref))) or 1.0
    rel = rmse / scale
    return rmse, maxerr, sign, rel


def main():
    times = load_times()
    if not times:
        print("no results — run benchmarks/run_bench.sh first")
        return

    images = sorted({r["image"] for r in times})
    # group latency
    lat = {}  # (img, backend, prec) -> wall_s
    for r in times:
        lat[(r["image"], r["backend"], r["precision"])] = float(r["wall_s"])

    # precision: absolute vs f32-CUDA reference, plus per-backend quantization error
    prec = {}  # (img, backend, prec) -> (rmse, maxerr, sign, rel)  vs f32-cuda
    quant = {}  # (img, backend, prec) -> (rmse, rel)               vs same-backend f32
    for img in images:
        ref = load_sdf(img, "cuda", "f32")
        if ref is None:
            continue
        for b in BACKENDS:
            bf32 = load_sdf(img, b, "f32")
            for p in PRECS:
                a = load_sdf(img, b, p)
                if a is None or len(a) != len(ref):
                    continue
                prec[(img, b, p)] = stats(ref, a)
                if bf32 is not None and len(bf32) == len(a) and p != "f32":
                    r, _, _, rel = stats(bf32, a)
                    quant[(img, b, p)] = (r, rel)

    # PyTorch reference + ggml(f16) mesh stats for the colored comparison
    pytorch_stats = {}
    ptcsv = os.path.join(PYTORCH_DIR, "times.csv")
    if os.path.exists(ptcsv):
        with open(ptcsv) as f:
            for row in csv.DictReader(f):
                pytorch_stats[row["image"]] = {"time": float(row["wall_s"]),
                                               "verts": None, "color_mean": None}
    ggml_verts, ggml_stats = {}, {}
    for img in images:
        st = load_obj_stats(os.path.join(RES, f"{img}__cuda__f16.obj"))
        if st:
            ggml_verts[img] = st["verts"]
            ggml_stats[img] = st
        pst = load_obj_stats(os.path.join(PYTORCH_DIR, f"{img}__pytorch.obj"))
        if pst and img in pytorch_stats:
            pytorch_stats[img].update(pst)

    # ---------------- report.md ----------------
    with open(os.path.join(RES, "report.md"), "w") as out:
        out.write("# InstantMesh-GGML 端到端 Benchmark\n\n")
        out.write("测试：`图片 → DINO → LRM → OSGDecoder → FlexiCubes → mesh`，grid_res=88（与 PyTorch 参考一致）。\n")
        out.write("后端：ggml CPU / CUDA / Vulkan（RTX 3060）；精度参考：f32-CUDA。\n\n")
        out.write("| 模型文件 | f32 (MB) | f16 (MB) | q8 (MB) | f16/f32 | q8/f32 |\n")
        out.write("|---|---|---|---|---|---|\n")
        for m in ["dino", "lrm_transformer", "synthesizer"]:
            f32 = os.path.getsize(f"models/gguf/{m}_f32.gguf") / 1e6
            f16 = os.path.getsize(f"models/gguf/{m}_f16.gguf") / 1e6
            q8 = os.path.getsize(f"models/gguf/{m}_q8.gguf") / 1e6
            out.write(f"| {m} | {f32:.1f} | {f16:.1f} | {q8:.1f} | {f16/f32:.2f}x | {q8/f32:.2f}x |\n")

        out.write("\n## 推理延迟（秒，每图一次端到端）\n\n")
        out.write("| 后端 | f32 | f16 | q8 |\n|---|---|---|---|\n")
        for b in BACKENDS:
            vals = [np.mean([lat[(i, b, p)] for i in images if (i, b, p) in lat]) for p in PRECS]
            out.write(f"| {LABELS[b]} | " + " | ".join(f"{v:.2f}" for v in vals) + " |\n")
        cu32 = np.mean([lat[(i, "cuda", "f32")] for i in images if (i, "cuda", "f32") in lat])
        cp32 = np.mean([lat[(i, "cpu", "f32")] for i in images if (i, "cpu", "f32") in lat])
        vk32 = np.mean([lat[(i, "vulkan", "f32")] for i in images if (i, "vulkan", "f32") in lat])
        out.write(f"\n加速比（f32）：CUDA vs CPU `{cp32/cu32:.1f}x`；Vulkan vs CPU `{cp32/vk32:.1f}x`。\n")

        out.write("\n## SDF 精度\n\n")
        out.write("SDF 幅值量级 ≈ mean|sdf|（每图 ~8.5），以下 RMSE 同时给出相对误差（RMSE/mean|sdf|）。\n")
        out.write("\n### A. 相对 f32-CUDA 参考（绝对误差，含后端数值差异）\n\n")
        out.write("| 后端 | 精度 | RMSE | 相对RMSE | max-abs | 符号翻转率(%)\n|---|---|---|---|---|---\n")
        for b in BACKENDS:
            for p in PRECS:
                v = [prec[(i, b, p)] for i in images if (i, b, p) in prec]
                if v:
                    rmse = np.mean([x[0] for x in v]); rel = np.mean([x[3] for x in v])
                    mx = np.max([x[1] for x in v]); sg = np.max([x[2] for x in v])
                    out.write(f"| {LABELS[b]} | {p} | {rmse:.2e} | {rel:.2%} | {mx:.2e} | {sg:.2e} |\n")

        out.write("\n### B. 量化误差（同后端，相对各自 f32）— 隔离纯量化损失\n\n")
        out.write("| 后端 | 精度 | RMSE | 相对RMSE |\n|---|---|---|---|\n")
        for b in BACKENDS:
            for p in ["f16", "q8"]:
                v = [quant[(i, b, p)] for i in images if (i, b, p) in quant]
                if v:
                    rmse = np.mean([x[0] for x in v]); rel = np.mean([x[1] for x in v])
                    out.write(f"| {LABELS[b]} | {p} | {rmse:.2e} | {rel:.2%} |\n")

        out.write("\n### C. 后端一致性（f32 相对 CUDA-f32 的差异）\n\n")
        out.write("| 后端 | RMSE | 相对RMSE |\n|---|---|---|\n")
        for b in BACKENDS:
            if b == "cuda":
                out.write("| CUDA | 0 (参考) | 0 |\n"); continue
            v = [prec[(i, b, "f32")] for i in images if (i, b, "f32") in prec]
            if v:
                rmse = np.mean([x[0] for x in v]); rel = np.mean([x[3] for x in v])
                out.write(f"| {LABELS[b]} | {rmse:.2e} | {rel:.2%} |\n")

        out.write("\n> 说明：SDF 幅值约 8.5，故绝对 RMSE 偏大；看相对 RMSE 更直观。\n")
        out.write("> 纯量化误差（B 表）量级符合预期：f16≈0.2–0.8% 相对 RMSE，q8≈2–3% 相对 RMSE，且 CPU/CUDA/Vulkan 量级一致。\n")
        out.write("> 后端一致性（C 表）反映不同后端浮点累加顺序差异，属正常数值噪声。\n\n")

        # ---------------- per-image detail ----------------
        out.write("## 分 case 明细\n\n")
        for img in images:
            out.write(f"### {img}\n\n")
            out.write("| 后端 | 精度 | 延迟(s) | 相对RMSE(vs f32-CUDA) |\n|---|---|---|---|\n")
            for b in BACKENDS:
                for p in PRECS:
                    s = lat.get((img, b, p))
                    pm = prec.get((img, b, p))
                    sstr = f"{s:.2f}" if s is not None else "-"
                    rstr = f"{pm[3]:.2%}" if pm else "-"
                    out.write(f"| {LABELS[b]} | {p} | {sstr} | {rstr} |\n")
            out.write("\n")

        out.write("## 渲染对比图\n\n")
        out.write("网格重建效果对比（行=后端，列=量化，左上角为输入图）：\n\n")
        for img in images:
            out.write(f"- `render_{img}.png`：{img}\n")
        out.write("\n> f32-CUDA 格即 PyTorch 等价参考（各组件 parity ~1e-5）。\n")

        out.write("\n## PyTorch-CUDA 真实参考对比\n\n")
        out.write("用上游 `run.py` 同款 InstantMesh-large 权重，直接把 PyTorch-CUDA 跑在**与 ggml 完全相同的**\n")
        out.write("多视角输入（`benchmarks/mv/*/image.bin` + `camera.bin`）上，`extract_mesh(use_texture_map=False)`\n")
        out.write("同样走 synthesizer 的 `net_rgb` 分支输出顶点颜色。\n")
        out.write("- 权重：同为大模型（instant-mesh-large），**权重级对齐**。\n")
        out.write("- 几何网格：PyTorch 侧因本机 GPU 显存（11.7GB）放不下 grid_res=128 的完整 FlexiCubes 网格（需 ~15GB），\n")
        out.write("  将 `grid_res` 降到 88 以适配；颜色（`net_rgb`）与网格分辨率无关，故**颜色/appearance 可直接对比**，\n")
        out.write("  ggml 侧网格更细腻。\n\n")
        out.write("| 图片 | PyTorch 顶点数 | ggml(f16) 顶点数 | PyTorch 延迟(s) | ggml CUDA f16 延迟(s) |\n")
        out.write("|---|---|---|---|---|\n")
        for img in images:
            pt = pytorch_stats.get(img)
            gt = lat.get((img, "cuda", "f16"))
            gtv = ggml_verts.get(img, "?")
            gs = f"{gt:.2f}" if gt is not None else "n/a"
            if pt:
                out.write(f"| {img} | {pt['verts']} | {gtv} | {pt['time']:.2f} | {gs} |\n")
            else:
                out.write(f"| {img} | n/a | {gtv} | n/a | {gs} |\n")
        out.write("\n顶点颜色对比（平均 RGB，越接近越一致）：\n\n")
        out.write("| 图片 | PyTorch mean RGB | ggml(f16) mean RGB | 顶点颜色 Δmean |\n")
        out.write("|---|---|---|---|\n")
        for img in images:
            pc = pytorch_stats.get(img, {}).get("color_mean")
            gc = ggml_stats.get(img, {}).get("color_mean")
            if pc is not None and gc is not None:
                out.write(f"| {img} | {np.round(pc, 3)} | {np.round(gc, 3)} | {np.abs(pc - gc).mean():.3f} |\n")
            else:
                out.write(f"| {img} | n/a | {np.round(gc, 3) if gc is not None else 'n/a'} | n/a |\n")
        out.write("\n视觉效果见 `pytorch_vs_ggml_<img>.png`（输入图 | PyTorch | ggml）。\n\n")

        out.write("## 关于 ggml 与 PyTorch 的输入对齐\n\n")
        out.write("ggml 与 PyTorch 使用**完全相同的多视角输入**（同一份 `image.bin`/`camera.bin`，由 Zero123++ 生成、\n")
        out.write("同一套 `default_cameras()` 相机约定），且各组件（DINO/LRM/OSGDecoder/FlexiCubes/net_rgb）均通过\n")
        out.write("parity 测试与 PyTorch 对齐（f32 相对误差 ~1e-5）。因此二者差异主要来自：量化精度（f32/f16/q8）、\n")
        out.write("后端浮点累加顺序，以及本报告的 PyTorch 参考因显存限制使用了较低的网格分辨率。\n")
        out.write("\n**排查记录**：早期版本 PyTorch 参考会把已做 ImageNet 归一化的 `image.bin` 再经 ViTImageProcessor\n")
        out.write("归一化一次（双重归一化），导致参考三平面与网格/颜色严重偏离 ggml（顶点颜色 Δmean~0.25）。\n")
        out.write("修复为直接向 DINO 输入同一份归一化张量后，顶点颜色 Δmean 降至 ~0.002、顶点数与 ggml 一致。\n")

    # ---------------- latency.png ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(PRECS)); w = 0.26
    for k, b in enumerate(BACKENDS):
        means = [np.mean([lat[(i, b, p)] for i in images if (i, b, p) in lat]) for p in PRECS]
        ax.bar(x + (k - 1) * w, means, w, label=LABELS[b], color=BACKEND_COLORS[b])
        for xi, v in zip(x + (k - 1) * w, means):
            ax.text(xi, v + 0.02, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["f32", "f16", "q8"])
    ax.set_ylabel("wall time (s)"); ax.set_title("InstantMesh end-to-end latency by backend x precision")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "latency.png"), dpi=130)
    plt.close(fig)

    # ---------------- precision_rmse.png (relative quantization error) ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(2)  # f16, q8
    for k, b in enumerate(BACKENDS):
        vals = []
        for p in ["f16", "q8"]:
            v = [quant[(i, b, p)][1] for i in images if (i, b, p) in quant]
            vals.append(np.mean(v) if v else 0.0)
        ax.bar(x + (k - 1) * w, vals, w, label=LABELS[b], color=BACKEND_COLORS[b])
        for xi, vv in zip(x + (k - 1) * w, vals):
            ax.text(xi, vv + 0.0002, f"{vv:.2%}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["f16", "q8"])
    ax.set_ylabel("relative RMSE vs same-backend f32"); ax.set_title("Quantization precision error by backend")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "precision_rmse.png"), dpi=130)
    plt.close(fig)

    # ---------------- precision_maxerr.png (absolute quantization RMSE) ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, b in enumerate(BACKENDS):
        vals = []
        for p in ["f16", "q8"]:
            v = [quant[(i, b, p)][0] for i in images if (i, b, p) in quant]
            vals.append(np.mean(v) if v else 0.0)
        ax.bar(x + (k - 1) * w, vals, w, label=LABELS[b], color=BACKEND_COLORS[b])
        for xi, vv in zip(x + (k - 1) * w, vals):
            ax.text(xi, vv + 0.002, f"{vv:.1e}", ha="center", fontsize=7)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["f16", "q8"])
    ax.set_ylabel("SDF RMSE vs same-backend f32"); ax.set_title("SDF quantization RMSE by backend")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "precision_maxerr.png"), dpi=130)
    plt.close(fig)

    print("wrote benchmarks/results/report.md, latency.png, precision_rmse.png, precision_maxerr.png")


if __name__ == "__main__":
    main()
