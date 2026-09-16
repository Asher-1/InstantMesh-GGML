# InstantMesh-GGML End-to-End Benchmark

Test: `image → DINO → LRM → OSGDecoder → FlexiCubes → mesh`, grid_res=88 (same as the PyTorch reference).
Backends: ggml CPU / CUDA / Vulkan (RTX 3060); accuracy reference: f32-CUDA.

| Model file | f32 (MB) | f16 (MB) | q8 (MB) | f16/f32 | q8/f32 |
|---|---|---|---|---|---|
| dino | 459.0 | 230.1 | 123.4 | 0.50x | 0.27x |
| lrm_transformer | 1054.8 | 527.8 | 284.0 | 0.50x | 0.27x |
| synthesizer | 0.8 | 0.4 | 0.3 | 0.50x | 0.32x |

## Inference latency (seconds, one end-to-end run per image)

| Backend | f32 | f16 | q8 |
|---|---|---|---|
| CPU | 21.66 | 33.03 | 24.80 |
| CUDA | 3.93 | 3.24 | 3.04 |
| Vulkan | 4.35 | 3.92 | 3.77 |

Speedup (f32): CUDA vs CPU `5.5x`; Vulkan vs CPU `5.0x`.

## SDF accuracy

SDF magnitude ≈ mean|sdf| (~8.5 per image); RMSE below is accompanied by the relative error (RMSE/mean|sdf|).

### A. Relative to the f32-CUDA reference (absolute error, including backend numeric differences)

| Backend | Precision | RMSE | Relative RMSE | max-abs | Sign-flip rate (%)
|---|---|---|---|---|---
| CPU | f32 | 5.37e-02 | 0.64% | 4.00e+00 | 3.92e-02 |
| CPU | f16 | 5.84e-02 | 0.70% | 3.79e+00 | 4.04e-02 |
| CPU | q8 | 2.46e-01 | 2.91% | 4.71e+00 | 7.16e-01 |
| CUDA | f32 | 0.00e+00 | 0.00% | 0.00e+00 | 0.00e+00 |
| CUDA | f16 | 6.81e-02 | 0.81% | 4.71e+00 | 9.36e-02 |
| CUDA | q8 | 2.39e-01 | 2.83% | 4.98e+00 | 7.20e-01 |
| Vulkan | f32 | 6.08e-02 | 0.72% | 3.70e+00 | 4.27e-02 |
| Vulkan | f16 | 7.16e-02 | 0.85% | 4.59e+00 | 9.36e-02 |
| Vulkan | q8 | 1.67e-01 | 1.97% | 4.72e+00 | 5.42e-01 |

### B. Quantization error (same backend, vs its own f32) — isolating pure quantization loss

| Backend | Precision | RMSE | Relative RMSE |
|---|---|---|---|
| CPU | f16 | 1.56e-02 | 0.18% |
| CPU | q8 | 2.42e-01 | 2.86% |
| CUDA | f16 | 6.81e-02 | 0.81% |
| CUDA | q8 | 2.39e-01 | 2.83% |
| Vulkan | f16 | 6.46e-02 | 0.76% |
| Vulkan | q8 | 1.69e-01 | 2.00% |

### C. Backend consistency (f32 vs CUDA-f32 difference)

| Backend | RMSE | Relative RMSE |
|---|---|---|
| CPU | 5.37e-02 | 0.64% |
| CUDA | 0 (reference) | 0 |
| Vulkan | 6.08e-02 | 0.72% |

> Note: SDF magnitude is ~8.5, so absolute RMSE looks large; the relative RMSE is more intuitive.
> Pure quantization error (table B) is as expected: f16≈0.2–0.8% relative RMSE, q8≈2–3% relative RMSE, consistent across CPU/CUDA/Vulkan.
> Backend consistency (table C) reflects floating-point accumulation-order differences across backends — normal numeric noise.

## Per-image details

### blue_cat

| Backend | Precision | Latency (s) | Relative RMSE (vs f32-CUDA) |
|---|---|---|---|
| CPU | f32 | 25.31 | 0.50% |
| CPU | f16 | 35.41 | 0.57% |
| CPU | q8 | 22.68 | 2.79% |
| CUDA | f32 | 3.74 | 0.00% |
| CUDA | f16 | 3.14 | 0.67% |
| CUDA | q8 | 3.04 | 2.67% |
| Vulkan | f32 | 4.54 | 0.61% |
| Vulkan | f16 | 3.95 | 0.73% |
| Vulkan | q8 | 4.11 | 1.68% |

### cute_horse

| Backend | Precision | Latency (s) | Relative RMSE (vs f32-CUDA) |
|---|---|---|---|
| CPU | f32 | 18.16 | 0.85% |
| CPU | f16 | 31.35 | 0.90% |
| CPU | q8 | 25.43 | 3.38% |
| CUDA | f32 | 4.43 | 0.00% |
| CUDA | f16 | 3.28 | 1.21% |
| CUDA | q8 | 2.98 | 3.06% |
| Vulkan | f32 | 4.34 | 0.94% |
| Vulkan | f16 | 3.82 | 1.05% |
| Vulkan | q8 | 3.57 | 2.40% |

### fox

| Backend | Precision | Latency (s) | Relative RMSE (vs f32-CUDA) |
|---|---|---|---|
| CPU | f32 | 19.41 | 0.50% |
| CPU | f16 | 31.09 | 0.53% |
| CPU | q8 | 23.06 | 2.65% |
| CUDA | f32 | 3.88 | 0.00% |
| CUDA | f16 | 3.33 | 0.59% |
| CUDA | q8 | 3.05 | 2.68% |
| Vulkan | f32 | 4.17 | 0.59% |
| Vulkan | f16 | 3.85 | 0.71% |
| Vulkan | q8 | 3.73 | 1.82% |

### robot

| Backend | Precision | Latency (s) | Relative RMSE (vs f32-CUDA) |
|---|---|---|---|
| CPU | f32 | 23.75 | 0.71% |
| CPU | f16 | 34.27 | 0.78% |
| CPU | q8 | 28.04 | 2.82% |
| CUDA | f32 | 3.68 | 0.00% |
| CUDA | f16 | 3.20 | 0.76% |
| CUDA | q8 | 3.09 | 2.93% |
| Vulkan | f32 | 4.33 | 0.75% |
| Vulkan | f16 | 4.07 | 0.91% |
| Vulkan | q8 | 3.68 | 2.00% |

## Render comparison

Mesh reconstruction comparison (rows = backends, columns = precisions, top-left cell is the input image):

- `render_blue_cat.png`: blue_cat
- `render_cute_horse.png`: cute_horse
- `render_fox.png`: fox
- `render_robot.png`: robot

> The f32-CUDA cell is the PyTorch-equivalent reference (per-component parity ~1e-5).

## Real PyTorch-CUDA reference comparison

Using the same InstantMesh-large weights as the upstream `run.py`, PyTorch-CUDA runs directly on **exactly the same**
multi-view inputs (`benchmarks/mv/*/image.bin` + `camera.bin`), and `extract_mesh(use_texture_map=False)`
also uses the synthesizer's `net_rgb` branch for vertex colors.
- Weights: same large model (instant-mesh-large), **weight-level alignment**.
- Geometry mesh: the local GPU (11.7GB) cannot fit the full grid_res=128 FlexiCubes mesh on the PyTorch side (~15GB needed),
  so `grid_res` is lowered to 88; color (`net_rgb`) is independent of mesh resolution, so **color/appearance is directly comparable**,
  while the ggml mesh is finer.

| Image | PyTorch verts | ggml(f16) verts | PyTorch latency (s) | ggml CUDA f16 latency (s) |
|---|---|---|---|---|
| blue_cat | 14550 | 14558 | 1.37 | 3.14 |
| cute_horse | 19354 | 19390 | 1.42 | 3.28 |
| fox | 14208 | 14240 | 1.34 | 3.33 |
| robot | 18068 | 18122 | 1.35 | 3.20 |

Vertex color comparison (mean RGB; the closer, the more consistent):

| Image | PyTorch mean RGB | ggml(f16) mean RGB | Vertex color Δmean |
|---|---|---|---|
| blue_cat | [0.28  0.358 0.445] | [0.282 0.36  0.447] | 0.002 |
| cute_horse | [0.611 0.671 0.722] | [0.612 0.673 0.725] | 0.002 |
| fox | [0.622 0.434 0.313] | [0.624 0.435 0.315] | 0.002 |
| robot | [0.674 0.541 0.393] | [0.676 0.543 0.395] | 0.002 |

See `pytorch_vs_ggml_<img>.png` for visuals (input | PyTorch | ggml).

## Input alignment between ggml and PyTorch

ggml and PyTorch use **exactly the same multi-view inputs** (the same `image.bin`/`camera.bin`, generated by Zero123++,
the same `default_cameras()` camera convention), and every component (DINO/LRM/OSGDecoder/FlexiCubes/net_rgb) passes
parity tests against PyTorch (f32 relative error ~1e-5). Remaining differences therefore come from: quantization precision (f32/f16/q8),
backend floating-point accumulation order, and the lower mesh resolution used by this report's PyTorch reference due to VRAM limits.

**Troubleshooting record**: an earlier PyTorch reference re-normalized the already ImageNet-normalized `image.bin` through
ViTImageProcessor (double normalization), pushing the reference triplane/mesh/colors far from ggml (vertex color Δmean~0.25).
After fixing it to feed DINO the same normalized tensor directly, the vertex color Δmean dropped to ~0.002 and the vertex count matches ggml.
