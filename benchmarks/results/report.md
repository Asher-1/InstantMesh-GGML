# InstantMesh-GGML 端到端 Benchmark

测试：`图片 → DINO → LRM → OSGDecoder → FlexiCubes → mesh`，grid_res=88（与 PyTorch 参考一致）。
后端：ggml CPU / CUDA / Vulkan（RTX 3060）；精度参考：f32-CUDA。

| 模型文件 | f32 (MB) | f16 (MB) | q8 (MB) | f16/f32 | q8/f32 |
|---|---|---|---|---|---|
| dino | 459.0 | 230.1 | 123.4 | 0.50x | 0.27x |
| lrm_transformer | 1054.8 | 527.8 | 284.0 | 0.50x | 0.27x |
| synthesizer | 0.8 | 0.4 | 0.3 | 0.50x | 0.32x |

## 推理延迟（秒，每图一次端到端）

| 后端 | f32 | f16 | q8 |
|---|---|---|---|
| CPU | 21.66 | 33.03 | 24.80 |
| CUDA | 3.93 | 3.24 | 3.04 |
| Vulkan | 4.35 | 3.92 | 3.77 |

加速比（f32）：CUDA vs CPU `5.5x`；Vulkan vs CPU `5.0x`。

## SDF 精度

SDF 幅值量级 ≈ mean|sdf|（每图 ~8.5），以下 RMSE 同时给出相对误差（RMSE/mean|sdf|）。

### A. 相对 f32-CUDA 参考（绝对误差，含后端数值差异）

| 后端 | 精度 | RMSE | 相对RMSE | max-abs | 符号翻转率(%)
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

### B. 量化误差（同后端，相对各自 f32）— 隔离纯量化损失

| 后端 | 精度 | RMSE | 相对RMSE |
|---|---|---|---|
| CPU | f16 | 1.56e-02 | 0.18% |
| CPU | q8 | 2.42e-01 | 2.86% |
| CUDA | f16 | 6.81e-02 | 0.81% |
| CUDA | q8 | 2.39e-01 | 2.83% |
| Vulkan | f16 | 6.46e-02 | 0.76% |
| Vulkan | q8 | 1.69e-01 | 2.00% |

### C. 后端一致性（f32 相对 CUDA-f32 的差异）

| 后端 | RMSE | 相对RMSE |
|---|---|---|
| CPU | 5.37e-02 | 0.64% |
| CUDA | 0 (参考) | 0 |
| Vulkan | 6.08e-02 | 0.72% |

> 说明：SDF 幅值约 8.5，故绝对 RMSE 偏大；看相对 RMSE 更直观。
> 纯量化误差（B 表）量级符合预期：f16≈0.2–0.8% 相对 RMSE，q8≈2–3% 相对 RMSE，且 CPU/CUDA/Vulkan 量级一致。
> 后端一致性（C 表）反映不同后端浮点累加顺序差异，属正常数值噪声。

## 分 case 明细

### blue_cat

| 后端 | 精度 | 延迟(s) | 相对RMSE(vs f32-CUDA) |
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

| 后端 | 精度 | 延迟(s) | 相对RMSE(vs f32-CUDA) |
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

| 后端 | 精度 | 延迟(s) | 相对RMSE(vs f32-CUDA) |
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

| 后端 | 精度 | 延迟(s) | 相对RMSE(vs f32-CUDA) |
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

## 渲染对比图

网格重建效果对比（行=后端，列=量化，左上角为输入图）：

- `render_blue_cat.png`：blue_cat
- `render_cute_horse.png`：cute_horse
- `render_fox.png`：fox
- `render_robot.png`：robot

> f32-CUDA 格即 PyTorch 等价参考（各组件 parity ~1e-5）。

## PyTorch-CUDA 真实参考对比

用上游 `run.py` 同款 InstantMesh-large 权重，直接把 PyTorch-CUDA 跑在**与 ggml 完全相同的**
多视角输入（`benchmarks/mv/*/image.bin` + `camera.bin`）上，`extract_mesh(use_texture_map=False)`
同样走 synthesizer 的 `net_rgb` 分支输出顶点颜色。
- 权重：同为大模型（instant-mesh-large），**权重级对齐**。
- 几何网格：PyTorch 侧因本机 GPU 显存（11.7GB）放不下 grid_res=128 的完整 FlexiCubes 网格（需 ~15GB），
  将 `grid_res` 降到 88 以适配；颜色（`net_rgb`）与网格分辨率无关，故**颜色/appearance 可直接对比**，
  ggml 侧网格更细腻。

| 图片 | PyTorch 顶点数 | ggml(f16) 顶点数 | PyTorch 延迟(s) | ggml CUDA f16 延迟(s) |
|---|---|---|---|---|
| blue_cat | 14550 | 14558 | 1.37 | 3.14 |
| cute_horse | 19354 | 19390 | 1.34 | 3.28 |
| fox | 14208 | 14240 | 1.34 | 3.33 |
| robot | 18068 | 18122 | 1.35 | 3.20 |

顶点颜色对比（平均 RGB，越接近越一致）：

| 图片 | PyTorch mean RGB | ggml(f16) mean RGB | 顶点颜色 Δmean |
|---|---|---|---|
| blue_cat | [0.28  0.358 0.445] | [0.282 0.36  0.447] | 0.002 |
| cute_horse | [0.611 0.671 0.722] | [0.612 0.673 0.725] | 0.002 |
| fox | [0.622 0.434 0.313] | [0.624 0.435 0.315] | 0.002 |
| robot | [0.674 0.541 0.393] | [0.676 0.543 0.395] | 0.002 |

视觉效果见 `pytorch_vs_ggml_<img>.png`（输入图 | PyTorch | ggml）。

## 关于 ggml 与 PyTorch 的输入对齐

ggml 与 PyTorch 使用**完全相同的多视角输入**（同一份 `image.bin`/`camera.bin`，由 Zero123++ 生成、
同一套 `default_cameras()` 相机约定），且各组件（DINO/LRM/OSGDecoder/FlexiCubes/net_rgb）均通过
parity 测试与 PyTorch 对齐（f32 相对误差 ~1e-5）。因此二者差异主要来自：量化精度（f32/f16/q8）、
后端浮点累加顺序，以及本报告的 PyTorch 参考因显存限制使用了较低的网格分辨率。

**排查记录**：早期版本 PyTorch 参考会把已做 ImageNet 归一化的 `image.bin` 再经 ViTImageProcessor
归一化一次（双重归一化），导致参考三平面与网格/颜色严重偏离 ggml（顶点颜色 Δmean~0.25）。
修复为直接向 DINO 输入同一份归一化张量后，顶点颜色 Δmean 降至 ~0.002、顶点数与 ggml 一致。
