# MODEL_CARD — InstantMesh GGUF Inference Model Card

This directory (`cpp_ggml/models/gguf/`) holds the 3 ggml models required by the
InstantMesh end-to-end reconstruction pipeline, each provided as `.gguf` weights in
three precisions: `f32 / f16 / q8`.

**Weight download (Hugging Face): https://huggingface.co/Asher-1/InstantMeshGGuf**

Pipeline and data flow:

```
Image → ① dino (image encoding) → ② lrm_transformer (triplane generation)
     → ③ synthesizer (OSGDecoder voxel decoding: SDF/deform/weight/RGB)
     → ④ flexicubes (SDF → mesh, no weights)
```

> Performance and accuracy data in this card come from `../benchmarks/results/report.md`
> (RTX 3060 / Ryzen 5950X, grid_res=88, same conditions as the PyTorch reference;
> end-to-end includes model loading and mesh export).

---

## 1. Download and Usage

```bash
# All 9 files (~2.7 GB)
huggingface-cli download Asher-1/InstantMeshGGuf --local-dir cpp_ggml/models/gguf

# Or the minimal usable set: the f16 trio (~760 MB)
for f in dino_f16 lrm_transformer_f16 synthesizer_f16; do
  curl -L -o cpp_ggml/models/gguf/$f.gguf \
       https://huggingface.co/Asher-1/InstantMeshGGuf/resolve/main/$f.gguf
done
```

- The three models **must use the same precision** (`dino_f16 + lrm_transformer_f16 +
  synthesizer_f16`); mixing precisions may cause inter-layer numeric mismatches.
- `.gguf` files are not committed (large binaries); the weights contain network
  parameters only, **no preprocessing parameters**. Inputs must be ImageNet-normalized
  per DINOv2 rules (see `../src/tools/instantmesh.cpp` and `../convert/prep_input.py`).

---

## 2. Model Overview

| Model | File prefix | Role | Input | Output |
|---|---|---|---|---|
| **dino** | `dino_*.gguf` | DINOv2 vision Transformer backbone extracting multi-scale patch features | Multi-view RGB tensor [V,3,224,224] (ImageNet-normalized) | Image token sequence |
| **lrm_transformer** | `lrm_transformer_*.gguf` | Reconstruction backbone fusing image features into triplane latents | Image token sequence | triplane [3, C, H, W] |
| **synthesizer** | `synthesizer_*.gguf` | OSGDecoder: samples the triplane at voxel points and decodes | triplane + sample point coordinates | SDF / deformation / FlexiCubes weight / RGB |

**Model details**

- **dino**: splits the input image into patches and encodes them with self-attention
  (flash attention), outputting a token sequence. It determines "how much detail the
  model can see in the image" and is the feature source for all subsequent reconstruction.
- **lrm_transformer**: 16 layers of 1024-dim BasicTransformerBlock (self/cross-attention +
  GELU MLP + adaLN modulation) projecting DINO features into three orthogonal feature
  planes (triplane), implicitly encoding the full object in a compact low-dimensional space.
- **synthesizer**: given sample points, projects them onto the triplanes, concatenates
  the features and decodes via a small MLP the point's **SDF** (signed distance),
  **deformation** (vertex deformation), **weight** (FlexiCubes weights) and **RGB**
  (`net_rgb` branch, shared by vertex colors and texture baking).

The three are chained: without dino there are no image features, without
lrm_transformer no triplane, without synthesizer no SDF/mesh/colors.

---

## 3. File Size and Memory Footprint

| Model | f32 (MB) | f16 (MB) | q8 (MB) | f16/f32 | q8/f32 |
|---|---|---|---|---|---|
| dino | 459.0 | 230.1 | 123.4 | **0.50x** | **0.27x** |
| lrm_transformer | 1054.8 | 527.8 | 284.0 | **0.50x** | **0.27x** |
| synthesizer | 0.8 | 0.4 | 0.3 | **0.50x** | **0.32x** |
| **All three models** | **1514.6** | **758.3** | **407.7** | 0.50x | 0.27x |

- f16 is always half of f32 and q8 about a quarter, matching the theoretical
  quantization size ratios.
- Size is **dominated by lrm_transformer** (~1 GB at f32): it drives most of the
  end-to-end memory/load time. Measured ggml CUDA f16 end-to-end peak memory is
  **1011 MB** (PyTorch under the same conditions: 3725 MB); an RTX 3060 with 12 GB
  VRAM runs comfortably at grid_res=88.

---

## 4. Performance (end-to-end, seconds per image)

| Backend | f32 | f16 | q8 | Notes |
|---|---|---|---|---|
| **CUDA** (RTX 3060) | 3.5 | **3.0** | **2.9** | Recommended; **4.5x** faster than PyTorch CUDA (13.5s f16) |
| **Vulkan** | 4.0 | 3.5 | 3.4 | Preferred path for cross-vendor GPUs (AMD/Intel) |
| **CPU** (Ryzen 5950X) | 24.8 | 24.4 | 23.5 | Usable without a GPU; **avoid f16** |
| PyTorch CUDA (official) | 13.7 | 13.5 | — | Reference baseline |

- On GPUs, **q8 is the fastest**, f16 next; CUDA is ~10–15% faster than Vulkan.
- **On CPU, f16 is actually the slowest/unstable**: caused by the CPU-side f16→f32
  dequantization overhead; choose f32 or q8 on CPU.
- Per-stage timing breakdown (ggml CUDA f16): see
  `../benchmarks/results/perf_ggml_stage_breakdown.png`; lrm_transformer dominates,
  dino is next, and synthesizer takes a tiny share.

---

## 5. Accuracy (SDF relative error, mean|sdf|≈8.5, grid_res=88)

### 5.1 Pure quantization error (same backend, vs each backend's own f32)

| Backend | f16 relative RMSE | q8 relative RMSE |
|---|---|---|
| CPU | 0.18% | 2.86% |
| CUDA | 0.81% | 2.83% |
| Vulkan | 0.76% | 2.00% |

### 5.2 Backend consistency (f32 vs CUDA-f32)

| Backend | Relative RMSE |
|---|---|
| CPU | 0.64% |
| Vulkan | 0.72% |

- **f16 is nearly lossless** (relative error <1%), mesh geometry unchanged; **q8 is
  acceptable** (2–3%), SDF sign-flip rate <0.8%, geometry essentially indistinguishable
  to the eye.
- **Across the three backends, same-precision results agree within <1%**: quantization
  and backend implementations are consistent — no "backend silently losing accuracy".

---

## 6. Recommended Use Cases

| Scenario | Recommendation | Rationale |
|---|---|---|
| Everyday inference on NVIDIA GPUs | **CUDA + f16** | <1% accuracy loss, 3.0 s/image, ~1 GB peak memory |
| Maximum speed / batch throughput | CUDA + **q8** | Fastest (2.9 s), smallest size (408 MB for the trio), 2–3% accuracy acceptable |
| AMD / Intel / GPUs without CUDA drivers | **Vulkan + f16** | The only cross-vendor GPU path, 3.5 s/image |
| Tight VRAM (≤4 GB) or mobile | **q8** (lrm_transformer only 284 MB) | Smallest peak memory |
| Server CPU deployment | CPU + **f32 or q8** | Avoids the slow CPU f16 dequantization pitfall |
| Accuracy benchmarking / regression tests / research reproduction | **f32** | Layer-by-layer comparable with the PyTorch reference; serves as the parity reference |
| Validating the pipeline on edge devices | **q8 CPU** | No GPU dependency at all, runs at ~24 s/image |

**Per-model advice**

- **dino**: determines feature quality; generally **not lower than f16**. Use q8 only
  for maximum speed when a slight quality loss is acceptable. Input resolution is
  fixed at 224×224.
- **lrm_transformer**: the largest model, where quantization gains and risks are most
  concentrated; if q8 shows geometric distortion, fall back to f16.
- **synthesizer**: <1 MB, quantization gains are negligible; **prefer f16 or f32
  directly** to avoid pointless accuracy loss; its `plane_dim` must match the
  lrm_transformer output.

---

## 7. Reproduction and Extension

- Batch runner: `bash ../benchmarks/run_bench.sh` (3 backends × 3 precisions × 4 demos,
  producing `times.csv` and per-run `.sdf.bin` / `.obj`)
- Analysis/plots: `python3 ../benchmarks/analyze.py` (generates `report.md`,
  `latency.png`, `precision_rmse.png`, `precision_maxerr.png`)
- Weight conversion (official ckpt → GGUF, three precisions): `python3 -m convert.convert_all`
- Manual inference: see the full commands in [README.md §3 Step 5](../README.md) at the
  module root.
