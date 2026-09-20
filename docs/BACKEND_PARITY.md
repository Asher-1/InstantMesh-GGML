# Backend Parity Report — CUDA / Vulkan precision alignment & E2E acceptance

**Acceptance window**: 2026-09-19 ~ 09-20　**baseline commit**: `69f82b3` (pushed to origin/main)
**Hardware**: RTX 3060 12GB　**build**: `cpp_ggml/build-gpu` (CUDA + Vulkan dual backend)
**Upstream ggml**: v0.21.0 (`8599e0ea`) + `patches/` (2 files)

---

## 1. Precision fixes: three CUDA floating-point traps

All root-caused by measurement (with bidirectional env-switch verification). Unified opt-out switch `GGML_CUDA_TF32=1` (bit-exact reproduction of the old behavior).

| # | Trap | Root cause | Hit surface | After fix |
|---|---|---|---|---|
| 1 | cuBLAS TF32 | `common.cuh` unconditionally set `CUBLAS_TF32_TENSOR_OP_MATH`, cutting f32 matmuls to 10-bit mantissas | All batch=1 f32 matmuls; CLIP 1.5e-3 (CPU 7.6e-6) | clip_vision **1.24e-5 PASS** |
| 2 | Ampere fp32 MMA = TF32 | `mmf.cu should_use_mmf` F32 branch admitted tensor-core MMA | **Thin layers only** (src1_ncols<=16): VAE conv_out/quant_conv (4ch output), synthesizer MLP heads (N=1/3) | test_vae cuda encode **1.47e-4 PASS** (was 5.97e-2) |
| 3 | flash_attn fp16 KV | f32 inputs hard-converted to fp16 in-kernel (upstream performance design; CUDA fattn ignores the prec parameter) | All attention layers | Known behavior, kept (see §4) |

Trap #2's stealth: that path bypasses cuBLAS entirely, so the math-mode fix (#1) and `GGML_CUDA_FORCE_CUBLAS` are both blind to it — only per-stage bisection (§5 tooling) can locate it.

**Related finding**: `zero123pp`'s VAE decode post-processing read the NCHW channel-plane output as interleaved HWC — the grid PNG came out as grey shuffled tiles (13.8dB), while latents-level acceptance was all green and the C++-internal cross-backend grid comparison was also all green (same-mistake cancellation). After fixing the read order: **62.2dB, pixel-identical**.
**Lesson**: cross-implementation acceptance must include at least one pixel-level/visual check; pure tensor statistics can pass while the image is completely wrong.

## 2. End-to-end consistency matrix

zero123pp full chain (encode → 75-step diffusion → VAE decode); free-run uses the same seed and RNG; the fixture-replay rows use the official torch per-step fixture (`benchmarks/fixtures/e2e/cute_horse`).

| Path | Comparison | final latents | grid.png |
|---|---|---|---|
| f32 free-run | cuda vs vulkan | **77.00 dB** (max 2.1e-2 / mean 2.6e-4) | **57.23 dB** |
| f16 free-run | cuda vs vulkan | **68.76 dB** (f16 quantization level) | **54.59 dB** |
| f32 fixture replay | both backends vs torch | 51.6–54.6 dB | 49.3–50.2 dB |
| f32 decode(torch latents) | C++ vs torch grid | — | **62.2 dB** |

- Cross-implementation bit-exactness is physically unreachable (floating-point summation order); the table above is the strongest provable equivalence.
- The max outliers (0.6–0.8) in fixture replay are the inherent floor of the torch fp16 fixture's representation noise amplified chaotically over ancestral sampling; impact on the image <0.31 pixel.
- zero123pp has no q8 weights; geometry q8 is covered by the 36-combination baseline (the mmq quantized branch is untouched by this round of fixes).

## 3. Geometry pipeline per-stage noise decomposition (f32 weights, blue_cat grid64)

| Stage | cuda vs cpu | vulkan vs cpu | Notes |
|---|---|---|---|
| dino feats (12 attn layers) | 90.31 dB | 95.69 dB | cuda flash_attn fp16 KV, 5.4dB gap |
| LRM triplane (same feats input, isolated) | 66.72 dB | 67.95 dB | dominant contributor of the planes gap |
| planes full chain | 66.82 dB | 67.89 dB | ≈ LRM stage (dino input delta drowned out) |
| synthesizer (same planes input, isolated) | **132.46 dB** | — | pure MLP, bit-level, no attention |
| SDF full chain | 53.71 dB | 59.79 dB | planes delta × MLP gain (~3.3× MSE) |

**Conclusions**:
- mmf fix benefit: planes **+6.5dB**, SDF **+2.8dB**; zero impact on vulkan (its f32 matmuls always went through the exact shader).
- The ~6dB cross-backend SDF gap = planes input delta (flash_attn fp16 KV) amplified by the MLP gain; **no optimizable op inside the synthesizer**.
- flash_attn fp16 KV is an upstream performance design (tensor cores are mandatory); not worth chasing (see §4).

## 4. flash_attn f32-KV switch evaluation — verdict: not worth it

Measured on cuda (`IM_FLASH_F32=1`):
- **Zero accuracy gain**: CUDA fattn ignores `GGML_PREC_F32` (no prec read anywhere in the source; K/Q are hard-converted to fp16 in-kernel for MMA) — dino 90.31dB / lrm 66.72dB / SDF 53.71dB are **bit-identical** with and without the gate.
- **Zero performance loss**: the F32KV path runs at the same speed (dino 3.84 vs 3.91s, lrm 4.83s flat).
- True f32 attention would require falling back to the manual softmax path (large performance loss) for a ~6dB visual-indistinguishable SDF gain.

**Decision**: do not chase. The `IM_FLASH_F32` gate stays in the code (`dino.cpp` / `lrm_transformer.cpp`, no-op by default) as a hook for the day upstream ggml honors prec on the CUDA fattn path.

## 5. Tooling & infrastructure

| Tool | Purpose |
|---|---|
| `scripts/vae_encode_bisect.sh [cuda\|vulkan]` | IM_VAE_DUMP per-stage bisection (CPU reference vs GPU; flags the first diverging stage) |
| `zero123pp --latents-in <bin>` | skip diffusion and decode external raw latents (isolates the decode stage) |
| `GgufModel::unload()` | stage weight unloading; zero123pp f16 peak 11593→8192 MiB, f32 VAE decode OOM fixed (latents bit-exact verified) |
| `clean_stale_baselines.sh` | stale baseline cleanup (was swallowed by the global `*.sh` ignore rule; now tracked via a negated pattern) |

## 6. Commit record

Main-repo main (all `ludahai19@163.com`, pushed to origin/main):

```
a3dc966 update                                                        09-16
f7fa087 update                                                        09-18
cd30b4e fix: exact FP32 CUDA matmul, zero123pp VAE decode layout,
         stage weight unloads                                          09-19
5ee1ef2 test: add scripts/vae_encode_bisect.sh (IM_VAE_DUMP per-stage
         bisection)                                                    09-19
7077b51 fix: route thin F32 matmuls away from Ampere TF32 MMA
         (VAE encode conv_out)                                         09-19
69f82b3 chore: bump RMBG submodule (BackendOptions explicit API)      09-20
```

Submodules:
- `RMBG-2.0-GGML`: `420c59d` (BackendOptions explicit-API refactor, dev branch pushed to origin/dev; main-repo gitlink in sync)
- `ggml`: parked at upstream `v0.21.0` (`8599e0ea`); worktree changes fully covered by `patches/ggml-rmbg-ops.patch` + `patches/ggml-cuda-f32-matmul-exact.patch` (empty comm difference; CMake re-applies them on a fresh clone)

## 7. Leftovers & follow-ups

- flash_attn fp16 KV: once upstream ggml's CUDA fattn honors prec, `IM_FLASH_F32=1` enables it (hook already in place)
- synthesizer f32 single allocation >6GB at grid88: 12GB cards need grid64 or the unload optimization
- For higher geometry parity: evaluate whether the cuda/vulkan flash_attn behavior difference can converge via upstream prec support
