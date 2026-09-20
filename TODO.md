# InstantMesh-GGML Alignment Backlog

> C++ ↔ official Python feature-alignment status, blockers, and next actions
>
> Generated 2026-09-07 · Updated 2026-09-17 (UNet/Pipeline ✅, Vulkan full-precision regression in progress)

---

## Overview

| # | Item | Status | ctest gate |
|---|---|---|---|
| 1 | Zero123++ scheduler | ✅ done | `test_scheduler` Passed (75-step 1.9e-6) |
| 2 | Zero123++ CLIPVision | ✅ done | `test_clip_vision` Passed (7.6e-6) |
| 3 | Zero123++ weight conversion chain | ✅ done (3 GGUF files produced) | — |
| 4 | Zero123++ VAE encode/decode | ✅ done (encode 2.4e-4 / decode 3.0e-3) | `test_vae` Passed (ungated) |
| 5 | UNet + RefOnly attention | ✅ done (f32 7.1e-4 / f16 8.8e-4) | `test_unet` Passed |
| 6 | Denoising pipeline / zero123pp CLI | 🔨 assembled (`src/tools/zero123pp.cpp`); E2E PSNR acceptance pending | none (`--fixture-dir` supported) |
| 7 | rembg | ✅ done | verified on blue_cat |
| 8 | NeRF variant + NeuralRender | ❌ not started | none |
| 9 | `--save_video` | ❌ not started | none |
| 10 | `--export_texmap` baking | ✅ aligned (xatlas UV + multi-view baking) | `test_texture_map` Passed |
| 11 | PBR textures | ➖ N/A (no PBR path in official InstantMesh code, nothing to align) | — |
| 12 | ggml layout-contract regression probe | ✅ done | `test_conv_layout` Passed |
| 13 | Vulkan full-precision per-layer regression | 🔨 in progress (first round done: see §7) | cuda/vulkan run separately once tests take a device arg |

---

## 1. Zero123++ Pipeline — UNet/assembly converged, E2E acceptance remains

### Completed parts

- **scheduler** (`cpp_ggml/src/models/scheduler.cpp`): EulerAncestral, **linear beta + v_prediction + trailing** (matches this repo's scheduler config, not the usual SD convention); the `alphas_cumprod` curve ships inside the GGUF to sidestep torch f32 cumprod rounding non-reproducibility. Parity: timesteps bit-exact, sigmas 9.5e-7, full 75-step walk 1.9e-6.
- **CLIPVision** (`cpp_ggml/src/models/clip_vision.cpp`): ViT-L/14@1280→1024, manual softmax attention (seq=257); fixed the missing q/k/v bias, the historic `pre_layrnorm` misspelling position, etc. Parity 7.6e-6.
- **Conversion chain** (`cpp_ggml/convert/convert_zero123pp.py`): `zero123pp_unet_f16` (1.73GB) / `vae_f16` / `vae_f32` / `cond_f32` (with text_emb[77,1024], negative_lat, ramp[65], alphas_cumprod[1000]) all in place.
- **VAE** (converged 2026-09-08): encode mode max_abs=2.36e-4 (f32 reordering-noise level), decode max_abs=3.0e-3 (f16 im2col floor, ≈50dB PSNR).

### VAE convergence log (five consecutive root causes, all probe/ confronted-evidence confirmed)

1. **"ne lying" was a misdiagnosis**: a contiguous ggml tensor's ne+nb+memory triple is necessarily self-consistent. v0.21 `conv_2d` output is a clean contiguous `[W,H,OC,N]` (memory = torch NCHW). `tests/test_conv_layout.cpp` (in-process naive reference) pins the contract.
2. **gallocr reuse interference**: dump tensors with no consumers get recycled as soon as their direct consumer runs (NOT transitive!). Fix: always `ggml_set_output()` on the read side (ggml-alloc.c: outputs are never recycled).
3. **Missing GN affine**: the mid attention's `group_norm.weight/bias` was not multiplied/added before.
4. **Asymmetric downsampling**: diffusers Downsample2D (`ds.padding=0`) = `F.pad(x,(0,1,0,1))` bottom/right zero-pad first, then a stride-2 **padding-0** conv; `ggml_pad` args are post-ne padding amounts, so `(1,1,0,0)`.
5. **f16 im2col floor**: `ggml_conv_2d` hardcodes an f16 im2col (~f16 eps per conv, accumulating to 1e-1 through the deep stack). encode switched to `conv_f32()` (manual im2col F32 + mul_mat, 2x memory) bringing it back to 2.4e-4; decode keeps f16 (already in spec).

### Next steps (E2E acceptance)

1. **UNet(RefOnly)** ✅ (converged 2026-09-16): time-emb MLP, ResBlock (GN/SiLU/conv+skip), self+cross attention (cross_attention_dim=1024, use_linear_projection=true, heads 8×head_dim 40), down/mid/up + RefOnly w/r double forward + ramping scale; f32 7.1e-4 / f16 8.8e-4 (synthetic fixture, torch per-layer reference).
2. **pipeline assembly** ✅: `src/tools/zero123pp.cpp` (rembg→VAE encode(cond)→CLIPVision→75-step double forward→cfg 4.0→VAE decode→3×2 grid), supports `--fixture-dir` (replays official same-seed noise) / `--dump-final-latents` / `--dump-steps`.
3. **E2E PSNR acceptance** (current): produce official per-step fixtures with `convert/dump_e2e.py` (r/w input, eps, latents, context, noise), replay via C++ `--fixture-dir` → compare `ref_latents.bin` and the 6-view PSNR; plan in `docs/ALIGNMENT.md` acceptance criteria.

---

## 2. Denoising Pipeline / zero123pp CLI — assembled, pending E2E PSNR acceptance

- **Status**: the official information flow is fully documented in `docs/ALIGNMENT.md`; `src/tools/zero123pp.cpp` implements `rembg → VAE encode(cond) → CLIPVision → 75-step double forward (RefOnly) → cfg 4.0 → VAE decode → unscale → 3×2 grid split into 6 views`.
- **Next step**: generate official same-seed fixtures with `convert/dump_e2e.py` → replay `zero123pp --fixture-dir ... --dump-final-latents` → per-step/final latents and 6-view PSNR acceptance (criteria listed).

---

## 4. NeuralRender / NeRF variant — not started

- **Status**: C++ has only the Mesh/FlexiCubes path; `cpp_ggml/convert/convert_lrm.py` has no nerf support (the `instant-nerf-large.yaml` synthesizer head is not exported).
- **Blockers**: none; independent of the Zero123++ line, can proceed in parallel.
- **Next actions**:
  1. extend `convert_lrm.py` with the NeRF variant head export (ray_sampler/ray_marcher MLP weights);
  2. host-side ray marcher (numerically dense, pinned to CPU for consistency) + MarchingCubes LUT generation (reusing the FlexiCubes table generation approach);
  3. e2e `--config instant-nerf-large` parity (acceptance criteria listed).

---

## 5. `--save_video` PNG frame sequence — not started

- **Status**: no implementation; official is `model.render_video`, 36 frames around Y (render distance=4.5/2.0 scale) + mp4.
- **Technical decision made**: C++ software z-buffer rasterizer → `videos/<name>/frame_%03d.png`, mp4 assembled by external ffmpeg.
- **Blockers**: none; suggested after the pipeline converges (reuse the existing canonical_render/texture-render benchmark assets for comparison).
- **Next steps**: rasterizer (vertex-color + textured paths), orbit camera aligned with the official trajectory (36 frames), per-frame PSNR comparison against official video frames.

---

## 6. Other feature deltas

| Item | Status | Notes |
|---|---|---|
| PBR textures | ➖ no alignment needed | official code has no metallic/roughness path (audited) |
| Baking `--export_texmap` | ✅ aligned | xatlas UV unwrap + multi-view baking, same as official |
| rembg | ✅ aligned | RMBG-2.0 vendor (better quality than official u2net) |
| DINO / LRM / FlexiCubes / synthesizer | ✅ aligned | existing baselines 7/7 green |
| VaeImageProcessor.postprocess | ❌ with the pipeline | pure numerics, no risk |
| text encoder | ➖ no port needed | empty-prompt constant precomputed into the GGUF at conversion time |

---

## Suggested execution order

1. ~~VAE numerical convergence~~ ✅ (2026-09-08, see §1 log);
2. ~~UNet(RefOnly) + pipeline assembly~~ ✅ (2026-09-16/17) → in progress: `zero123pp` e2e PSNR acceptance (`dump_e2e.py` + `--fixture-dir`);
3. **Vulkan full-precision per-layer regression** (in progress): dual-GPU-backend build + ctest cuda/vulkan split runs + staged-dump per-layer comparison;
4. Parallel tracks: NeRF variant conversion + ray marcher; `--save_video` rasterizer;
5. Final e2e: `instantmesh --image x.png --rmbg rmbg.gguf` aligned with `python run.py` on same input/output (acceptance criteria in `docs/ALIGNMENT.md`).

---

## 7. Vulkan full-precision per-layer regression (first round 2026-09-17)

- **Infrastructure**: `backend.cpp` supports explicit device names (`cuda`/`vulkan` prefix match); `test_unet/vae/clip_vision` take an argv[1] device arg (default cpu); `build-gpu` (CUDA+Vulkan) build script in `cpp_ggml/README.md` §3 Route C. Multiple local CUDA coinstalls (11.1/11.4/11.8) create a libcudart version ambiguity at exe link time; needs `CMAKE_EXE_LINKER_FLAGS` pointing at 11.8 explicitly (this machine only).
- **Bugs fixed**: UNet GEGLU passed a non-contiguous view to `ggml_gelu` → CUDA `unary.cu` asserts `ggml_is_contiguous` fails (CPU/Vulkan tolerate); added `ggml_cont`, CUDA UNet went from crash → 9.4e-4 PASS.
- **Numeric matrix (vs torch fixture, f32 weight path)**:

  | Component | CPU | CUDA | CUDA(TF32 off) | Vulkan |
  |---|---|---|---|---|
  | CLIPVision | 7.6e-6 ✅ | 1.5e-3 ❌ | 1.2e-5 ✅ | 1.5e-3 ❌ |
  | VAE encode | 2.4e-4 ✅ | 6.0e-2 ❌ | 1.45e-2 ❌ | 1.17e-1 ❌ |
  | VAE decode | 3.0e-3 ✅ | 3.5e-3 ✅ | 3.1e-3 ✅ | 1.7e-2 ❌ |
  | UNet(refonly) | 7.1e-4 ✅ | crash→fixed | 9.4e-4 ✅ | 2.5e-2 ✅(0.35) |

  (This is the 09-17 snapshot; the 09-19/20 TF32/mmf fixes supersede the ❌ entries — see docs/BACKEND_PARITY.md.)

- **VAE encoder staged-dump (CUDA vs Vulkan per-layer max_abs)**: conv_in_raw 1.4e-4 → rs_conv1 4.3e-3 → down0 4.4e-3 → down1 5.4e-2 → down2 4.1e-1 → mid1 1.76. k_in 0 diff. Conclusion: **monotonically accumulating f32 re-rounding, not a semantic/layout difference**.
- **CLIP staged-dump (CUDA vs Vulkan per-layer max_abs)**: patch/emb 5.2e-5 → layer0 8.5e-4 → layer1 1.2e-3 → last 3.0e-2 → post_ln 6.2e-2. Same pattern as VAE: 24-layer monotonic accumulation (softmax/attention amplification), not a semantic difference; after the 1024-dim projection (token-averaged) the error returns to ~1e-3 (vs torch: cuda 1.2e-5 / vulkan 1.5e-3).
- **CLIP layer0 internals (IM_CV_L0, CUDA vs Vulkan max_abs)**: ln1 4.1e-4 (mean 2.9e-7) → q/k/v 3.5-5.2e-4 → **attn_raw 1.3e-3 (first/largest amplification point, softmax attention path)** → attn_proj 3.6e-4 (drops back after the large matmul averages). **Excludes the convs-fp16-dot hypothesis**: CLIP has no conv (patch embed goes through a manual f32 im2col+mul_mat, see clip_vision.cpp comments) and the encoder is all F32 fma matmuls.
- **The real location of the fp16 dot path**: ggml Vulkan's fp16 mixed dot (`v_dot2_f32_f16`, dot_product_funcs.glsl) only hits on **F16-weight matmuls**; **the conv2d op's shmem is FP16 on coopmat2/cm1 devices** (ggml-vulkan.cpp `conv2d_use_fp16_shmem = coopmat2 || cm1`; RTX3060 hits KHR_coopmat). This affects the **VAE decoder (f16 conv) and RMBG** (candidate main cause of vulkan decode 1.7e-2 vs cuda 3.1e-3), not CLIP/VAE encoder (f32 path).
- **VAE encoder first amplification point**: rs_norm1 (GroupNorm) 1.4e-4 → 1.2e-3, then conv/matmul accumulate monotonically to 1.76. Norm reduction (sum/var) reordering is the amplifier (near-zero output elements get scaled by inv_std, mean still ~1e-6).
- **UNet staged-dump**: `IM_VAE_DUMP=1` on GPU blows the 65536 graph-node capacity with dump nodes and fails gallocr allocation (CPU fine) — UNet per-layer dumps need pruned dump points or a larger graph first; component level is already covered numerically by test_unet (cuda 9.4e-4 / vulkan 2.5e-2).
- **E2E (zero123pp 8-step f16, seed 42)**: CUDA 32.3s / Vulkan 34.8s both run; final latents CUDA vs Vulkan PSNR 45.8dB, grid PSNR 33.5dB — visually consistent.
- **Next step**: set GPU acceptance thresholds for VAE encode/clip per the expected re-rounding (e.g. encode ≤1e-1 f16 path / clip ≤5e-3); or tighten op by op (locate the first Vulkan amplification point).

---

## Constraints & notes

- Always build with `make -j ≤ 6` (avoid OOM);
- **staged dumps must be paired with `ggml_set_output()`** (gallocr recycles tensors without direct consumers, non-transitive!);
- before comparisons clean `rm /tmp/vae_*.bin /tmp/ref_*.bin` and verify file sizes (cross-run contamination);
- torch reference generation uses the `/tmp/vref` venv (transformers 4.52.4 + huggingface_hub 0.36.2 + diffusers 0.39.0); watch two easy mistakes in reference scripts: bare GN output vs GN-with-affine, and diffusers' asymmetric downsampling;
- **Vulkan regression notes**: test device selection via argv[1] (`cuda`/`vulkan`/`cpu`); the Vulkan backend's math differs from CUDA at re-rounding level (f32 should be ~1e-6); thresholds align to the torch fixture, not CUDA bit-level.

---
