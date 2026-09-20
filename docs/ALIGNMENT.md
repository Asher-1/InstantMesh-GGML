# Official Feature Alignment Plan (C++/ggml)

Goal: bring the C++ side to full feature parity with the official Python side —
`single image → rembg → Zero123++ multi-view → reconstruction (Mesh/NeRF variants) →
OBJ/texture / orbit video` — with no Python runtime dependency.

Technical decisions (confirmed):
- Zero123++: **hand-written ggml graphs** (same methodology as DINO/LRM: conversion
  scripts + staged parity thresholds).
- rembg: **vendor RMBG-2.0-GGML** (BiRefNet; matches the official rembg/u2net
  capability with higher quality).
- `--save_video`: **C++ software rasterizer emitting a PNG frame sequence** (mp4
  assembled by external ffmpeg).

---

## Phase 0: rembg (done)

- ✅ `third_party/RMBG-2.0-GGML` submodule; ggml patched with
  `patches/ggml-rmbg-ops.patch` (swin / deform-im2col custom ops; originally applied
  cleanly on v0.18.1, auto-applied idempotently by CMake configure).
- ✅ `rmbg_core` static library compiles its `src/*.cpp` directly (deliberately
  bypassing its own CMake's second ggml materialization); stb_image headers come from
  `src/utils/`.
- ✅ CLI `cpp_ggml/build/rembg --model models/gguf/rmbg_f16.gguf --input x.png --out y.png`.
- ✅ Weights `models/gguf/rmbg_{f32,f16,q8}.gguf` (same source as the trellis project;
  its parity already verified).

## Phase 1: Zero123++ Multi-View Diffusion (UNet + pipeline converged; E2E acceptance remains)

Completed components (each gated by a ctest parity test; tests SKIP (77) when
fixtures are absent):

- ✅ **scheduler**: `src/models/scheduler.{hpp,cpp}` — EulerAncestral with
  **linear betas + v_prediction** + trailing spacing (note: the repo's scheduler
  config differs from the usual SD scaled_linear/epsilon pairing). The
  `alphas_cumprod` curve travels with the model in GGUF (torch's float32 cumprod
  rounding is not bit-reproducible in portable C++). Parity: timesteps bit-exact,
  sigmas 9.5e-7, full 75-step walk 1.9e-6 (`test_scheduler`).
- ✅ **CLIPVision**: `src/models/clip_vision.{hpp,cpp}` — ViT-L/14@1280→1024
  projection with manual softmax attention (seq=257, negligible cost; HF's
  q/k/v **have biases**, visual_projection has none, and the embeddings norm
  keeps HF's historic `pre_layrnorm` misspelling). Parity: embeds max_abs
  7.6e-6 (`test_clip_vision`).
- ✅ **conversion**: `convert/convert_zero123pp.py` (diffusers-free: the scheduler
  math and the VAE-encode constant are reimplemented with the same torch ops and
  are bit-exact). Produced `models/gguf/zero123pp_{unet,vae}_f16.gguf` and
  `zero123pp_cond_f32.gguf` (with text_emb[77,1024], negative_lat, ramp[65],
  alphas_cumprod[1000] constants).
- ✅ **UNet + RefOnly**: `src/models/unet.{hpp,cpp}` — time-emb MLP, ResBlock
  (GN/SiLU/conv+skip), self+cross attention (heads 8×head_dim 40,
  cross_attention_dim=1024, linear projection), down/mid/up + RefOnly w/r
  double-forward with `ramping_coefficients` scaling. Parity vs torch staged
  refs: f32 max_abs 7.1e-4, f16 8.8e-4 (`test_unet`, synthetic fixture).
- ✅ **pipeline assembly**: `src/tools/zero123pp.cpp` — rembg → VAE
  encode(cond) → CLIPVision → 75-step double-forward (cfg 4.0) → VAE decode →
  3×2 grid. Supports `--fixture-dir` (replays the official per-step randn),
  `--dump-final-latents`, `--dump-steps` for E2E acceptance.

Official information flow (`zero123plus/pipeline.py` + `run.py`):

```
Input image (RGBA→RGB)
 ├─ feature_extractor_vae (resize 320×320)  ─→ VAE.encode ─→ cond_lat [1,4,40,40]
 │                                            (cfg: encode an all-zero image → negative_lat; constant can be pre-stored)
 ├─ feature_extractor_clip (224×224)        ─→ CLIPVisionModelWithProjection
 │                                            ─→ image_embeds [1,1,1024] × ramping_coefficients
 ├─ CLIP text embedding of an empty prompt [1,77,1024] (constant → precomputed into GGUF at conversion, no text encoder port needed)
 └─ Denoising loop (default 75 steps, EulerAncestralDiscreteScheduler, timestep_spacing='trailing')
      Two UNet forwards per step:
        w-forward: unet(add_noise(cond_lat, noise, t))  collects each layer's self-attn K/V → ref_dict
        r-forward: unet(x_t, t) self-attn concatenates ref K/V to its own K/V (RefOnly attention)
      cfg guidance_scale=4.0 (negative branch = the negative_lat branch)
      Sampling noise randn (seed 42); ancestral noise added within the step
 → latents [1,4,80,120] → unscale_latents → VAE.decode → unscale_image
 → 960×640 3×2 grid → split into 6×(3,320,320) → DINO preprocessing (224) + zero123 cameras (already available)
```

Porting components and op mapping:

| Component | PyTorch | C++ (ggml) | Notes |
|---|---|---|---|
| scheduler | EulerAncestralDiscreteScheduler(trailing) | `src/models/scheduler.cpp` | Faithful reproduction of the diffusers numeric flow; independent parity |
| CLIPVision | CLIPVisionModelWithProjection (ViT-L/14→1024) | `src/models/clip_vision.cpp` | Reuse the DINO DiT/flash-attn pattern |
| text embedding | CLIPTextModel(empty prompt) | conversion-time constant | no port needed |
| VAE | AutoencoderKL encode+decode | `src/models/vae.cpp` | conv2d stack + GroupNorm/SiLU + mid attn |
| UNet | UNet2DConditionModel (SD2.1) + RefOnly | `src/models/unet.cpp` | time-emb/ResBlock/self+cross attn/down-up |
| Post-processing | VaeImageProcessor.postprocess | host C++ | pure numerics |

Weight sources: HF `sudo-ai/zero123plus-v1.2` (diffusers format) + the
`diffusion_pytorch_model.bin` from `TencentARC/InstantMesh` (white-background
fine-tuned UNet override). Conversion script `cpp_ggml/convert/convert_zero123pp.py`,
outputs `models/gguf/zero123pp_{unet,vae,cond}_<p>.gguf`.

## Phase 2: NeRF Variant + NeuralRender (ray marcher)

- `src/model.py`: ray_sampler/ray_marcher (volume-rendering the triplane features) →
  MarchingCubes.
- C++: reuse the synthesizer MLP infrastructure + a host-side ray marcher (numerically
  dense, pinned to CPU for consistency); MarchingCubes lookup tables generated by
  `convert/` (same approach as the FlexiCubes tables).
- `convert_lrm.py` extended for the NeRF variant (the synthesizer head of
  instant-nerf-large.yaml).

## Phase 3: --save_video

- C++ software z-buffer rasterizer (vertex colors/texture; orbit camera trajectory
  aligned with the official `model.render_video`: 36 frames, one full revolution around
  Y, render distance=4.5/2.0 scale).
- Outputs `videos/<name>/frame_%03d.png`; mp4 assembled by the user via
  `ffmpeg -framerate 30 -i ...`.

## Phase 1 bring-up notes (VAE layout debugging, RESOLVED 2026-09-08)

The earlier "conv_2d ne mislabeled" theory in this section was a
MISDIAGNOSIS — a *contiguous* ggml tensor cannot disagree with its own
ne/nb/memory triple. The real contract (pinned by `tests/test_conv_layout.cpp`
against in-process naive references):

- `ggml_conv_2d` consumes a contiguous `[W,H,IC,N]` tensor (im2col steps
  channels via nb12, reads each channel plane as row-major `[H,W]`) and
  returns a plain contiguous `[OW,OH,OC,N]` tensor whose memory IS torch
  NCHW `[N,OC,OH,OW]`.
- `ggml_group_norm` normalizes over ne0·ne1 with groups over ne2 — on that
  layout this is exactly torch GroupNorm (groups = consecutive channels =
  consecutive H·W memory blocks). Exact (probe-verified).
- `ggml_upscale(NEAREST)` scales ne0/ne1 of `[W,H,C,N]`. Exact.
- `ggml_permute` is **scatter** (`ne[axis_i] = a->ne[i]`);
  `cont(permute(x,2,1,0,3))` on `[W,H,C,B]` is the genuine gather into
  torch `[C,H,W,B]` (used inside attention for the `[C,HW]` flatten).
- The whole model therefore runs in ONE layout — ne `[W,H,C,B]`, whose
  byte order IS torch NCHW for any B: host images/latents map onto it with
  a plain byte copy, and NO conv output reorder exists anywhere.

Actual root causes of the old divergence, in the order they were found:

1. **gallocr reuse corrupted the staged dumps** (the historical "galloc reuse
   interference"): a tensor is recycled once its *direct* consumers ran (NOT
   transitive), so consumer-less dump tensors were overwritten by later
   activations. Fix: every post-compute-read tensor is marked
   `ggml_set_output()` (ggml-alloc.c: "graph outputs are never freed").
2. **Attention group_norm affine was missing** (diffusers AttentionBlock's
   `group_norm.weight/bias`).
3. **Asymmetric downsampling**: diffusers `Downsample2D` (constructed with
   `padding=0`) zero-pads the right/bottom edge (`F.pad(x,(0,1,0,1))`) and
   THEN applies the stride-2 **padding-0** conv — not the symmetric
   padding=1. In ggml: `ggml_pad(x, 1, 1, 0, 0)` (p_i pads AFTER ne_i).
4. **F16 im2col floor**: `ggml_conv_2d` hardcodes a F16 im2col for non-bf16
   kernels (~f16-eps relative per conv; the deep encoder stack accumulated
   it to ~1e-1 on the latents). The encoder now builds convs via
   `conv_f32()` (manual `ggml_im2col(..., GGML_TYPE_F32)` + mul_mat,
   identical layout, 2x im2col memory) → encode max_abs 2.4e-4 (pure f32
   summation-order noise). The decoder keeps the built-in f16 conv
   (max_abs 3.0e-3 on pixels ≈ 1.2 8-bit codes, ~50dB PSNR).

Staged-comparison tooling: `convert/dump_vae_stages.py` (torch refs, run
under /tmp/vref) + `convert/compare_vae_stages.py` (numpy). Watch-outs for
reference scripts: bare GroupNorm vs affine'd (the C++ dumps the bare op),
and the asymmetric downsample above. `IM_VAE_DUMP=1` triggers the dumps.

`tests/test_vae.cpp` gate removed — encode/decode both PASS in ctest.

## Acceptance Criteria (alignment scope)

- [ ] `cpp_ggml/build/zero123pp` single image → 6 views, matching the official PyTorch
      (same seed) within a PSNR threshold. Harness: `convert/dump_e2e.py` (official
      per-step fixtures: noise/latents/eps/context) + `zero123pp --fixture-dir
      benchmarks/fixtures/e2e/<name> --dump-final-latents` → compare
      `ref_latents.bin` and per-view PNGs (PSNR).
- [x] Scheduler numeric parity: add_noise/scale_model_input/step whole-sequence
      max_abs < 1e-6 (`test_scheduler` 1.9e-6).
- [x] Staged parity for VAE/CLIPVision/UNet: f32 ≤ 2e-3 (atol+rtol threshold), f16/q8
      within their tolerances (`test_vae` encode 2.4e-4 / decode 3e-3,
      `test_clip_vision` 7.6e-6, `test_unet` f32 7.1e-4 / f16 8.8e-4).
- [~] CUDA / Vulkan / CPU backend consistency: the same components run on each backend
      against the same torch fixtures (device via test argv[1]). Status 2026-09-18:
      CPU all green; CUDA needed `NVIDIA_TF32_OVERRIDE=0` (clip 1.5e-3→1.2e-5); UNet
      fixed for the GEGLU non-contiguous input, now 9.4e-4. Vulkan's two f16 traps
      located and fixed model-side (see "Vulkan f16 traps" below); CLIP per-layer
      regression brought every stage down to torch fp32's own rounding level:
      patch conv 2.1e-4→1.0e-6, attn_raw 1.8e-4→5.5e-6, end-to-end embeds
      1.5e-3→**4.2e-6** (better than CPU's 7.6e-6). zero123pp full 75-step E2E
      (blue_cat, f16 UNet/VAE, seed 42, real image no fixture): Vulkan long runs
      need device fp16 off — now built in via the explicit parameter
      `BackendInitOptions::vulkan_fp16` (default false, `init_best_backend`
      applies it through `ggml_backend_vk_set_fp16` before device creation), no
      env needed; exact f32-matmul routing is likewise built in via an explicit
      API, coopmat stays on serving f16/quantized matmuls. Final latents vs CPU
      max_abs 2.3e-2 / mean 5.2e-4 (PSNR 70.6dB), grid PSNR **54.9dB** (visually
      indistinguishable; the residual is per-op fp32 rounding accumulated over
      150 double-forwards). Note: with device fp16 on, the activation fp16
      staging error of f16-weight matmuls accumulates over sampling steps —
      an 8-step short run is only latents 7.2e-3 / grid 53.4dB, but the full
      75 steps degrade to latents mean 1.9e-2 / grid 28.8dB; **long runs must
      keep device fp16 off**. The GPU acceptance threshold for VAE encode still
      needs relaxing per the expected re-rounding, or tightening op by op.

  Update 2026-09-19 (Vulkan full E2E + f16/f32 dual-path acceptance with probes):
  - **Full chain including VAE decode all green**: with VRAM free (no desktop app
    holding memory) the Vulkan f16 75-step full run took 162.9s, producing VAE
    decode + 6 view PNGs + grid — the earlier VAE decode OOM was purely a desktop
    app (ACloudViewer ~3GB) squeezing VRAM, not a code regression. Check
    `nvidia-smi` for free VRAM before GPU regressions (see "12GB VRAM budget" below).
  - **NaN probe has zero numerical impact (E2E-level re-confirmation)**: with
    `IM_UNET_PROBE=1` on throughout, Vulkan f16 latents vs CPU f16 = max
    **2.286e-2** / mean **5.196e-4**, matching the 9-18 no-probe baseline
    (2.3e-2/5.2e-4); 2-step short runs with probe1/probe2 are bit-exact with the
    no-probe run. Probe usage in `docs/IM_LRM_PROBE.md` (`IM_LRM_PROBE` for LRM's
    single forward; `IM_UNET_PROBE=1` for long diffusion runs to monitor each
    step's w/r eps nan/amax; on divergence `=2 + IM_UNET_PROBE_T=<timestep>`
    bisects down to block/op).
  - **f16/f32 dual paths**: Vulkan f32 75-step latents vs CPU f16 = max 1.955e-2 /
    mean 5.359e-4; Vulkan f16 vs f32 mutual difference 1.306e-2 / 5.334e-4 — both
    paths are stable at fp32 rounding level, f32 weights bring no extra
    degradation. nan=0 throughout all 75 steps, amax stable with no divergence
    (w.eps 4.77→2.15 decreasing, r.eps 0.84→2.71 slowly rising).
    The f32 75-step **full chain (incl. VAE decode) also passes after the unload
    optimization**: 186.5s, 0 OOM, all 6 view PNGs produced, and the final
    latents are **bit-exact** with the pre-optimization f32 75-step run
    (75-step-level confirmation of unload's numerical neutrality).
  - **flexicubes defensive change equivalence (triple verification)**:
    `unique_pairs`' `empty()` → `size() < 2` (removing the implicit invariant
    that paired pushes keep the size even) is bit-exact on CPU f32 and CPU f16;
    Vulkan f16 verified equivalent via a stash experiment (revert just that
    line, rebuild, rerun, compare). Note: cross-version rebuilds (e.g.
    unrelated `models/*.o` changes such as the lrm_transformer probe refactor)
    cause a small global shift in the Vulkan geometry pipeline's SDF (9-18 vs
    9-19: max 1.4, 100% of elements) — see "Regression baseline management" below.

  Vulkan f16 traps (root causes and mitigations, all verified):
  1. **f32 matmul shader stages through f16**: every `matmul_f32_*` SPIR-V
     variant (the coopmat `_cm1` and the fp16-compiled scalar intermediate
     branch) has `FLOAT_TYPE=float16_t` baked in at build time by
     vulkan-shaders-gen; runtime switches can only swap the whole branch. Only
     the `_fp32` variants used by the `!coopmat && !fp16` branch are true fp32.
     Mitigation: the patch provides the explicit API
     `ggml_backend_vk_set_f32_matmul_exact(bool)` (default true); all
     F32×F32→F32 matmuls are routed to the true fp32 scalar pipeline; f16/
     quantized matmuls are unaffected and always use coopmat/tensor cores;
     set false to trade accuracy for speed. The fp16 switches in production
     paths have all converged to explicit parameters: InstantMesh via
     `BackendInitOptions::vulkan_fp16` (explicit API `ggml_backend_vk_set_fp16`,
     applied by `init_best_backend` before device creation), RMBG via env (read
     during device creation, to be converged later); the `GGML_VK_DISABLE_F16`
     env is still supported and ANDs with the explicit parameter, as a
     diagnostic escape hatch. Measured minimal reproduction: mul_mat
     4.6e-3→1.8e-5.
  2. **Non-contiguous matmul operands are implicitly converted to F16**:
     `ggml_vk_mul_mat_q_f16` first does `cpy → F16` on x/y operands that fail
     `ggml_vk_dim01_contiguous`, silently rounding f32 data (per-head strided
     views, un-materialized reshape+permute both hit this; typical symptom: a
     single attention output at ~1e-4 instead of ~1e-6). Mitigation: materialize
     every matmul operand with `ggml_cont` (clip_vision's `qr`, unet's `qh/kh`,
     dino's `to_head` are all handled).
- [ ] `instantmesh --image x.png --rmbg rmbg.gguf` (after rembg and Zero123++ are in)
      produces the same mesh as `python run.py` for the same input (same bridged
      baseline).
- [ ] NeRF variant: the `--config instant-nerf-large` path runs end-to-end and meets
      parity.
- [ ] `--save-video` outputs the frame sequence.

## 12GB VRAM budget (which paths are f16-only)

Measured on RTX 3060 12GB (driver allocation + the ggml single-buffer
constraint: gallocr's `reserve` requires one contiguous block, so the peak is
governed by the **largest single buffer**, not total usage):

| Pipeline stage | f16 ggml single-block request | f32 ggml single-block request | 12GB verdict |
|---|---|---|---|
| zero123pp UNet 75 steps | ~2GB | ~4GB | f16/f32 both fine |
| zero123pp VAE decode | ~1.4GB | **6.25GB** (2.83GB allocation failure measured) | **f32 always OOM before the unload optimization; f32 runs after it (image output verified)** |
| instantmesh geometry (synthesizer) | ~4.7GB | unmeasured (expected >6GB) | f16 usable; f32 recommend CPU |
| rembg (RMBG) | small | small | f16/f32 both fine |

**Two units of measure, do not conflate** (2026-09-19 `nvidia-smi` 0.3-0.5s
sampling, zero123pp 4-step short run, idle ~1.3GB):
- The table's "ggml single-block request" = the size of the **single contiguous
  buffer** reserved by gallocr (the number in OOM errors), not the process
  total;
- **Process residency** (nvidia-smi `memory.used`, before the unload
  optimization): ~7.0GB during stable diffusion, **peaking at 11593 MiB during
  VAE decode (whole-run peak)**;
- **After the `GgufModel::unload()` optimization (re-measured same day):
  whole-run peak down to 8192 MiB (-3.4GB)**; the diffusion stage is unchanged
  (weights still in use), and the VAE decode stage no longer peaks because
  UNet/CLIP weights are released;
- 12GB f16 headroom grows from ~0.6GB to **~4GB**; **the full-f32 E2E is fixed
  accordingly**: after unload the f32 VAE decode (6.25GB single block)
  allocates successfully and the 4-step short run produces a full image
  (0 OOM, 11.9s) — the old "f16-only" conclusion below is overturned.
- **f32 post-fix peak ~9.2GB (9161 MiB, 0.5s sampling over the whole run, same
  environment and idle)**, occurring in the f32 UNet diffusion stage (3.3GB
  weights + ~4GB activation block); the VAE decode stage (6.25GB single block)
  no longer peaks since those weights are released. **f32 headroom ~3GB** (f16
  ~4GB): doubling the f32 batch pushes both the UNet activation block and the
  VAE decode block past 12GB — not feasible; f16 batch=2 has headroom but
  needs measurement.

Rules and lessons:
1. **zero123pp full-f32 E2E on a 12GB card** — before `GgufModel::unload()`
   the latents computed but VAE decode always OOM'd; **fixed after the
   optimization** (f32 4-step short run produces a full image, 0 OOM). The
   f32 75-step full chain works the same way; no need to fall back to CPU
   for image output.
2. **External VRAM usage is the invisible killer**: a resident desktop app
   (e.g. ACloudViewer ~3GB) OOMs even the comfortably-sized f16 VAE decode
   (1.4GB contiguous). Before GPU regressions check
   `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader`
   and inspect leftovers with `nvidia-smi --query-compute-apps=pid,used_memory`.
3. **Do not run GPU tasks concurrently**: launching instantmesh while
   zero123pp's 75 steps run (synthesizer needs 4.7GB) OOMs. GPU regressions
   are strictly serial; CPU tasks can run alongside GPU freely.

## Regression baseline management (same-version principle)

**Background**: on 2026-09-19, comparing the Vulkan geometry pipeline's SDF
across code versions produced a false regression of "100% element difference,
max 1.4" — the cause is that unrelated translation units (`models/*.o`, e.g.
the lrm_transformer probe refactor) recompiled, shifting the GPU numeric path
as a whole (fp32-rounding-level per-op accumulation; final stats
distinguishable, visually not). CPU paths are immune (bit-exact same input).

**Rules**:
1. **Same-version principle**: both sides of any error comparison
   (bit-exact / max / PSNR) must come from the **same source and same build
   artifacts**. When baseline files (`benchmarks/results/*.sdf.bin` etc.) are
   refreshed, record the generating commit hash, ideally as a sidecar file
   (e.g. `blue_cat__vulkan__f16.sdf.bin.rev` containing `git rev-parse HEAD`).
2. **Correct way to prove a change is numerically neutral**:
   `git stash push -- <file>` → rebuild → rerun → compare with the modified
   version's output bit-exact → `git stash pop` → rebuild back. **Never**
   compare an old-dated baseline file against a new build.
3. **Baseline refresh flow**: after a change lands that intentionally accepts a
   new numeric path (e.g. compilation-unit drift), re-run all precision
   combinations of that backend to refresh baselines and update the commit
   hash sidecar; record one line in ALIGNMENT.md (date + reason + new numbers).
4. **Interpreting cross-version diffs**: when a new binary vs an old baseline
   shows "many small differences + a few large ones", first suspect a stale
   baseline (compilation-unit drift), then run the stash experiment to
   attribute — do not treat it as a regression to fix.
5. **Run-to-run determinism**: the same binary on the same backend must be
   bit-exact across runs (verified for Vulkan/CPU). If the same binary
   disagrees with itself, look for non-determinism sources (threads, env, GPU
   state) before talking about precision.

**Baseline ledger** (refreshed 2026-09-19 @ commit `f7fa0877`):
- ✅ Refreshed (with `.rev` sidecar): `benchmarks/results/{img}__{backend}__{prec}.sdf.bin`
  and matching `.obj` (blue_cat/cute_horse/fox/robot × cpu/cuda/vulkan × f32/f16/q8,
  36 combinations). Sanity: all finite; blue_cat cross PSNR — vulkan/f32 60.0dB,
  vulkan/f16 58.4dB, cuda/f16 53.7dB, q8 ~38.6dB (quantization level, expected).
- ✅ Cleaned (2026-09-19, `benchmarks/clean_stale_baselines.sh`): old bench
  drafts `_bench_*.obj` ×9 and the old SDF set `results/sdf/` ×36, 45 files
  total. If the same stale files reappear after a rerun, just run the script
  again (supports --dry-run).
- ⚠️ Leftover unrefreshed artifacts (generated by old code, no sidecar; decide
  keep/delete at next integration):
  - `results/cute_horse_{official_,}tex.{obj,mtl,png}`, `ggml_texmap_*.png`,
    `texmap_ggml_vs_official_*.png`, `tex_*.png`, `tex_render_vs_true.png`,
    `e2e_tex_clean_compare.png` — texture-mapping visual comparisons (some
    referenced by README); visual artifacts, not numeric baselines,
    regenerable by rerunning the texmap flow
  - `results/render_*.png`, `pytorch_vs_ggml_*.png` (README references),
    `precision_*.png`, `latency*.png`, `perf_*.png`, `*.csv`, `report.md`,
    `PERFORMANCE.md` — historical performance/precision reports,
    regenerable by rerunning `bench_full.py`
  - `results/bench.log`, `times.csv`, `full_bench.csv`, `stage_times*.csv` —
    old timing records; rewrite alongside the next baseline refresh
- ✅ No refresh needed (non-C++ artifacts): `benchmarks/fixtures/e2e/cute_horse/`
  (official PyTorch per-step fixture, unaffected by C++ rebuilds);
  `benchmarks/pytorch_ref/*__pytorch.obj`.
- ⚠️ **Clarified (2026-09-19)**: the git-modified state of
  `fixtures/e2e/cute_horse/ref_grid.png` and `ref_view_0..5.png` is **not a
  C++ output overwrite** — evidence: all ref_* files (including the untracked
  ref_latents.bin etc., 531 files) share the same-second mtime (04:08:29),
  i.e. one full re-run of `convert/dump_e2e.py` (the official torch reference
  generator); C++ output filenames differ (grid.png / view_*.png). torch GPU
  inference is non-deterministic and caused pixel drift (vs the HEAD version,
  mean diff ~23-26) — an **intentional reference regeneration**. The new
  fixture set is verified internally self-consistent and aligned with C++:
  `zero123pp --fixture-dir benchmarks/fixtures/e2e/cute_horse --device vulkan`
  (f16 UNet replay, 75 steps) vs the regenerated ref_latents.bin = max 0.81 /
  mean 3.3e-3 / **PSNR 43.0dB** (f16 replay rounding level). Keep the new
  version; note C++ free sampling (no fixture) latents are incomparable to
  ref_latents by design (different noise sequences) — only the fixture-replay
  mode is comparable.

### End-to-end vs PyTorch full-matrix acceptance (2026-09-19; two real inconsistencies found and fixed)

Anchored on the cute_horse official fixture, a full vs-torch matrix was
measured for `{cuda, vulkan} × {f32, f16}`: component level (test_clip_vision
/ test_vae / test_unet, directly against torch dumps) + 75-step E2E replay
(per-stage intermediates + grid PNG). Findings:

1. **CUDA TF32 trap (fixed)**: ggml-cuda unconditionally set
   `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` on every cuBLAS handle —
   f32 matmuls were cut to 10-bit mantissas: `test_clip_vision` 1.49e-3,
   `test_vae` encode 6e-2 (Vulkan/CPU both at ~1e-6/1e-4). **The CUDA twin of
   Vulkan's coopmat f32→f16 staging (fixed 9-18)**. Fix: new patch
   `patches/ggml-cuda-f32-matmul-exact.patch` defaults to
   `CUBLAS_DEFAULT_MATH` (exact FMA), `GGML_CUDA_TF32=1` opts out. After the
   fix clip_vision is 1.24e-5 (PASS); on GeForce TF32 throughput already
   equals fp32 CUDA cores so performance is unchanged; f16/q8 tensor-core
   paths are unaffected (bidirectional switch reproduces the old numbers).
   **Also corrects a historical bias**: `analyze.py` treated f32-CUDA as the
   "highest-fidelity reference" while it actually carried TF32 error; only
   after the fix is that assumption true.
2. **zero123pp VAE decode output read-order bug (fixed)**: the memory returned
   by `vae_decode` is torch `[B,3,H,W]` (channel planes; see the `vae.cpp`
   L382 comment), but zero123pp's PNG writer read it linearly as interleaved
   HWC — channel planes got mis-sliced and RGB mixed, and the grid came out
   as "grey shuffled tiles". The **51.7dB latents acceptance could not see it**
   (the bug is in host post-processing after decode), and the C++-internal
   cross-backend grid comparison could not either (same-mistake cancellation,
   54.9dB). After fixing the read order: **decode(fixture ref_latents) vs
   ref_grid = 62.2dB**, pixel-identical visually (see `zero123pp
   --latents-in`, a debug entry point added for this localization). Lesson:
   **vs-torch consistency requires at least one pixel-level/visual check;
   pure tensor statistics can pass while the image is completely wrong**.
3. **The second VAE-encode CUDA error source (located and fixed)**: after the
   TF32 math-mode fix, `test_vae` encode was still 1.45e-2 (CPU 2.4e-4 /
   Vulkan 1.3e-4). Bisection via `scripts/vae_encode_bisect.sh`: the main
   chain's relative error is flat at 1.4e-6 (normal), and the **only jump is
   `encoder.convout` (rel 3.8e-4, a 270× jump)**. Root cause:
   `ggml_cuda_should_use_mmf`'s F32 branch admitted **fp32 MMA = TF32 tensor
   core** on Ampere (10-bit mantissa), and only thin layers with
   `src1_ncols <= 16` hit it — the encoder tail conv_out/quant_conv (4-channel
   output) matches, while down blocks/resnets (128/256 channels) go through
   cuBLAS and are unaffected. That path bypasses cuBLAS, so math-mode and
   FORCE_CUBLAS are both blind to it. Fix: mmf's F32 branch is gated by
   `GGML_CUDA_TF32` (off by default, `=1` restores). After the fix encode is
   **1.47e-4 PASS**; `GGML_CUDA_TF32=1` reproduces the old 5.97e-2 bit-exact
   (attribution closed). The bisect script's first run also exposed two script
   robustness issues (a FAIL verdict should not abort collection; compare
   dirs must be cleared), fixed alongside.

Post-fix vs-torch E2E matrix (cute_horse fixture replay, 75 steps; grid/view
are PNG pixel PSNR):

| Config | final latents | grid.png | view_0 |
|---|---|---|---|
| cuda/f16 | 51.68dB (mean 4.9e-3) | 50.23dB | 52.74dB |
| cuda/f32 | 51.61dB | 50.08dB | 52.52dB |
| vulkan/f16 | 54.29dB (mean 3.3e-3) | 49.30dB | 52.41dB |
| vulkan/f32 | 54.57dB | 49.49dB | 52.57dB |

The latents max_abs outliers (0.6-0.8) are the inherent level of the torch
fp16 fixture's representation noise chaotically amplified over the last
ancestral steps (the fixture latents are themselves f16 values); impact on
the final image <0.31 pixel (grid 50dB). Component level: UNet f16 double
pass cuda 9.4e-4 / vulkan 5.1e-4 (f16 quantization level); VAE decode
cuda 3.1e-3 / vulkan 2.9e-3; CLIP (after fix) cuda 1.24e-5 / vulkan 4.2e-6;
scheduler bit-exact. Geometry (dino/lrm/synth/flexicubes) vs torch CPU
three-precision parity is in PLAN.md (f32 3e-7 ~ q8 3.4e-2); GPU paths are
anchored to the cpu f32 reference via the 36-combination baseline in
`benchmarks/results` (@ f7fa0877). zero123pp has no q8 weights (quantized
coverage = rmbg/dino/lrm/synthesizer q8 GGUF).

### Peak VRAM optimization: stage weight unloading (GgufModel::unload)

Measured 2026-09-19: zero123pp's process VRAM peak occurs at the VAE decode
stage (f16 11593 MiB / 12GB). Reason: after diffusion, UNet (f16 1.7GB / f32
3.3GB) and CLIP/cond GGUF (2.4GB) weights are still resident while no longer
participating in decode.

Implemented `GgufModel::unload()` (`core/gguf_io.{hpp,cpp}`): explicitly
releases the weight backend buffer + metadata (previously the destructor only
freed metadata; the backend buffer had no release path). Unload points across
the pipeline (placed by the "release immediately after last use" principle):

| Pipeline | Unload point | Freed |
|---|---|---|
| zero123pp | after CLIP encode + scheduler KV reads, before the diffusion loop: `clip.gguf.unload()` | 2.4GB |
| zero123pp | after the diffusion loop, before VAE decode: `unet.gguf.unload()` | 1.7GB (f16) / 3.3GB (f32) |
| instantmesh | after DINO encode: `dino.gguf.unload()` | 0.2GB (f16) / 0.4GB (f32) |
| instantmesh | after TriplaneTransformer, before synthesizer (the pipeline peak stage): `trans.gguf.unload()` | 0.5GB (f16) / 1.0GB (f32) |

VAE weights cannot be unloaded as a whole buffer (encoder/decoder share one
GGUF buffer; decode still needs the decoder subset) — splitting by subgraph
would require changing the allocation strategy; not adopted for now.

**Measured (4-step short run + 0.3s sampling): zero123pp f16 whole-run peak
11593 → 8192 MiB**, matching the freed amounts; **the f32 path's VAE decode
is fixed accordingly** (see above). Structurally neutral to numerics
(releases only happen after that stage's output has landed in host memory).

**Destructor-path audit (2026-09-19)**: all backend buffers in the repo are
allocated through `GgufModel` (the only `ggml_backend_alloc_ctx_tensors` call
site); after unload() gained the release path there are no gaps; all 9
`ggml_gallocr_new/free` sites are paired; graph input tensors are freed with
the gallocr; `read_backend_tensor` data read into host does not occupy VRAM;
`ggml_backend_free(backend)` at main's tail is the backstop. clip's text_emb
and other constants are read into host vectors before cond is unloaded —
unaffected.
