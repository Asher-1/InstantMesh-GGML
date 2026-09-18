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

## Phase 1: Zero123++ Multi-View Diffusion (UNet + pipeline 已收敛，剩 E2E 验收)

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

1. **gallocr reuse corrupted the staged dumps** (the historical "galloc 复用
   干扰"): a tensor is recycled once its *direct* consumers ran (NOT
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
      CPU 全绿；CUDA 需 `NVIDIA_TF32_OVERRIDE=0`（clip 1.5e-3→1.2e-5），UNet 修复
      GEGLU 非连续输入后 9.4e-4。Vulkan 的两个 f16 陷阱已定位并在模型侧修复
      （见下 "Vulkan f16 traps"），CLIP 逐层回归全部 stage 降到 torch fp32 自身
      舍入水平：patch conv 2.1e-4→1.0e-6、attn_raw 1.8e-4→5.5e-6、端到端
      embeds 1.5e-3→**4.2e-6**（优于 CPU 7.6e-6）。zero123pp 完整 75 步 E2E
      （blue_cat，f16 UNet/VAE，seed 42，真实图像无 fixture）：Vulkan 长跑需
      关 device fp16——现由显式参数 `BackendInitOptions::vulkan_fp16`（默认
      false，`init_best_backend` 经 `ggml_backend_vk_set_fp16` 在设备创建前
      应用）内建，无需任何 env；f32 matmul 精确路由同样由显式 API 内建，
      coopmat 可保持开启服务 f16/量化 matmul。vs CPU 最终 latents max_abs
      2.3e-2 / mean 5.2e-4（PSNR 70.6dB）、grid PSNR **54.9dB**（视觉不可分
      辨；残余为 150 次 double-forward 的逐 op fp32 舍入累积）。注意：device
      fp16 开启时，f16 权重 matmul 的激活 f16 暂存误差随采样步数累积——
      8 步短跑仅 latents 7.2e-3 / grid 53.4dB，75 步全跑恶化到 latents mean
      1.9e-2 / grid 28.8dB，**长跑必须关 device fp16**。VAE encode 的 GPU
      阈值仍需按 re-rounding 预期放宽或逐 op 收紧。

  Vulkan f16 traps（根因与规避，均已验证）:
  1. **f32 matmul shader 以 f16 暂存**：所有 `matmul_f32_*` SPIR-V 变体
     （coopmat `_cm1` 与 fp16 编译的 scalar 中间分支）的 `FLOAT_TYPE=float16_t`
     在构建期由 vulkan-shaders-gen 嵌死，运行时开关只能整支切换。只有
     `!coopmat && !fp16` 分支用的 `_fp32` 变体是真 fp32。规避：patch 提供
     显式 API `ggml_backend_vk_set_f32_matmul_exact(bool)`（默认 true），
     F32×F32→F32 matmul 一律路由到真 fp32 scalar pipeline；f16/量化 matmul
     不受影响，永远走 coopmat/tensor core；设 false 可换回性能。生产路径的
     fp16 开关已全部收敛为显式参数：InstantMesh 经
     `BackendInitOptions::vulkan_fp16`（显式 API `ggml_backend_vk_set_fp16`，
     `init_best_backend` 在设备创建前应用），RMBG 经 env（设备创建期读取，
     待后续收敛）；env `GGML_VK_DISABLE_F16` 仍受支持并与显式参数相与，仅作
     诊断逃生口。实测最小复现 mul_mat 4.6e-3→1.8e-5。
  2. **非连续 matmul 操作数被隐式转 F16**：`ggml_vk_mul_mat_q_f16` 对未通过
     `ggml_vk_dim01_contiguous` 的 x/y 操作数先 `cpy → F16` 再派发，f32 数据
     被静默舍入（per-head strided view、未 cont 的 reshape+permute 都会踩中，
     典型症状：attention 输出单层 ~1e-4 而非 ~1e-6）。规避：一切 matmul
     操作数用 `ggml_cont` 物化（clip_vision 的 `qr`、unet 的 `qh/kh`、dino
     的 `to_head` 已统一处理）。
- [ ] `instantmesh --image x.png --rmbg rmbg.gguf` (after rembg and Zero123++ are in)
      produces the same mesh as `python run.py` for the same input (same bridged
      baseline).
- [ ] NeRF variant: the `--config instant-nerf-large` path runs end-to-end and meets
      parity.
- [ ] `--save-video` outputs the frame sequence.
