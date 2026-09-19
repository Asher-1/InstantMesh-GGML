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

  更新 2026-09-19（探针加持下的 Vulkan 完整 E2E + f16/f32 双路径验收）:
  - **完整流程含 VAE decode 全绿**：VRAM 空闲（无桌面程序占用）时 Vulkan f16
    75 步全流程 162.9s，VAE decode + 6 视图 PNG + grid 全部产出——此前
    VAE decode OOM 纯因桌面程序（ACloudViewer ~3GB）挤占显存，非代码回归。
    跑 GPU 回归前先 `nvidia-smi` 确认空闲（见下 "12GB 卡 VRAM 预算"）。
  - **NaN 探针零数值影响（E2E 级二次确认）**：`IM_UNET_PROBE=1` 全程开启下
    Vulkan f16 latents vs CPU f16 = max **2.286e-2** / mean **5.196e-4**，
    与 9-18 无探针基线（2.3e-2/5.2e-4）一致；2 步短跑 probe1/probe2 latents
    与无探针 bit-exact。探针用法见 `docs/IM_LRM_PROBE.md`（LRM 单次前向用
    `IM_LRM_PROBE`；扩散长跑用 `IM_UNET_PROBE=1` 监测每步 w/r eps 的
    nan/amax，发散时 `=2 + IM_UNET_PROBE_T=<timestep>` 二分到 block/算子）。
  - **f16/f32 双路径**：Vulkan f32 75 步 latents vs CPU f16 = max 1.955e-2 /
    mean 5.359e-4；Vulkan f16 vs f32 互差 1.306e-2 / 5.334e-4——两条路径均
    稳定在 fp32 舍入水平，f32 权重未带来额外退化。75 步全程 nan=0，amax
    平稳无发散（w.eps 4.77→2.15 递减，r.eps 0.84→2.71 缓增）。
    f32 75 步**完整流程（含 VAE decode）在 unload 优化后亦通过**：
    186.5s、0 OOM、6 视图 PNG 全出，且最终 latents 与优化前 f32 75 步
    **bit-exact**（unload 数值无关性的 75 步级确认）。
  - **flexicubes 防御改动等价性（三重验证）**：`unique_pairs` 的
    `empty()` → `size() < 2`（消除"成对 push 尺寸恒偶"隐式不变式依赖）在
    CPU f32、CPU f16 均 bit-exact；Vulkan f16 经 stash 对照实验（只回退该行
    重编重跑）bit-exact 证实等价。注意：跨代码版本重编（如 models/*.o 的
    无关改动触发重编）会使 Vulkan 几何管线 SDF 整体微移（9-18 vs 9-19
    max 1.4，100% 元素级）——见下 "回归基线管理规范"。

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

## 12GB 卡 VRAM 预算（哪些路径只能跑 f16）

实测于 RTX 3060 12GB（驱动分配 + ggml 单块 buffer 约束，gallocr 的
`reserve` 要求整块连续 buffer，峰值取决于**最大单个 buffer**而非总占用）：

| 管线阶段 | f16 ggml 单块请求 | f32 ggml 单块请求 | 12GB 卡结论 |
|---|---|---|---|
| zero123pp UNet 75 步 | ~2GB | ~4GB | f16/f32 均可 |
| zero123pp VAE decode | ~1.4GB | **6.25GB**（2.83GB 分配失败实测） | **unload 优化前 f32 必 OOM；优化后 f32 可跑（已实测出图）** |
| instantmesh 几何管线（synthesizer） | ~4.7GB | 未测（预期 >6GB） | f16 可用；f32 建议直接 CPU |
| rembg（RMBG） | 小 | 小 | f16/f32 均可 |

**两个口径，别混淆**（2026-09-19 `nvidia-smi` 0.3-0.5s 采样实测，zero123pp
4 步短跑，idle ~1.3GB）：
- 上表"ggml 单块请求"= gallocr reserve 的**单个连续 buffer**大小（OOM 报错
  里的数字），不代表进程总占用；
- **进程总驻留**（nvidia-smi `memory.used` 口径，unload 优化前）：扩散稳定
  期 ~7.0GB，**VAE decode 阶段冲到 11593 MiB（全流程峰值）**；
- **`GgufModel::unload()` 优化后（同日重测）：全流程峰值降到
  8192 MiB（-3.4GB）**，扩散阶段不变（权重仍在用），VAE decode 阶段
  因 UNet/CLIP 权重已释放不再冲顶；
- 12GB 卡 f16 余量从 ~0.6GB 提升到 **~4GB**；**f32 全权重 E2E 因此修复**：
  unload 后 f32 VAE decode（6.25GB 单块）分配成功，4 步短跑完整出图
  （0 OOM，11.9s）——下表"只能 f16"的旧结论已被优化推翻。
- **f32 优化后峰值 ~9.2GB（9161 MiB，0.5s 采样全程日志，同环境同 idle）**，
  出现在 f32 UNet 扩散阶段（3.3GB 权重 + ~4GB 激活单块）；VAE decode 阶段
  （6.25GB 单块）因权重已释放不再冲顶。**f32 余量 ~3GB**（f16 ~4GB）：
  f32 路径 batch 翻倍会同时把 UNet 激活单块与 VAE decode 单块推过 12GB，
  不可行；f16 路径 batch=2 尚有余量但需实测确认。

规则与经验：
1. **12GB 卡上 zero123pp 的 f32 全权重 E2E**——`GgufModel::unload()` 优化前
   latents 能算完但 VAE decode 必 OOM；**优化后已修复**（f32 4 步短跑完整
   出图，0 OOM）。75 步 f32 完整流程同理可用，无需再退回 CPU 出图。
2. **外部显存占用是隐形杀手**：桌面程序（如 ACloudViewer ~3GB）常驻会使
   本来够用的 f16 VAE decode（1.4GB 连续分配）也 OOM。跑 GPU 回归前先
   `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader`
   确认空闲，并用 `nvidia-smi --query-compute-apps=pid,used_memory` 排查残留。
3. GPU 任务**不要并发**：zero123pp 75 步运行中再起 instantmesh（synthesizer
   需 4.7GB）会 OOM。GPU 回归一律串行；CPU 任务可与 GPU 并行互不干扰。

## 回归基线管理规范（同版本基线原则）

**背景**：2026-09-19 发现，跨代码版本对比 Vulkan 几何管线 SDF 会得到
"100% 元素差异、max 1.4"的假回归——根因是 `models/*.o`（如 lrm_transformer
探针重构）等**与被测逻辑无关的编译单元**重编后，GPU 数值路径发生整体微移
（fp32 舍入级逐 op 累积，最终 stats 可分辨但视觉不可分）。CPU 路径不受此
影响（同输入 bit-exact）。

**规范**：
1. **同版本原则**：误差对比（bit-exact / max / PSNR）的两侧必须由**同一份
   源码同一编译产物**生成。基线文件（`benchmarks/results/*.sdf.bin` 等）
   更新时必须记录生成它的 commit hash，建议存 sidecar 文件
   （如 `blue_cat__vulkan__f16.sdf.bin.rev` 写入 `git rev-parse HEAD`）。
2. **改动等价性验证的正确姿势**：证明某改动不改变数值，用
   `git stash push -- <该文件>` → 重编 → 重跑 → 与改动版输出对比
   bit-exact → `git stash pop` → 重编恢复。**不要**拿旧日期的基线文件对比
   新编译产物。
3. **基线刷新流程**：代码变更合入后，若有意接受新的数值路径（如编译单元
   变化导致的微移），重跑该后端全部精度组合刷新基线并更新 commit hash
   sidecar；刷新后在 ALIGNMENT.md 记录一行（日期 + 变更原因 + 新基线数值）。
4. **跨版本 diff 的判读**：新二进制 vs 旧基线出现"大量元素小差异 +
   少量大差异"时，先怀疑基线过期（编译单元微移），再做 stash 对照实验
   归因，不要直接当回归修。
5. **运行间确定性**：同二进制同后端重复运行应 bit-exact（Vulkan/CPU 已
   验证）。若同一二进制两次运行都不一致，先查非确定性来源（线程数、
   env、GPU 状态），再谈精度。

**基线台账**（2026-09-19 刷新 @ commit `f7fa0877`）：
- ✅ 已刷新（附 `.rev` sidecar）：`benchmarks/results/{img}__{backend}__{prec}.sdf.bin`
  及同名 `.obj`（blue_cat/cute_horse/fox/robot × cpu/cuda/vulkan × f32/f16/q8，
  36 组合）。健全性：全 finite；blue_cat 交叉 PSNR——vulkan/f32 60.0dB、
  vulkan/f16 58.4dB、cuda/f16 53.7dB、q8 ~38.6dB（量化水平，符合预期）。
- ✅ 已清理（2026-09-19，`benchmarks/clean_stale_baselines.sh`）：旧 bench
  草稿 `_bench_*.obj` ×9 与旧版 SDF 集合 `results/sdf/` ×36，共 45 个文件。
  重跑新基线后再次出现同名残留时重跑该脚本即可（支持 --dry-run）。
- ⚠️ 遗留未刷新产物（旧代码生成、无 sidecar，下次合入时决定去留）：
  - `results/cute_horse_{official_,}tex.{obj,mtl,png}`、`ggml_texmap_*.png`、
    `texmap_ggml_vs_official_*.png`、`tex_*.png`、`tex_render_vs_true.png`、
    `e2e_tex_clean_compare.png` —— 纹理映射视觉对比（README 引用部分图），
    视觉产物非数值基线，重跑 texmap 流程可再生
  - `results/render_*.png`、`pytorch_vs_ggml_*.png`（README 引用）、
    `precision_*.png`、`latency*.png`、`perf_*.png`、`*.csv`、`report.md`、
    `PERFORMANCE.md` —— 历史性能/精度报告，重跑 `bench_full.py` 可再生
  - `results/bench.log`、`times.csv`、`full_bench.csv`、`stage_times*.csv` ——
    旧耗时记录，刷新基线时应随跑重写
- ✅ 无需刷新（非 C++ 产物）：`benchmarks/fixtures/e2e/cute_horse/`（官方
  PyTorch per-step fixture，C++ 重编不影响）；`benchmarks/pytorch_ref/*__pytorch.obj`。
- ⚠️ **已澄清（2026-09-19）**：`fixtures/e2e/cute_horse/ref_grid.png`、
  `ref_view_0..5.png` 的 git 修改状态**不是 C++ 输出覆盖**——证据：所有
  ref_* 文件（含 untracked 的 ref_latents.bin 等 531 个）mtime 为同一秒
  （04:08:29），是 `convert/dump_e2e.py`（官方 torch 参考生成脚本）的一次
  完整重跑；C++ 输出文件名不同（grid.png / view_*.png）。torch GPU 推理
  非确定导致像素漂移（vs HEAD 版 mean diff ~23-26），属**有意的参考重生成**。
  已验证新 fixture 集内部自洽且与 C++ 对齐：`zero123pp --fixture-dir
  benchmarks/fixtures/e2e/cute_horse --device vulkan`（f16 UNet 重放 75 步）
  vs 重生成后的 ref_latents.bin = max 0.81 / mean 3.3e-3 / **PSNR 43.0dB**
  （f16 重放舍入水平）。保留新版本；注意 C++ 自由采样（无 fixture）的
  latents 与 ref_latents 本就不可比（噪声序列不同），只有 fixture 重放
  模式可对比。

### 端到端 vs PyTorch 全矩阵验收（2026-09-19，发现并修复 2 个真实不一致点）

以 cute_horse 官方 fixture 为锚，对 `{cuda, vulkan} × {f32, f16}` 做了
组件级（test_clip_vision / test_vae / test_unet，直接对比 torch dump）+
75 步 E2E 重放（逐级中间张量 + grid PNG）的 vs torch 全矩阵实测。发现：

1. **CUDA TF32 陷阱（已修复）**：ggml-cuda 对每个 cuBLAS handle 无条件
   `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)`——f32 matmul 被砍到
   10-bit mantissa：`test_clip_vision` 1.49e-3、`test_vae` encode 6e-2
   （Vulkan/CPU 均为 ~1e-6/1e-4 水平）。**Vulkan coopmat 的 f32→f16 暂存
   （9-18 已修）在 CUDA 的孪生问题**。修复：新 patch
   `patches/ggml-cuda-f32-matmul-exact.patch` 默认 `CUBLAS_DEFAULT_MATH`
   （精确 FMA），`GGML_CUDA_TF32=1` 显式退出。修复后 clip_vision
   1.24e-5（PASS）；GeForce 上 TF32 吞吐本就等于 fp32 CUDA core，
   性能无损；f16/q8 tensor-core 路径不受影响（双向开关复现验证过）。
   **连带修正一个历史偏差**：此前 `analyze.py` 以 f32-CUDA 为"最高保真
   参考"，实际带着 TF32 误差；修复后该假设才真正成立。
2. **zero123pp VAE decode 输出读序错误（已修复）**：`vae_decode` 返回的
   内存是 torch `[B,3,H,W]`（通道平面序，见 `vae.cpp` L382 注释），而
   zero123pp 的 PNG 写出按 HWC 交错序线性读——通道平面被错切、RGB 混叠，
   输出 grid 是"多块灰白拼贴"。**latents PSNR 51.7dB 的验收完全发现不了
   它**（错在 decode 之后的 host 后处理）；C++ 内部跨后端 grid 互比也
   发现不了（两边同错抵消出 54.9dB）。修复读序后：**decode(fixture
   ref_latents) vs ref_grid = 62.2dB**，视觉逐像素一致（见
   `zero123pp --latents-in`，为定位新增的调试入口）。教训：**vs torch 的
   一致性必须至少有一次像素级/视觉级验证，纯张量统计可以全部达标而图像
   全错**。
3. **VAE encode CUDA 的第二个误差源（已定位并修复）**：TF32 math-mode 修复后
   `test_vae` encode 仍 1.45e-2（CPU 2.4e-4 / Vulkan 1.3e-4）。经
   `scripts/vae_encode_bisect.sh` 逐层二分：主链相对误差恒定 1.4e-6（正常），
   **唯一跳变点 = `encoder.convout`（rel 3.8e-4，跳升 270 倍）**。根因：
   `ggml_cuda_should_use_mmf` 的 F32 分支在 Ampere 上放行 **fp32 MMA =
   TF32 tensor core**（10-bit mantissa），且仅 `src1_ncols <= 16` 的薄层命中
   ——encoder 尾部 conv_out/quant_conv（4 通道输出）正好命中，down 块/
   resnet（128/256 通道）走 cuBLAS 不受影响。该路径绕开 cuBLAS，故
   math-mode/FORCE_CUBLAS 均不响应。修复：mmf 的 F32 分支与
   `GGML_CUDA_TF32` 语义统一（默认关闭，`=1` 恢复）。修复后 encode
   **1.47e-4 PASS**；`GGML_CUDA_TF32=1` 逐位复现 5.97e-2 的旧行为（归因
   闭环）。bisect 脚本首跑曾暴露两处脚本健壮性问题（FAIL 判定不应中断
   采集、对比目录需清空），已随修复一并处理。

修复后 vs torch E2E 矩阵（cute_horse fixture 重放 75 步，grid/view 为
PNG 像素 PSNR）：

| 配置 | final latents | grid.png | view_0 |
|---|---|---|---|
| cuda/f16 | 51.68dB（mean 4.9e-3） | 50.23dB | 52.74dB |
| cuda/f32 | 51.61dB | 50.08dB | 52.52dB |
| vulkan/f16 | 54.29dB（mean 3.3e-3） | 49.30dB | 52.41dB |
| vulkan/f32 | 54.57dB | 49.49dB | 52.57dB |

latents 的 max_abs 离群（0.6-0.8）是 torch fp16 fixture 的表示噪声经
ancestral 采样末段混沌放大的固有水平（fixture latents 本身是 f16 值），
对最终图像的影响 < 0.31 像素（grid 50dB）。组件级：UNet f16 双 pass
cuda 9.4e-4 / vulkan 5.1e-4（f16 量化水平）；VAE decode
cuda 3.1e-3 / vulkan 2.9e-3；CLIP（修复后）cuda 1.24e-5 / vulkan 4.2e-6；
scheduler bit-exact。几何管线（dino/lrm/synth/flexicubes）vs torch 的
CPU 三精度 parity 见 PLAN.md（f32 3e-7 ~ q8 3.4e-2），GPU 路径由
`benchmarks/results` 36 组合基线（@ f7fa0877）锚定 cpu f32 参考。
zero123pp 无 q8 权重（quantized 覆盖 = rmbg/dino/lrm/synthesizer 的
q8 GGUF）。

### 峰值显存优化：stage 权重卸载（GgufModel::unload）

2026-09-19 实测确认 zero123pp 的进程 VRAM 峰值出现在 VAE decode 阶段
（f16 11593 MiB / 12GB）。原因：扩散结束后 UNet（f16 1.7GB / f32 3.3GB）
与 CLIP/cond GGUF（2.4GB）的权重仍驻留，而它们在 decode 中已无参与。

已实现 `GgufModel::unload()`（`core/gguf_io.{hpp,cpp}`）：显式释放权重
backend buffer + 元数据（此前析构只释放元数据，backend buffer 从无释放
路径）。全管线的卸载点（按"权重最后一次使用后立即释放"原则布置）：

| 管线 | 卸载点 | 释放量 |
|---|---|---|
| zero123pp | CLIP 编码 + scheduler KV 读取完成后、扩散循环前：`clip.gguf.unload()` | 2.4GB |
| zero123pp | 扩散循环结束、VAE decode 前：`unet.gguf.unload()` | 1.7GB (f16) / 3.3GB (f32) |
| instantmesh | DINO 编码完成后：`dino.gguf.unload()` | 0.2GB (f16) / 0.4GB (f32) |
| instantmesh | TriplaneTransformer 后、synthesizer（管线峰值阶段）前：`trans.gguf.unload()` | 0.5GB (f16) / 1.0GB (f32) |

VAE 权重不能整 buffer 卸载（encoder/decoder 共用一个 GGUF buffer；decode
仍需 decoder 子集）——按子图拆分需改分配策略，暂不采用。

**实测（4 步短跑 + 0.3s 采样）：zero123pp f16 全流程峰值 11593 →
8192 MiB**，与释放量吻合；**f32 路径 VAE decode 随之修复**（见上）。
结构上无数值影响（释放只发生在该 stage 输出已落到 host 之后）。

**析构路径审查结论（2026-09-19）**：全仓 backend buffer 只经
`GgufModel`（`ggml_backend_alloc_ctx_tensors` 唯一调用点）分配，unload()
补上释放路径后无遗漏；`ggml_gallocr_new/free` 9 处全部配对；graph 输入
张量随 gallocr 释放；`read_backend_tensor` 读入 host 的数据不占 VRAM；
main 尾部 `ggml_backend_free(backend)` 兜底。clip 的 text_emb 等常量在
unload cond 前已读入 host vector，不受影响。
