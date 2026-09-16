# InstantMesh-GGML 对齐遗留条目盘点

> C++ ↔ 官方 Python 功能对齐现状、阻塞原因与下一步行动
>
> 生成时间：2026-09-07 · 更新：2026-09-08（VAE 收敛 ✅，布局契约定案）

---

## 总览

| # | 条目 | 状态 | ctest 门禁 |
|---|---|---|---|
| 1 | Zero123++ scheduler | ✅ 完成 | `test_scheduler` Passed（75 步 1.9e-6） |
| 2 | Zero123++ CLIPVision | ✅ 完成 | `test_clip_vision` Passed（7.6e-6） |
| 3 | Zero123++ 权重转换链 | ✅ 完成（3 份 GGUF 已产出） | — |
| 4 | Zero123++ VAE encode/decode | ✅ 完成（encode 2.4e-4 / decode 3.0e-3） | `test_vae` Passed（无门控） |
| 5 | UNet + RefOnly attention | ❌ 未开始（布局契约已就绪） | 无 |
| 6 | Denoising pipeline / zero123pp CLI | ❌ 未开始（被 5 阻塞） | 无 |
| 7 | rembg | ✅ 完成 | 实测 blue_cat 跑通 |
| 8 | NeRF 变体 + NeuralRender | ❌ 未开始 | 无 |
| 9 | `--save_video` | ❌ 未开始 | 无 |
| 10 | `--export_texmap` 烘焙 | ✅ 已对齐（xatlas UV + 多视角烘焙） | `test_texture_map` Passed |
| 11 | PBR 纹理 | ➖ N/A（官方 InstantMesh 代码中不存在 PBR 通路，无需对齐） | — |
| 12 | ggml 布局契约回归探针 | ✅ 完成 | `test_conv_layout` Passed |

---

## 1. Zero123++ Pipeline — VAE 已收敛，剩 UNet + 组装

### 已完成部分

- **scheduler**（`cpp_ggml/src/models/scheduler.cpp`）：EulerAncestral，**linear beta + v_prediction + trailing**（注意与本仓库 scheduler config 对齐而非 SD 惯例）；`alphas_cumprod` 曲线随 GGUF 下发绕开 torch f32 cumprod 舍入不可复现问题。parity：timesteps 逐位一致、sigmas 9.5e-7、全 75 步 1.9e-6。
- **CLIPVision**（`cpp_ggml/src/models/clip_vision.cpp`）：ViT-L/14@1280→1024，手动 softmax attention（seq=257）；已修复 q/k/v bias、`pre_layrnorm` 历史拼写位置等。parity 7.6e-6。
- **转换链**（`cpp_ggml/convert/convert_zero123pp.py`）：`zero123pp_unet_f16`（1.73GB）/ `vae_f16` / `vae_f32` / `cond_f32`（含 text_emb[77,1024]、negative_lat、ramp[65]、alphas_cumprod[1000]）全部就位。
- **VAE**（2026-09-08 收敛）：encode mode max_abs=2.36e-4（f32 重排噪声级），decode max_abs=3.0e-3（f16 im2col 地板，≈50dB PSNR）。

### VAE 收敛实录（五连根因，全部有探针/对质实锤）

1. **"ne 谎报"是误诊**：连续 ggml 张量的 ne+nb+内存三元组必然自洽。v0.21 `conv_2d` 输出即干净连续 `[W,H,OC,N]`（内存=torch NCHW）。`tests/test_conv_layout.cpp`（同进程 naive 参考）钉死契约。
2. **gallocr 复用干扰**：无消费者的 dump 张量在其直接消费者运行后即被回收（非传递！）。修复：读出端一律 `ggml_set_output()`（ggml-alloc.c：outputs 永不回收）。
3. **GN affine 缺失**：mid attention 的 `group_norm.weight/bias` 之前没乘加。
4. **非对称下采样**：diffusers Downsample2D（`ds.padding=0`）= 先 `F.pad(x,(0,1,0,1))` 右下零填充，再 stride-2 **padding-0** conv；`ggml_pad` 参数是 ne 后缀填充量，应为 `(1,1,0,0)`。
5. **f16 im2col 地板**：`ggml_conv_2d` 硬编码 f16 im2col（~f16 eps/conv，深链累积到 1e-1）。encode 改用 `conv_f32()`（手动 im2col F32 + mul_mat，内存×2）压回 2.4e-4；decode 维持 f16（已达标）。

### 下一步（沿布局契约直接铺 UNet）

1. **UNet(RefOnly)**：time-emb MLP（sinusoidal+linear）、ResBlock（GN/SiLU/conv+skip，GN affine 走 `[1,1,C,1]`）、self+cross attention（cross_attention_dim=1024、use_linear_projection=true、heads 8×head_dim 40）、down/mid/up 结构；**RefOnly**：w-forward 收集各层 self-attn K/V → ref_dict，r-forward 拼接（零初始化权重，仅 cat）；`ramping_coefficients[65]` 逐 step 缩放。复用 `conv_f32` 与 staged-dump 方法论（`convert/dump_vae_stages.py` + `compare_vae_stages.py` 模式）。
2. **pipeline 组装** → `zero123pp` e2e PSNR 验收。

---

## 2. Denoising Pipeline / zero123pp CLI — 未开始（被 UNet 阻塞）

- **状态**：官方信息流已在 `docs/ALIGNMENT.md` 完整记录。
- **下一步**：组装 `rembg → VAE encode(cond) → CLIPVision → 75 步双前向(RefOnly) → cfg 4.0 → VAE decode → unscale → 3×2 网格切分 6 视图`；`VaeImageProcessor.postprocess` 后处理为 host C++ 纯数值；验收标准为同 seed 与 PyTorch PSNR 达标。

---

## 4. NeuralRender / NeRF 变体 — 未开始

- **状态**：C++ 仅有 Mesh/FlexiCubes 路径；`cpp_ggml/convert/convert_lrm.py` 中无任何 nerf 支持（`instant-nerf-large.yaml` 的 synthesizer 头未导出）。
- **阻塞原因**：无硬阻塞，独立于 Zero123++ 线，可并行推进。
- **下一步行动**：
  1. `convert_lrm.py` 扩展 NeRF 变体头导出（ray_sampler/ray_marcher 所需 MLP 权重）；
  2. host 侧 ray marcher（数值稠密、锁定 CPU 保证一致性）+ MarchingCubes 查找表生成（复用 FlexiCubes 表的生成方式）；
  3. e2e `--config instant-nerf-large` parity（验收标准已列）。

---

## 5. `--save_video` PNG 帧序列 — 未开始

- **状态**：无任何实现；官方为 `model.render_video` 36 帧绕 Y 一周（render distance=4.5/2.0 scale）+ mp4。
- **技术决策已定**：C++ 软件 z-buffer 光栅化 → `videos/<name>/frame_%03d.png`，mp4 由外部 ffmpeg 合成。
- **阻塞原因**：无；建议放在 pipeline 收敛后（复用已有的 canonical_render/纹理渲染基准资产做对照）。
- **下一步**：光栅化器（顶点色 + 贴图两种路径）、与官方轨迹对齐的 orbit 相机（36 帧）、逐帧与官方视频抽帧做 PSNR 对照。

---

## 6. 其他功能差异

| 项 | 状态 | 说明 |
|---|---|---|
| PBR 纹理 | ➖ 无需对齐 | 官方代码无 metallic/roughness 通路（已审计） |
| 烘焙 `--export_texmap` | ✅ 已对齐 | xatlas UV 展开 + 多视角烘焙，官方即此实现 |
| rembg | ✅ 已对齐 | RMBG-2.0 vendor（质量优于官方 u2net） |
| DINO / LRM / FlexiCubes / synthesizer | ✅ 已对齐 | 既有基线 7/7 全绿 |
| VaeImageProcessor.postprocess | ❌ 随 pipeline 做 | 纯数值，无风险 |
| text encoder | ➖ 无需移植 | 空 prompt 常量已在转换期预计算进 GGUF |

---

## 建议执行顺序

1. ~~VAE 数值收敛~~ ✅（2026-09-08，见 §1 实录）；
2. **UNet(RefOnly)**（直接复用 §1 布局契约 + conv_f32 + staged-dump 方法论）→ **pipeline 组装** → `zero123pp` e2e PSNR 验收；
3. 并行支线：NeRF 变体转换 + ray marcher；`--save_video` 光栅化器；
4. 最终 e2e：`instantmesh --image x.png --rmbg rmbg.gguf` 与 `python run.py` 同输入同输出对齐（验收标准已写入 `docs/ALIGNMENT.md`）。

---

## 约束与注意事项

- 构建一律 `make -j ≤ 6`（避免爆内存）；
- **staged dump 必须配 `ggml_set_output()`**（gallocr 会回收无直接消费者的张量，非传递！）；
- 对照前必须 `rm /tmp/vae_*.bin /tmp/ref_*.bin` 清理并核对文件大小（跨轮次污染）；
- torch 参考生成用 `/tmp/vref` venv（transformers 4.52.4 + huggingface_hub 0.36.2 + diffusers 0.39.0）；参考脚本注意 GN 裸输出 vs 带 affine、diffusers 非对称下采样两处易错点；
- CMake 注释中 "v0.18.1" 已过时：vendored ggml 实为 **v0.21.0**。

---
