# InstantMesh-GGML 对齐遗留条目盘点

> C++ ↔ 官方 Python 功能对齐现状、阻塞原因与下一步行动
>
> 生成时间：2026-09-07 · 更新：2026-09-17（UNet/Pipeline ✅，Vulkan 全精度回归进行中）

---

## 总览

| # | 条目 | 状态 | ctest 门禁 |
|---|---|---|---|
| 1 | Zero123++ scheduler | ✅ 完成 | `test_scheduler` Passed（75 步 1.9e-6） |
| 2 | Zero123++ CLIPVision | ✅ 完成 | `test_clip_vision` Passed（7.6e-6） |
| 3 | Zero123++ 权重转换链 | ✅ 完成（3 份 GGUF 已产出） | — |
| 4 | Zero123++ VAE encode/decode | ✅ 完成（encode 2.4e-4 / decode 3.0e-3） | `test_vae` Passed（无门控） |
| 5 | UNet + RefOnly attention | ✅ 完成（f32 7.1e-4 / f16 8.8e-4） | `test_unet` Passed |
| 6 | Denoising pipeline / zero123pp CLI | 🔨 已组装（`src/tools/zero123pp.cpp`），E2E PSNR 验收待做 | 无（`--fixture-dir` 已支持） |
| 7 | rembg | ✅ 完成 | 实测 blue_cat 跑通 |
| 8 | NeRF 变体 + NeuralRender | ❌ 未开始 | 无 |
| 9 | `--save_video` | ❌ 未开始 | 无 |
| 10 | `--export_texmap` 烘焙 | ✅ 已对齐（xatlas UV + 多视角烘焙） | `test_texture_map` Passed |
| 11 | PBR 纹理 | ➖ N/A（官方 InstantMesh 代码中不存在 PBR 通路，无需对齐） | — |
| 12 | ggml 布局契约回归探针 | ✅ 完成 | `test_conv_layout` Passed |
| 13 | Vulkan 全精度逐层回归 | 🔨 进行中（首轮已跑：见 §7） | 测试增加设备覆盖后 cuda/vulkan 分跑 |

---

## 1. Zero123++ Pipeline — UNet/组装已收敛，剩 E2E 验收

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

### 下一步（E2E 验收）

1. **UNet(RefOnly)** ✅（2026-09-16 收敛）：time-emb MLP、ResBlock（GN/SiLU/conv+skip）、self+cross attention（cross_attention_dim=1024、use_linear_projection=true、heads 8×head_dim 40）、down/mid/up + RefOnly w/r 双前向 + ramping 缩放；f32 7.1e-4 / f16 8.8e-4（合成 fixture，torch 逐层参照）。
2. **pipeline 组装** ✅：`src/tools/zero123pp.cpp`（rembg→VAE encode(cond)→CLIPVision→75 步双前向→cfg 4.0→VAE decode→3×2 网格），已支持 `--fixture-dir`（回放官方同 seed 噪声）/`--dump-final-latents`/`--dump-steps`。
3. **E2E PSNR 验收**（当前）：用 `convert/dump_e2e.py` 产出官方逐 step fixture（r/w input、eps、latents、context、噪声），C++ `--fixture-dir` 回放 → 对比 `ref_latents.bin` 与 6 视图 PSNR；方案见 `docs/ALIGNMENT.md` 验收标准。

---

## 2. Denoising Pipeline / zero123pp CLI — 已组装，待 E2E PSNR 验收

- **状态**：官方信息流已在 `docs/ALIGNMENT.md` 完整记录；`src/tools/zero123pp.cpp` 已实现 `rembg → VAE encode(cond) → CLIPVision → 75 步双前向(RefOnly) → cfg 4.0 → VAE decode → unscale → 3×2 网格切分 6 视图`。
- **下一步**：`convert/dump_e2e.py` 生成官方同 seed fixture → `zero123pp --fixture-dir ... --dump-final-latents` 回放 → 逐 step/最终 latents 与 6 视图 PSNR 验收（验收标准已列）。

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
2. ~~UNet(RefOnly) + pipeline 组装~~ ✅（2026-09-16/17）→ 进行中：`zero123pp` e2e PSNR 验收（`dump_e2e.py` + `--fixture-dir`）；
3. **Vulkan 全精度逐层回归**（进行中）：GPU 双后端构建 + ctest cuda/vulkan 分跑 + staged-dump 逐层对比；
4. 并行支线：NeRF 变体转换 + ray marcher；`--save_video` 光栅化器；
5. 最终 e2e：`instantmesh --image x.png --rmbg rmbg.gguf` 与 `python run.py` 同输入同输出对齐（验收标准已写入 `docs/ALIGNMENT.md`）。

---

## 7. Vulkan 全精度逐层回归（2026-09-17 首轮）

- **基础设施**：`backend.cpp` 支持显式设备名（`cuda`/`vulkan` 前缀匹配）；`test_unet/vae/clip_vision` 增加 argv[1] 设备参数（默认 cpu）；`build-gpu`（CUDA+Vulkan 双后端）构建脚本见 `cpp_ggml/README.md` §3 Route C。本机多 CUDA 共存（11.1/11.4/11.8）导致 exe 链接时 libcudart 版本歧义，需 `CMAKE_EXE_LINKER_FLAGS` 显式指向 11.8（仅本机）。
- **修复的 bug**：UNet GEGLU 把非连续 view 传给 `ggml_gelu` → CUDA `unary.cu` 断言 `ggml_is_contiguous` 失败（CPU/Vulkan 容忍）；已加 `ggml_cont`，CUDA UNet 由崩溃 → 9.4e-4 PASS。
- **数值矩阵（vs torch fixture，f32 权重路径）**：

  | 组件 | CPU | CUDA | CUDA(TF32 off) | Vulkan |
  |---|---|---|---|---|
  | CLIPVision | 7.6e-6 ✅ | 1.5e-3 ❌ | 1.2e-5 ✅ | 1.5e-3 ❌ |
  | VAE encode | 2.4e-4 ✅ | 6.0e-2 ❌ | 1.45e-2 ❌ | 1.17e-1 ❌ |
  | VAE decode | 3.0e-3 ✅ | 3.5e-3 ✅ | 3.1e-3 ✅ | 1.7e-2 ❌ |
  | UNet(refonly) | 7.1e-4 ✅ | 崩溃→修复 | 9.4e-4 ✅ | 2.5e-2 ✅(0.35) |

- **VAE encoder staged-dump（CUDA vs Vulkan 逐层 max_abs）**：conv_in_raw 1.4e-4 → rs_conv1 4.3e-3 → down0 4.4e-3 → down1 5.4e-2 → down2 4.1e-1 → mid1 1.76。k_in 0 差。结论：**单调累积的 f32 重排误差（re-rounding），非语义/布局差异**。
- **CLIP staged-dump（CUDA vs Vulkan 逐层 max_abs）**：patch/emb 5.2e-5 → layer0 8.5e-4 → layer1 1.2e-3 → last 3.0e-2 → post_ln 6.2e-2。与 VAE 同模式：24 层单调累积（软max/attention 放大），非语义差异；1024 维投影后（token 平均）误差回到 ~1e-3 量级（vs torch：cuda 1.2e-5 / vulkan 1.5e-3）。
- **CLIP layer0 内部（IM_CV_L0，CUDA vs Vulkan max_abs）**：ln1 4.1e-4（均值 2.9e-7）→ q/k/v 3.5-5.2e-4 → **attn_raw 1.3e-3（首个/最大放大点，softmax attention 路径）** → attn_proj 3.6e-4（大 matmul 平均后回落）。**排除 convs fp16 dot 假设**：CLIP 无 conv（patch embed 走手动 f32 im2col+mul_mat，见 clip_vision.cpp 注释），编码器全为 F32 fma matmul。
- **fp16 dot 路径的真实位置**：ggml Vulkan 的 fp16 混合点积（`v_dot2_f32_f16`，dot_product_funcs.glsl）只在 **F16 权重 matmul** 命中；**conv2d op 的 shmem 在 coopmat2/cm1 设备上为 FP16**（ggml-vulkan.cpp `conv2d_use_fp16_shmem = coopmat2 || cm1`，RTX3060 命中 KHR_coopmat）。这影响 **VAE decoder（f16 conv）与 RMBG**（vulkan decode 1.7e-2 vs cuda 3.1e-3 的候选主因），不影响 CLIP/VAE encoder（f32 路径）。
- **VAE encoder 首个放大点**：rs_norm1（GroupNorm）1.4e-4 → 1.2e-3，之后 conv/matmul 单调累积到 1.76。norm 归约（sum/var）重排是放大器（near-zero 输出元素被 inv_std 放大，均值仍 ~1e-6）。
- **UNet staged-dump**：`IM_VAE_DUMP=1` 在 GPU 上会因 dump 节点撑爆 65536 图容量导致 gallocr 分配失败（CPU 正常）——UNet 逐层 dump 需先裁剪 dump 点或扩容图；组件级已用 test_unet 数值覆盖（cuda 9.4e-4 / vulkan 2.5e-2）。
- **E2E（zero123pp 8 步 f16，seed 42）**：CUDA 32.3s / Vulkan 34.8s 跑通；最终 latents CUDA vs Vulkan PSNR 45.8dB、grid PSNR 33.5dB —— 视觉一致。
- **下一步**：把 VAE encode/clip 的 GPU 验收阈值按 re-rounding 预期定档（如 encode ≤ 1e-1 f16 路径 / clip ≤ 5e-3）；或逐 op 收紧（定位 Vulkan 首个放大点）。

---

## 约束与注意事项

- 构建一律 `make -j ≤ 6`（避免爆内存）；
- **staged dump 必须配 `ggml_set_output()`**（gallocr 会回收无直接消费者的张量，非传递！）；
- 对照前必须 `rm /tmp/vae_*.bin /tmp/ref_*.bin` 清理并核对文件大小（跨轮次污染）；
- torch 参考生成用 `/tmp/vref` venv（transformers 4.52.4 + huggingface_hub 0.36.2 + diffusers 0.39.0）；参考脚本注意 GN 裸输出 vs 带 affine、diffusers 非对称下采样两处易错点；
- **Vulkan 回归注意**：测试设备选择通过 argv[1]（`cuda`/`vulkan`/`cpu`）；Vulkan 后端数学与 CUDA 存在 re-rounding 级差异（f32 应 ~1e-6 量级），阈值对齐 torch fixture 而非 CUDA 位级。

---
