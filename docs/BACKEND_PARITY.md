# Backend Parity Report — CUDA / Vulkan 精度对齐与端到端一致性验收

**验收日期**：2026-09-19 ~ 09-20　**基线 commit**：`69f82b3`（已推送 origin/main）
**硬件**：RTX 3060 12GB　**build**：`cpp_ggml/build-gpu`（CUDA + Vulkan 双后端）
**上游 ggml**：v0.21.0（`8599e0ea`）+ `patches/` 两文件

---

## 1. 精度修复：3 个 CUDA 浮点陷阱

全部实测归因（含双向开关验证），统一退出开关 `GGML_CUDA_TF32=1`（逐位复现旧行为）。

| # | 陷阱 | 根因 | 命中面 | 修复后 |
|---|---|---|---|---|
| 1 | cuBLAS TF32 | `common.cuh` 无条件 `CUBLAS_TF32_TENSOR_OP_MATH`，f32 matmul 被砍到 10-bit mantissa | 全部 batch=1 f32 matmul；CLIP 1.5e-3（CPU 7.6e-6） | clip_vision **1.24e-5 PASS** |
| 2 | Ampere fp32 MMA = TF32 | `mmf.cu should_use_mmf` F32 分支放行 tensor core MMA | **仅薄层**（src1_ncols≤16）：VAE conv_out/quant_conv（4ch 输出）、synthesizer MLP 输出（N=1/3） | test_vae cuda encode **1.47e-4 PASS**（原 5.97e-2） |
| 3 | flash_attn fp16 KV | f32 输入在 kernel 内硬转 fp16（上游性能设计，prec 参数被 CUDA fattn 忽略） | 所有 attention 层 | 已知行为，默认保留（见 §4） |

Trap #2 的隐蔽性：该路径绕开 cuBLAS，math-mode 修复（#1）与 `GGML_CUDA_FORCE_CUBLAS` 均不响应——只有逐层二分（§5 工具）能定位。

**连带发现**：`zero123pp` VAE decode 后处理按交错 HWC 读取 NCHW 通道平面输出——grid PNG 呈灰白拼贴（13.8dB），而 latents 级验收全绿、C++ 内部跨后端互比也全绿（同错抵消）。修复读序后 **62.2dB，视觉逐像素一致**。
**教训**：跨实现一致性验收必须包含至少一次像素级/视觉级检查，纯张量统计可以全部达标而图像全错。

## 2. 端到端一致性矩阵

zero123pp 全链（encode → 75 步扩散 → VAE decode），free-run 同 seed 同 RNG；fixture 重放口径为官方 torch per-step fixture（`benchmarks/fixtures/e2e/cute_horse`）。

| 路径 | 对比 | final latents | grid.png |
|---|---|---|---|
| f32 free-run | cuda vs vulkan | **77.00 dB**（max 2.1e-2 / mean 2.6e-4） | **57.23 dB** |
| f16 free-run | cuda vs vulkan | **68.76 dB**（f16 量化水平） | **54.59 dB** |
| f32 fixture 重放 | 双后端 vs torch | 51.6–54.6 dB | 49.3–50.2 dB |
| f32 decode(torch latents) | C++ vs torch grid | — | **62.2 dB** |

- 跨实现 bit-exact 物理不可达（浮点求和顺序差异）；上表为可证明的最强等价水平
- fixture 重放口径的 max 离群（0.6–0.8）是 torch fp16 fixture 表示噪声经 ancestral 采样混沌放大的固有下限，对图像影响 <0.31 像素
- zero123pp 无 q8 权重；几何 q8 由 36 组合基线覆盖（mmq 量化分支不受本轮修复影响）

## 3. 几何管线逐段噪声分解（f32 权重，blue_cat grid64）

| 段 | cuda vs cpu | vulkan vs cpu | 说明 |
|---|---|---|---|
| dino feats（12 层 attn） | 90.31 dB | 95.69 dB | cuda flash_attn fp16 KV，5.4dB 差 |
| LRM triplane（同 feats 输入隔离） | 66.72 dB | 67.95 dB | planes 全链差异的主导项 |
| planes 全链 | 66.82 dB | 67.89 dB | ≈ LRM 段（dino 输入差异被淹没） |
| synthesizer（同 planes 输入隔离） | **132.46 dB** | — | 纯 MLP，逐位级，无 attention |
| SDF 全链 | 53.71 dB | 59.79 dB | planes 差异 × MLP 增益（~3.3×MSE） |

**结论**：
- mmf 修复收益：planes **+6.5dB**、SDF **+2.8dB**；vulkan 零影响（其 f32 matmul 走 exact shader，从未有此问题）
- SDF 跨后端 ~6dB 差距 = planes 输入差异（flash_attn fp16 KV）经 MLP 增益放大，**synthesizer 自身无可优化算子**
- flash_attn fp16 KV 是上游性能设计（tensor core 强制），不值得追（见 §4）

## 4. flash_attn f32 KV 开关评估——结论：不值得

实测（cuda，`IM_FLASH_F32=1`）：
- **精度零收益**：CUDA fattn 忽略 `GGML_PREC_F32`（源码无 prec 读取；K/Q 在 kernel 内固定转 fp16 走 MMA）——开关前后 dino 90.31dB / lrm 66.72dB / SDF 53.71dB **逐位不变**
- **性能零损失**：F32KV 路径等速（dino 3.84 vs 3.91s，lrm 4.83s 持平）
- 要真 f32 attention 只能退回手动 softmax 路径（大幅性能损失），收益仅 SDF 差距缩小 ~6dB（视觉不可分）

**决策**：不追此差异。代码保留 `IM_FLASH_F32` gate（`dino.cpp` / `lrm_transformer.cpp`，默认无效零开销）作为上游 ggml 支持 prec 时的预留钩子。

## 5. 工具与基础设施沉淀

| 工具 | 用途 |
|---|---|
| `scripts/vae_encode_bisect.sh [cuda\|vulkan]` | IM_VAE_DUMP 逐层二分（CPU 基准 vs GPU，自动标出首个分歧 stage） |
| `zero123pp --latents-in <bin>` | 跳过扩散直接 decode 外部 raw latents（隔离 decode 段） |
| `GgufModel::unload()` | 阶段权重卸载；zero123pp f16 峰值 11593→8192 MiB，f32 VAE decode OOM 修复（latents bit-exact 验证） |
| `clean_stale_baselines.sh` | 过期基线清理（原被全局 `*.sh` 忽略规则误伤，已加反排除入库） |

## 6. 提交记录

主仓 main（全部 `ludahai19@163.com`，已推送 origin/main）：

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

子模块：
- `RMBG-2.0-GGML`：`420c59d`（BackendOptions 显式 API 重构，dev 分支已推送 origin/dev；主仓 gitlink 同步）
- `ggml`：停在上游 `v0.21.0`（`8599e0ea`），工作区改动由 `patches/ggml-rmbg-ops.patch` + `patches/ggml-cuda-f32-matmul-exact.patch` 完整覆盖（comm 差集为空，clone 后 CMake 自动应用）

## 7. 遗留与后续

- flash_attn fp16 KV：待上游 ggml 的 CUDA fattn 支持 prec 后，`IM_FLASH_F32=1` 即可启用（钩子已就位）
- synthesizer f32 单块 >6GB（grid88）：12GB 卡需 grid64 或继续依赖 unload 优化
- 需要更高几何精度时优先评估：cuda/f32 与 vulkan 的 flash_attn 行为差异是否可通过上游 prec 支持收敛
