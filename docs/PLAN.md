# InstantMesh-GGML 纯 C++ 集成方案

目标：对 InstantMesh 的完整推理管线做 **ggml 原生移植**，不依赖 PyTorch 运行时（仅转换期用 Python 生成参考与 GGUF），并实现：

1. **纯 C++/ggml 运行**：CPU / CUDA / Vulkan 三后端一份代码。
2. **精度对齐**：ggml 三后端输出与 PyTorch 参考逐层一致。
3. **性能达标**：CUDA 与 Vulkan 推理均快于 PyTorch CUDA。
4. **GGUF 多精度**：所有子模型同时产出 f32 / f16 / q8 三种量化版本。

---

## 1. 结论先行（第一性原理推导）

- ggml 用统一 backend registry 抽象。**CUDA/Vulkan 只是编译期选项，推理代码同一份**：运行时 `ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)` 自动取第一个可用 GPU 设备，否则回退 CPU。
- 因此"全面支持"≠ 写三套代码，而是 **`-DGGML_CUDA=ON -DGGML_VULKAN=ON` 一次编译 + 运行时探测**。
- 真正的难点不在后端，而在 **3D 算子覆盖**（FlexiCubes / NeuralRender / 3D conv）。策略：可移植的算子走 GPU，ggml 后端缺失的 3D 算子固定到 CPU，保证三后端数值一致。
- 精度对齐用 trellis 的 **parity 流程**：Python dump 参考激活 → C++ 逐层 tap 比对 `atol + rtol*|ref|`。

---

## 2. 推理管线拆解与算子映射

InstantMesh 推理本质（信息流）：

```
输入图
 ├─ (A) Background removal   rembg/BiRefNet
 ├─ (B) 多视图扩散            Zero123++ UNet + VAE + Euler scheduler  → 6 视角图
 └─ (C) 重建 LRM
      ├─ DINO ViT-B/16 特征
      ├─ TriplaneTransformer → triplane
      ├─ FlexiCubes 几何
      └─ NeuralRender 渲染
```

| ID | 组件 | 主要算子 | CUDA | Vulkan | 移植难度 | 备注 |
|----|------|---------|:----:|:------:|:--------:|------|
| A | rembg(BiRefNet) | 2D conv / swin | ✅ | ✅(需补op) | 中 | 直接复用 trellis `RMBG-2.0-GGML` |
| B | Zero123++ UNet | 2D conv / attn | ✅ | ✅ | 高 | 参考 stable-diffusion.cpp |
| B | VAE decoder | 2D conv/up | ✅ | ✅ | 中 | 同上 |
| B | scheduler | 纯 CPU 数值 | ✅ | ✅ | 低 | Euler, 无张量后端 |
| C | DINO ViT-B | 2D transformer | ✅ | ✅ | 低 | 参考 trellis DINOv3 |
| C | TriplaneTransformer | 2D attn | ✅ | ✅ | 低 | 16 层 1024 维 |
| C | FlexiCubes 几何 | 3D SDF/deform | ⚠️ | ⚠️ | 高 | 缺少体素算子,pin CPU |
| C | NeuralRender | ray-marching | ⚠️ | ⚠️ | 高 | 自制采样 op 或 pin CPU |
| C | UV / mesh 导出 | 纯 CPU | — | — | 低 | mesh_export |

**后端一致性的关键纪律**：任何算子只要 CUDA 与 Vulkan 行为不同，就统一 pin 到 CPU（如 3D conv），而不是让两端各跑各的 → 保证三后端精度一致。

---

## 3. 精度对齐方案（parity）

复用 trellis **深度-anything 式分层验证**：

1. **参考容器**：`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`，仅官方 pip 依赖；3D 稀疏算子 monkeypatch 为纯 torch gather-GEMM（可跑 CPU，兼作移植规范）。
2. **参考激活 dump**：Python forward hook 把每层激活写成 `.gguf` + manifest，C++ 用 ggml API 直接读。
3. **C++ 侧逐层比对**：`tests/parity.hpp` 的 `atol + rtol*|ref|` 门限（默认 2e-3），整层用 `SKIP_RETURN_CODE 77` 缺 fixture 时跳过。
4. **三后端联合验证**：同一输入分别跑 CPU/CUDA/Vulkan，三者互比 + 各自与 PyTorch 比。f16/q8 与 f32 的误差需在量化容差内（见 §5）。

---

## 4. 性能目标

| 后端 | 目标 | 参照 |
|------|------|------|
| CUDA | **快于 PyTorch CUDA** | trellis 实测 ggml CUDA 142.7s vs PyTorch 317.9s（2.2×） |
| Vulkan | **快于 PyTorch CUDA** | trellis 实测 ggml Vulkan 136.1s（2.3×） |
| CPU | 可运行即可（无硬性速度要求） | 用于精度回归 |

关键性能决策：
- 注意力一律 `ggml_flash_attn_ext`（与全 softmax 位级一致，CPU 上也是）。
- 2D conv / transformer / 扩散走 GPU；3D conv 解码按 trellis pin CPU（其 GPU 版尚不成熟，先求一致）。
- 用 `TRELLIS2_TIMING` 式逐阶段 wall-clock 定位瓶颈，避免对错误阶段做优化。

> 注：`> PyTorch` 的达成主要来自两处——(1) 扩散/Transformer 的 CUDA/Vulkan 原生 kernel；(2) 3D 稀疏算子用原生 kernel 替代 PyTorch 纯 Python 稀疏循环（trellis 的 3D conv 达 20×）。

---

## 5. GGUF 多精度（f32 / f16 / q8）

### 5.1 转换目标
**所有子模型**（BiRefNet、Zero123++ UNet、VAE、DINO、TriplaneTransformer，以及 FlexiCubes/NeuralRender 若含权重）都产出三种 GGUF：

| 精度 | ggml 类型 | 用途 |
|------|-----------|------|
| f32 | `GGML_TYPE_F32` | 精度基准，双精度回归 |
| f16 | `GGML_TYPE_F16` | 默认推理，GPU 友好 |
| q8 | `GGML_TYPE_Q8_0` | 低显存/低带宽，CPU 与移动端 |

### 5.2 要求
- GGUF 内写入 `general.architecture` 与各子模型 hparams KV，供 C++ loader 按架构分发。
- 权重张量名与 PyTorch state_dict 一一对应（`lrm_generator.` 前缀剥离）。
- 转换脚本幂等、可重跑；`quantize_to_q8.py` 式从 f16 量化到 q8_0。

### 5.3 精度容差
- f16 相对 f32：逐层 `atol+rtol*|ref|` 门限放宽到约 1e-2（半精度容差）。
- q8 相对 f32：约 1e-2 ~ 5e-2。**三后端必须使用同一份量化权重**，禁止按后端区分量化 → 保证跨后端可比。

---

## 6. 工程结构

```
InstantMesh-GGML/
├── CMakeLists.txt            # 根构建：vendor ggml + 双后端
├── src/
│   ├── core/
│   │   ├── backend.hpp/.cpp  # 后端自动探测(CPU/CUDA/Vulkan)
│   │   ├── gguf_io.hpp/.cpp  # GGUF 读取(KV + 权重)
│   │   └── tensor.hpp/.cpp   # 张量封装(命名->tensor 索引)
│   ├── models/
│   │   ├── dino.cpp          # DINO ViT-B/16
│   │   ├── unet.cpp          # Zero123++ UNet
│   │   ├── vae.cpp
│   │   ├── transformer.cpp   # TriplaneTransformer
│   │   ├── flexicubes.cpp    # 几何(可 pin CPU)
│   │   └── neural_render.cpp # ray-marching
│   └── pipeline.cpp          # 管线编排(rembg->扩散->LRM->mesh)
├── convert/                  # GGUF 转换器 (Python, 仅转换期)
│   ├── convert_*.py          # 各子模型转换
│   └── quantize_to_q8.py
├── tests/                    # parity 回归
│   ├── parity.hpp            # atol+rtol 门限
│   └── test_*.cpp
├── src/tools/                # C++ CLI 工具 (examples/ 仅存放输入图片)
│   └── instantmesh.cpp       # CLI: 图->OBJ (DINO→LRM→OSGDecoder→FlexiCubes+顶点色)
└── third_party/ggml/         # ggml v0.18.1 submodule (含 flash_attn head_dim=64 修复)
```

---

## 7. 实施顺序（每步可交付、可验证）

1. **基础设施**：根 `CMakeLists.txt`（`-DGGML_CUDA=ON -DGGML_VULKAN=ON`）、`src/core/backend` 后端探测、参考容器与权重、parity 框架。→ **本阶段交付**
2. **DINO 移植 + parity**：图像→triplane 的 DINO/Transformer 部分端到端在 C++ 跑通。
3. **Zero123++ 扩散 + VAE + scheduler 移植 + parity**：单图→多视图。
4. **rembg(BiRefNet) 接入**：复用 trellis 子模块。
5. **几何/渲染**：FlexiCubes + NeuralRender，3D 算子先 pin CPU 求一致。
6. **性能调优**：逐阶段计时，flash-attn、GPU 化 2D 算子、必要时补 3D 算子。
7. **GGUF 三精度全量** + 三后端联合回归 + 基准对比。

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| 3D 算子(体素/ray-march)后端缺失 | CUDA/Vulkan 无法全 GPU | 首期 pin CPU 求一致；后续补自定义 op |
| 自定义 op(Vulakn shader)工作量大 | Vulkan 覆盖不全 | 复用 trellis `ggml-rmbg-ops.patch`；每个 op 双后端各写一份并做 parity |
| 量化引入跨后端误差 | 精度标注失效 | 三后端共用同一量化权重 + 按精度设独立门限 |
| ggml 上游变动 | 双后端回归 | **冻结 submodule 版本** + 维护 `patches/` 目录 |
| 参考生成依赖 GPU/显存 | parity 无法跑 | 分阶段 dump + 稀疏算子 monkeypatch(可 CPU) |

---

## 9. 验收标准

- [ ] `python run.py` 的 PyTorch 输出与 C++ `instantmesh` 输出（同 seed/同输入）在网格几何上一致（可设容差）。
- [ ] CUDA 与 Vulkan 推理耗时均 < PyTorch CUDA 基线。
- [ ] 每个子模型均有 f32/f16/q8 三份 GGUF，`general.architecture` 正确。
- [ ] CPU/CUDA/Vulkan 三后端在 f32 下逐层 parity 通过；f16/q8 在各自容差内通过。
- [ ] 运行期零 PyTorch 依赖（`ldd` 无 torch，且无 `import torch`）。

---

## 10. 当前落地状态

已完成（本仓库）：

- ✅ `docs/PLAN.md`：纯 C++ 集成、精度对齐、性能目标、f32/f16/q8 多精度方案。
- ✅ **C++ 工程骨架**：[CMakeLists.txt](../CMakeLists.txt) vendor `third_party/ggml`；后端自动探测 [src/core/backend.hpp](../src/core/backend.hpp)、GGUF 加载 [src/core/gguf_io.hpp](../src/core/gguf_io.hpp)。
- ✅ **GGUF 转换框架**（`convert/`）：`convert_common.py`（f32/f16/q8 三精度 + 每张量类型策略）、`convert_lrm.py`（LRM 拆分为 dino / lrm_transformer / geometry / synthesizer 四份）、`convert_all.py`（批量三精度）。已用合成 ckpt 对 f32/f16/q8 全量验证，且 C++ 加载器能正确读回 `general.architecture` 与张量。
- ✅ **后端验证**：CPU 构建通过；**CUDA 构建通过**，运行时正确探测 RTX 3060 并在 GPU 上加载 GGUF 权重。
- ✅ **Vulkan 后端**：见下方说明，双后端（CUDA+Vulkan）构建通过，运行时两种 GPU 后端均能初始化 RTX 3060。
- ✅ **DINO 编码器移植**：[src/models/dino.hpp](../src/models/dino.hpp)/[dino.cpp](../src/models/dino.cpp) —— 完整 adaLN DiT 图：patch conv → cls → pos-enc → 12 层（adaLN 调制 + flash 注意力 + qkv-bias + GELU MLP），运行于 CPU/CUDA/Vulkan 任一后端；CLI [src/tools/dino.cpp](../src/tools/dino.cpp)。
- ✅ **ggml 升级到 v0.18.1 并启用 flash_attn**：submodule 从 v0.18.0 切到 v0.18.1（`git submodule update third_party/ggml`）。用 `ggml_flash_attn_ext` 替换 DINO/TriplaneTransformer 中的**手动 softmax 注意力**（`mul_mat`+`soft_max_ext`+`mul_mat`）。v0.18.1 修复了 v0.18.0 中 flash_attn 在 head_dim=64 的 bug——用隔离测试对三种真实配置（DINO 自注意 hd=64/nh=12/seq=197、LRM 自注意 hd=64/nh=16/seq=3072 tiled 路径、LRM 交叉注意 q=3072/kv=197）验证 flash_attn 与手动 softmax 逐元素一致（corr=1.0，maxdiff≈1e-7）。替换后三精度 CPU parity 保持不变（DINO rel≈1.2e-3、LRM f32/f16/q8 = 1.2e-5 / 1.7e-3 / 3.4e-2），且 flash 路径在 GPU 上显著更快。
- ✅ **TriplaneTransformer 移植**：[src/models/lrm_transformer.hpp](../src/models/lrm_transformer.hpp)/[lrm_transformer.cpp](../src/models/lrm_transformer.cpp) —— 16 层 1024 维 BasicTransformerBlock（LayerNorm + 交叉注意力 + 自注意力 + GELU MLP，均带残差）、adaLN 调制、最终 LayerNorm、转置卷积上采样为 triplane；CLI [src/tools/lrm_transformer.cpp](../src/tools/lrm_transformer.cpp)。f32/f16/q8 三精度 CPU parity 全部通过（rel≈1.1e-5 / 1.7e-3 / 3.4e-2）。关键踩坑：
  - 注意力用 `ggml_flash_attn_ext`（v0.18.1 起，此前 v0.18.0 在 head_dim=64 有 bug 才退化为手动 softmax）。
  - MLP 用 `ggml_gelu_erf`（PyTorch `nn.GELU()` 默认精确 erf），否则 tanh 近似逐层累积误差。
  - `ggml_conv_transpose_2d_p0` **不支持 batch>1**（只算 batch-0 切片），需逐平面 batch=1 跑再 `ggml_concat` 沿 batch 维拼接；且要求 F32 输入，量化模型的激活需先 `ggml_cpy` 转 F32。
  - 激活始终以 F32 运行（量化仅作用于权重）：初始 `pos_embed` 若为 F16 需先转 F32，否则后续 `ggml_norm`/`conv_transpose` 断言失败。
- ✅ **几何预测 / OSGDecoder 移植**：[src/models/synthesizer.hpp](../src/models/synthesizer.hpp)/[synthesizer.cpp](../src/models/synthesizer.cpp) —— FlexiCubes 的神经部分（含权重）：宿主侧 triplane 双线性采样（`grid_sample` align_corners=False，3 平面投影为 (x,y)/(x,z)/(z,y)）+ 8 立方角特征 gather，再以单个 ggml 图跑 sdf / deformation / weight 三个 4 层 Relu MLP；CLI [src/tools/synthesizer.cpp](../src/tools/synthesizer.cpp)、参考 [convert/parity_synth.py](../convert/parity_synth.py)。64×64 平面 + grid_res=64 下三精度 CPU parity：f32 rel≈3e-7、f16≈2-7e-4、q8 可接受（sdf 符号翻转率 0.003%）。关键点：
  - 采样是固定几何操作（无权重），在宿主 C++ 实现；`px=(nx+1)/2*W-0.5`，4 邻域加权、越界填 0。
  - GGUF 张量名带 `decoder.` 前缀（去 `synthesizer.` 后），`get_t` 需加上。
  - OSGDecoder 的 weight 网络输入按立方 gather 8 角特征（`decoder.net_weight.0.weight` 输入 8*3C=1920），输出 ×0.1。
- ✅ **FlexiCubes 网格提取移植**：[src/models/flexicubes.hpp](../src/models/flexicubes.hpp)/[flexicubes.cpp](../src/models/flexicubes.cpp)、表头 [flexicubes_tables.hpp](../src/models/flexicubes_tables.hpp)（由 [convert/gen_flexicubes_tables.py](../convert/gen_flexicubes_tables.py) 生成）、CLI [src/tools/flexicubes.cpp](../src/tools/flexicubes.cpp)、参考 [convert/parity_flexicubes.py](../convert/parity_flexicubes.py)。纯 C++ 固定查找表算法（无权重）：表面立方体选取 → 对偶顶点（含 `_linear_interp` 边插值）→ 三角化。parity：顶点 max_abs≈6e-8（float32 极限）、面完全一致。关键踩坑：
  - `DMC_TABLE` 是 `[256][4][7]` 三维数组，索引用 `DMC_TABLE[case][v][e]`。
  - 对偶顶点 `ue = (x0*w1 - x1*w0)/(w1 - w0)`（w0↔v0、w1↔v1），权重绑定错位会导致约 9% 顶点误差。
- ✅ **端到端管线整合**：[src/tools/instantmesh.cpp](../src/tools/instantmesh.cpp) 把 DINO → TriplaneTransformer → OSGDecoder → FlexiCubes 串成单命令 `图片 → mesh.obj`；输入预处理由 [convert/prep_input.py](../convert/prep_input.py) 生成（多视角图 `[V,3,224,224]` + 相机 `[V,16]`）。宿主 C++ 内置 `construct_voxel_grid`（voxel 网格 + 去重 cube 索引）、`center_boundary_index`、默认 zero123plus 相机（radius=4.0/fov=30）与 deformation tanh 归一化、空形状 sdf fix。修复多视角 batch 相关断言：
  - `ggml_concat` 前 CLS token 需 `ggml_repeat` 广播到 batch 维。
  - `ggml_flash_attn_ext` 输出对 batch>1 非连续，reshape 前需 `ggml_cont`。
  - adaLN 4 段 `ggml_view_2d` 是 padded 非连续视图，`modulate` 里 reshape 到 `[hidden,1,B]` 前需 `ggml_cont`（batch=1 时因 `ggml_is_contiguous` 对 `ne==1` 维跳过检查而侥幸通过）。
  - 真实图片跑通：`sdf` 分布合理（大部分正、局部负形成表面），产出含 bbox 收敛的 mesh；mesh 几何质量受限于输入（单视角复制无法替代 Zero123++ 多视角生成）。
- ✅ **顶点颜色（net_rgb）**：[src/models/synthesizer.cpp](../src/models/synthesizer.cpp) 新增 `synthesizer_texture_forward`，用 synthesizer 的 `net_rgb` 分支为每个网格顶点采样 triplane 计算 RGB（`sigmoid` + MipNeRF clamp），与几何同一 triplane。parity（[convert/parity_synth.py](../convert/parity_synth.py) 的 rgb 对比）：`max_abs≈3.3e-7`。OBJ 输出形如 `v x y z r g b`（RGB ∈ [0,1]）。
- ✅ **端到端颜色/几何校验（benchmarks/）**：新增 [benchmarks/pytorch_reference.py](../benchmarks/pytorch_reference.py)（官方 `LRMGenerator` 上采样顶点色参考）、[benchmarks/run_bench.sh](../benchmarks/run_bench.sh)（CPU/CUDA/Vulkan × f32/f16/q8 端到端基准）、[benchmarks/analyze.py](../benchmarks/analyze.py)（报告 + 图）、[benchmarks/render_meshes.py](../benchmarks/render_meshes.py)（对比渲染）、[benchmarks/color_parity_e2e.py](../benchmarks/color_parity_e2e.py)（在相同网格顶点上比对两端 net_rgb）。结论：**ggml 与 PyTorch 顶点颜色 Δmean≈0.002、顶点数基本一致**（如 blue_cat 14550 vs 14558）。
- ✅ **输入归一化 bug 修复**：`benchmarks/mv/<img>/image.bin` 已是 ImageNet 归一化张量，PyTorch 参考原先经 `ViTImageProcessor` 再归一化一次（双重归一化），导致参考三平面/网格/颜色严重偏离 ggml（Δmean≈0.25）。修复后参考直接向 DINO 输入同一份归一化张量，Δmean 降至 ~0.002。**ggml 一侧一直是正确的。**
- ✅ **网格分辨率对齐**：因 RTX 3060（12GB）无法容纳 grid_res=128（需 ~18GB），ggml 与 PyTorch 参考统一用 **grid_res=88** 以同条件对比；`--grid-res` 由 [benchmarks/run_bench.sh](../benchmarks/run_bench.sh) 控制。

### 未完成 / TODO

- ❌ **Zero123++ 多视图扩散 + VAE + scheduler**（单图→6 视角）尚未移植；当前输入由 [convert/gen_multiview.py](../convert/gen_multiview.py)/[convert/prep_input.py](../convert/prep_input.py) 预处理（非真正 Zero123++ 生成，限制重建质量）。
- ❌ **rembg(BiRefNet) 前景分割**尚未接入。
- ❌ **NeuralRender 渲染**（ray-marching）尚未移植。
- ❌ **OBJ→GLB/纹理贴图导出**（`--export_texmap`）未实现。
- ⏳ **Vulkan 全量 parity**：已能初始化并跑 GPU，尚未对三精度做完整逐层 parity 回归（CUDA 已对齐）。
- ⏳ 更大网格分辨率（grid_res=128）受显存限制未验证。

构建命令：

```bash
# CPU
cmake -B build -DINSTANTMESH_BUILD_TESTS=ON && cmake --build build -j
# CUDA (推荐 GPU 路径)
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-cuda -j
# 双后端 (CUDA + Vulkan)
cmake -B build-gpu -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build-gpu -j
```

### Vulkan 构建说明（已解决）

本机工具链：Vulkan SDK 位于 `~/VulkanSDK`，可运行的 glslc 是 `/usr/local/bin/glslc`（VulkanSDK 自带的 glslc 是 2026.x 新版，需更高 glibc 无法运行）。配置时需显式指定 glslc：

```bash
export VULKAN_SDK=$HOME/VulkanSDK/1.4.350.0/x86_64
cmake -B build-gpu -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DVulkan_GLSLC_EXECUTABLE=/usr/local/bin/glslc
cmake --build build-gpu -j
```

**已在本机验证**：双后端构建通过；运行时两种 GPU 后端均能正确初始化 RTX 3060 —— CUDA 与 Vulkan 两条路径都可用。三后端（CPU/CUDA/Vulkan）自动探测逻辑在 `test_backend` 中通过。