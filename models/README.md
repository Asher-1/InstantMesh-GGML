# models/ — 推理模型说明与 Benchmark

本目录存放 InstantMesh 端到端重建管线所需的 3 个 ggml 量化模型。管线为：

```
图片 → [DINO 图像编码器] → [LRM 三平面生成器] → [OSGDecoder 体素解码] → FlexiCubes → mesh
```

每个模型都有 `f32 / f16 / q8` 三种精度（`.gguf`）。下表与下文的数据来自
`benchmarks/results/report.md`（RTX 3060，输入为 Zero123++ 生成的多视角，
grid_res=64，端到端跑批 4 张图 × 3 后端 × 3 精度 = 36 次）。

---

## 1. 模型总览

| 模型 | 文件前缀 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| **dino** | `dino_*.gguf` | 图像主干（DINOv2 视觉 Transformer），提取每张输入图的多尺度 patch 特征 | 单张/多视图 RGB（224×224，ImageNet 归一化） | 图像 token 序列特征 |
| **lrm_transformer** | `lrm_transformer_*.gguf` | 生成式重建主干，把图像特征融合为三平面（triplane）潜变量 | 图像 token 序列 | triplane [3, C, H, W] 潜变量 |
| **synthesizer** | `synthesizer_*.gguf` | OSGDecoder，在体素网格上对 triplane 采样并解码 | triplane + 采样点坐标 | SDF、deformation、weight（供 FlexiCubes 建 mesh） |

### 各模型作用详解

- **dino（DINOv2 图像编码器）**
  类似 InstantMesh 原版 DINOv2 backbone，是一个视觉 Transformer（ViT），
  把输入图片切成 patch 后做自注意力编码，输出 token 序列。它决定了"模型能
  从图像里看到多少细节"，是所有后续重建的输入特征来源。

- **lrm_transformer（Large Reconstruction Model 主干）**
  这是重建的核心。它把 DINO 给出的图像特征投影后，通过若干层交叉/自注意力，
  输出一个三平面（triplane）表示——三个正交特征平面把物体隐式编码在一个
  紧凑的低维空间里，相比直接回归体素/点云更省内存且利于高质量重建。

- **synthesizer（OSGDecoder）**
  负责把三平面"翻译"成几何：给定三平面和一个采样点，它按点投影到三个平面取
  特征后拼接，再经一个小 MLP 解码出该点的 **SDF**（有符号距离场）、
  **deformation**（网格顶点形变）和 **weight**（FlexiCubes 权重）。
  这三个量最终由 FlexiCubes 算法提成三角网格。

> 三者串联成一条链：**没有 dino 就没有图像特征，没有 LRM 就没有三平面，
> 没有 synthesizer 就没有 SDF/mesh**。量化任一个都会影响端到端精度，
> 但它们的体量差异极大（见下）。

---

## 2. 量化体积（符合预期）

| 模型 | f32 (MB) | f16 (MB) | q8 (MB) | f16/f32 | q8/f32 |
|---|---|---|---|---|---|
| dino | 459.0 | 230.1 | 123.4 | **0.50x** | **0.27x** |
| lrm_transformer | 1054.8 | 527.8 | 284.0 | **0.50x** | **0.27x** |
| synthesizer | 0.8 | 0.4 | 0.3 | **0.50x** | **0.32x** |

- f16 恒为 f32 的一半，q8 约为 f32 的 1/4，符合量化理论体积比。
- 体量上 **lrm_transformer 最大**（约 1 GB f32），**dino 次之**，**synthesizer 可忽略**。
  因此整体显存占用主要由 lrm_transformer 决定。

---

## 3. 推理性能（端到端，秒/图，三模型串联）

| 后端 | f32 | f16 | q8 |
|---|---|---|---|
| CPU | 18.01 | 22.03 | 17.73 |
| CUDA | 4.73 | 3.72 | **3.63** |
| Vulkan | 5.76 | 5.32 | **4.75** |

- **CUDA/Vulkan 比 CPU 快约 4–5x**（f32 下 CUDA 3.8x、Vulkan 3.1x）。
- GPU 上 **q8 最快**，f16 次之，f32 最慢——量级越小计算越省。
- **CPU 上 f16 反而比 f32 慢**（22s vs 18s）：这是 ggml CPU 端 f16→f32
  反量化开销导致的已知现象；CPU 上选 f32 或 q8 更合适。
- 由于 lrm_transformer 体量最大，端到端耗时主要被它占据，dino 次之，
  synthesizer 占比极小。

---

## 4. 精度表现（SDF 相对误差，mean|sdf|≈8.5）

### 4.1 纯量化误差（同后端，相对各自 f32）
| 后端 | f16 相对RMSE | q8 相对RMSE |
|---|---|---|
| CPU | 0.18% | 2.90% |
| CUDA | 0.81% | 2.87% |
| Vulkan | 0.77% | 2.05% |

### 4.2 后端一致性（f32 相对 CUDA-f32）
| 后端 | 相对RMSE |
|---|---|
| CPU | 0.64% |
| Vulkan | 0.73% |

- **f16 精度很高**（相对误差 <1%），几乎不损失重建质量，推荐日常使用。
- **q8 精度可接受**（2–3%），网格几何基本不变（SDF 符号翻转率 <0.7%），
  适合显存/带宽紧张或追求速度的场景。
- **后端一致性良好**：同一精度下 CPU/CUDA/Vulkan 结果差异 <1%，说明量化
  与后端实现一致，不存在某个后端"偷偷掉精度"的情况。

---

## 5. 各模型使用注意事项

### 通用
- 三个模型**必须配套使用同一精度**（如 `dino_f16 + lrm_transformer_f16 +
  synthesizer_f16`），混合精度可能导致层间数值不匹配。
- 模型只含权重，**不含归一化统计/预处理参数**；输入需按
  DINOv2 规则做 ImageNet 归一化（见 `src/tools/instantmesh.cpp`）。
- `.gguf` 文件被 `.gitignore` 忽略（大二进制不入库），目录用 `.gitkeep` 保留。

### dino
- 输入分辨率固定为 **224×224**；输入尺寸不符会报错或退化（见工具内校验）。
- 决定特征质量，一般不要低于 f16；若追求极致速度且接受轻微质量损失再考虑 q8。

### lrm_transformer
- **体量最大**，是显存/内存占用与加载耗时的主要来源。
- 输出 triplane 的通道/尺寸由模型超参决定，需与 synthesizer 的 `plane_dim` 匹配。
- 量化对本模型影响最明显（参数量最大），q8 时若重建出现几何畸变，建议回退 f16。

### synthesizer
- 体量极小（<1 MB），量化影响可忽略，**建议直接用 f32/f16，避免无谓精度损失**。
- 它的 `plane_dim` 必须与 lrm_transformer 输出匹配，否则解码错位。

### 后端选择建议
- 有 NVIDIA GPU：优先 **CUDA**；跨厂商/无 CUDA 的 GPU 用 **Vulkan**（性能略低于 CUDA）。
- 无 GPU：CPU 上**避免 f16**，用 f32 或 q8。

---

## 6. 复现与扩展

- 跑批脚本：`bash benchmarks/run_bench.sh`（生成 `benchmarks/results/times.csv` 与各 run 的 `.sdf.bin`/`.obj`）
- 分析/绘图：`python3 benchmarks/analyze.py`（生成 `report.md`、`latency.png`、`precision_rmse.png`、`precision_maxerr.png`）
- 手动推理：`build-cuda/instantmesh --dino models/gguf/dino_f16.gguf --transformer models/gguf/lrm_transformer_f16.gguf --synthesizer models/gguf/synthesizer_f16.gguf --image <img.bin> --camera <cam.bin> --device gpu`
