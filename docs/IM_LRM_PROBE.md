# IM_LRM_PROBE：NaN 二分定位探针使用指南

`IM_LRM_PROBE=1` 是 `models/lrm_transformer.cpp` 内置的数值诊断探针，用于在
单次前向中捕获计算图各节点的 NaN / 数值异常，快速定位"输出全 NaN"类问题的
确切触发层和算子。本文说明其用法、定位方法论，以及如何将同样的模式移植到
其他模型。

## 1. 快速开始

```bash
cd cpp_ggml
IM_LRM_PROBE=1 ./build-gpu/lrm_transformer --device cpu \
    models/gguf/lrm_transformer_f16.gguf
```

不带 `IM_LRM_PROBE` 时探针完全关闭（零开销：不注册 tap、不打印、
`ggml_set_output` 也不会被调用，内存复用行为与正常路径一致）。

## 2. 输出解读

每次运行输出两类信息：

```text
probe pre-compute cond: amax=0.25 size=151296    # 输入写入自检
probe init_x      : nan=0/3145728 amax=0.302002  # 初始 token 流（pos_embed）
probe L0.norm1    : nan=0/3145728 amax=4.09583   # L0 cross-attn 的 norm1 输出
probe L0.q        : nan=0/3145728 amax=10.4975   # q/k/v 投影
probe L0.k        : nan=201728/201728 amax=0     # ← 全 NaN，问题在这
probe L0.xattn.qh : ...                          # to_head 重排后的 q/k/v
probe L0.xattn.fa : ...                          # flash_attn_ext 原始输出
probe L0.outproj  : ...                          # out_proj 输出
probe L0_cross    : ...                          # cross-attn 残差后
probe L0_self     : ...                          # self-attn 残差后
probe layer 0     : ...                          # MLP 残差后（整层输出）
...
probe layer 15    : ...                          # 最后一层
```

每个 tap 报告三项：

| 字段 | 含义 | 判读 |
|---|---|---|
| `nan=N/total` | NaN 元素个数 / 总元素数 | 第一个 `nan>0` 的 tap 就是污染点或其直接下游 |
| `amax` | 非 NaN 元素的最大绝对值 | `amax=0` 且全 NaN 说明整块被污染；`amax` 异常巨大（如 >1e10）说明输入可能是垃圾数据但尚未溢出 |
| tap 名 | 注册时的标签 | 见 §3 的 tap 列表 |

**关键判读规则**：NaN 是顺着残差连接传播的，所以"第一个出 NaN 的节点"才是
源头，其后所有层都会显示全 NaN——不要被满屏 NaN 吓到，只需看最早的那个。

## 3. Tap 列表与二分流程

### 第一轮：全层扫描

先只看 `layer N` / `init_x`。确定 NaN 从哪一层开始出现（如 `layer 0` 全 NaN
而 `init_x` 正常 → 问题在 L0 内部）。

### 第二轮：L0 子阶段细分

首轮已内置 L0 的细粒度 tap，直接看：

1. `L0.norm1` — LayerNorm 输出。NaN → 输入流已被污染（往前查）。
2. `L0.q` / `L0.k` / `L0.v` — 三个投影的 `mul_mat` 输出。
   - 只 k/v NaN 而 q 正常 → 问题在 k/v 的输入（外部 cond）或其权重。
3. `L0.xattn.qh/kh/vh` — head 重排（reshape/permute/cont）之后。如果投影
   正常而这里 NaN → 重新排布算子的问题。
4. `L0.xattn.fa` — `ggml_flash_attn_ext` 原始输出。NaN 且 q/k/v 正常 →
   attention 计算本身（如 scale、mask、类型支持）。
5. `L0.outproj` — out_proj 后。

### 第三轮：输入 / 权重自检

第二轮指向某输入或权重时，看这几个特殊 tap（它们读的是 loader 分配的
buffer，不经过计算）：

- `cond` — 外部输入本身。**若 amax 异常巨大，先怀疑宿主侧传入的数据**。
- `L0.cq_w` / `L0.ck_w` / `L0.cv_w` — cross-attn 权重（f16 权重会自动转
  f32 后统计）。

### 实战案例：size_t 下溢 → f16 全 NaN

2026-09 定位的一次真实问题，完整走完了上述流程：

1. 第一轮：`init_x` 正常（amax 0.302），`layer 0` 起全 NaN → 问题在 L0。
2. 第二轮：`L0.q` 正常（f16 权重 × f32 激活），`L0.k`/`L0.v` 全 NaN →
   排除算子/后端问题，指向 k/v 的共同输入 `cond` 或其权重。
3. 第三轮：权重全部正常，`cond` 读回 amax=9.2e17（应为 0.25）→ 宿主侧
   `tensor_set` 写入的数据本身就是垃圾。
4. 根因：tool 的合成输入 `((i % 10) - 5) * 0.05f` 中 `i` 是 `size_t`，
   `i%10 < 5` 时无符号下溢为 ~2^64；f16 matmul 把 f32 输入转 f16 时
   9.2e17 → inf → NaN。f32 模型不转 f16，巨大值被 softmax 饱和 +
   LayerNorm 归一化掩盖，最终 stats"假正常"。

**教训**：判断"哪个精度/后端有问题"时不要只看最终输出的 mean/std——
softmax 饱和和 LayerNorm 能把中间的垃圾数据掩盖掉。f32 显示"正常"不代表
它没有吃到同样的坏输入。

## 4. 实现要点（移植到其他模型时）

探针代码在 `src/models/lrm_transformer.cpp`，核心约 40 行，移植要点：

1. **单一全局 sink**：`ProbeSink g_probe`（taps + tags 两个 vector，
   tags 必须用 `std::string` 存储——不要把栈上 char buffer 的指针存进
   vector，循环重用会覆盖标签）。
2. **注册即标记 output**：`ggml_set_output(t)`。ggml-alloc 永不复用
   output 节点的存储，这是所有 tap 能在一次完整 compute 后仍可读的前提。
3. **单次全图 compute**：绝不要"跑到第 N 层 compute 一次"做多次截断式
   compute——gallocr 释放后残留的已分配指针会导致 use-after-free 段错误
   （`tensor buffer not set` / memmove crash 的来源）。
4. **compute 后统一读取**：`ggml_backend_tensor_get` 逐 tap 读回，按
   `t->type` 解释（F32 直接 memcpy；F16 用 `ggml_fp16_to_fp32` 转换）。
5. **权重 tap 直接读 loader buffer**：权重节点已有 buffer，gallocr 会
   跳过重分配，直接 get 即可。
6. **开销**：output 节点阻止内存复用会增大峰值内存，探针仅用于诊断，
   不要在性能测试或生产路径开启。

### 环境变量约定

| 变量 | 作用 |
|---|---|
| `IM_LRM_PROBE=1` | 开启 lrm_transformer 的 NaN 探针 |
| `IM_UNET_PROBE=1` | UNet 每次前向一行 summary（w/r eps 的 nan/amax） |
| `IM_UNET_PROBE=2` | UNet 额外注册全部 stage tap 并逐 tap 打印 |
| `IM_UNET_PROBE_T=<int>` | UNet stage tap 仅在匹配 timestep 的 call 打印 |
| `IM_DUMP_STAGES=<substr>` | UNet stage tap 名称子串过滤（与 probe=2 组合） |

## 5. zero123pp UNet 探针（IM_UNET_PROBE）

扩散模型与 LRM 的调用模式不同：UNet 每个去噪步调用一次前向（w/r 两个 pass
在同一张图里），75 步 = 150 个 pass / 75 次 compute。探针据此做了两级设计
（实现于 `src/models/unet.cpp`，复用 `IM_VAE_DUMP` 的 tap 基础设施）：

### 用法一：全流程 summary 扫描（`=1`）

```bash
IM_UNET_PROBE=1 ./build-gpu/zero123pp --image ../examples/blue_cat.png \
    --device cpu --steps 75 --seed 42 2>&1 | grep unet_probe
```

每个 call 打印一行对：

```text
unet_probe w.eps          nan=0/32768 amax=4.77203    # w-pass eps（cond 尺度）
unet_probe r.eps          nan=0/76800 amax=0.842033   # r-pass eps（去噪尺度）
```

75 行一眼看出 NaN 从哪个 timestep 开始出现。若 `amax` 逐步异常增长也
值得关注（数值发散通常先于 NaN）。

### 用法二：单步 stage 二分（`=2`）

summary 定位到起始 timestep 后，用 `IM_UNET_PROBE_T` 锁定该步看内部 tap：

```bash
IM_UNET_PROBE=2 IM_UNET_PROBE_T=979 IM_DUMP_STAGES=d0 \
    ./build-gpu/zero123pp --device cpu --steps 75 --seed 42
```

tap 覆盖两类点（名字与 `IM_VAE_DUMP` 一致）：

- **dmark 主干**：`w.conv_in` / `w.d0.r0` / `w.d0.attn0` / ... / `w.mid` /
  `wup0.upc` / `w.conv_out`（r-pass 同名前缀 `r.`）——resblock/attention
  输出与上采样链，定位第一个异常 block。
- **u_mark 内部**：`*.norm1` / `*.silu1` / `*.conv1` / `*.te` / `*.gn2_bare`
  / `*.norm2` / `*.conv2`（resnet 内部）与 `*.tgn` / `*.tgather` / `*.tproj`
  （transformer 内部）——定位 block 内确切算子。

`IM_DUMP_STAGES` 子串过滤可缩小打印量（如 `IM_DUMP_STAGES=mid` 只看
mid_block 相关 tap）。

### 与 LRM 探针的差异（重要）

| 事项 | LRM（单次调用） | UNet（150 次调用） |
|---|---|---|
| 详细 tap gate | 不需要 | `IM_UNET_PROBE_T`（默认仅第 1 个 call 详细） |
| tap 实现 | `ggml_set_output` 标记原节点 | 复用 `IM_VAE_DUMP` 的 `ggml_cont` 拷贝立即 expand |
| 内存开销 | output 阻止复用，一次性 | 拷贝增加峰值内存，75 步循环会累积 |

两个 UNet 特有注意点：

1. **默认只详细打印第一个 call**（未设 `IM_UNET_PROBE_T` 时）。NaN 若在
   后期 step 才出现，务必用 `IM_UNET_PROBE_T=<该步 timestep>` 锁定。
2. **stage tap 优先在 CPU 上跑**：每个 tap 是一份 `ggml_cont` 拷贝，
   ~50 个 tap 增加数百 MB 峰值内存；12GB Vulkan 路径的 VAE decode 本就
   临近 OOM，全 tap 可能直接爆显存。summary 模式（`=1`）只给图尾
   `set_output`，开销可忽略，GPU 上可用。

### 验证记录（2026-09-18）

- `IM_UNET_PROBE=1/2` 2 步短跑输出正常，全 tap `nan=0`、amax 合理。
- probe1/probe2 的最终 latents 与不带探针 **bit-exact**——探针对数值零影响。
