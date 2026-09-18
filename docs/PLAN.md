# InstantMesh-GGML Pure C++ Integration Plan

Goal: a **native ggml port** of the full InstantMesh inference pipeline with no PyTorch
runtime dependency (Python is used only at conversion time to generate references and
GGUF), achieving:

1. **Pure C++/ggml runtime**: one codebase for the CPU / CUDA / Vulkan backends.
2. **Accuracy alignment**: ggml outputs on all three backends match the PyTorch
   reference layer by layer.
3. **Performance targets**: both CUDA and Vulkan inference faster than PyTorch CUDA.
4. **Multi-precision GGUF**: every sub-model produced in f32 / f16 / q8.

---

## 1. Conclusions First (First-Principles Reasoning)

- ggml abstracts backends through a unified backend registry. **CUDA/Vulkan are
  compile-time options only; the inference code is the same**: at runtime
  `ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)` picks the first available
  GPU device, falling back to CPU otherwise.
- So "full support" ≠ writing three codebases; it means **one build with
  `-DGGML_CUDA=ON -DGGML_VULKAN=ON` + runtime detection**.
- The real difficulty is not the backends but **3D op coverage** (FlexiCubes /
  NeuralRender / 3D conv). Strategy: portable ops run on GPU; 3D ops missing from the
  ggml backends are pinned to CPU to keep the three backends numerically consistent.
- Accuracy alignment follows the trellis **parity workflow**: Python dumps reference
  activations → C++ taps each layer and compares with `atol + rtol*|ref|`.

---

## 2. Pipeline Breakdown and Op Mapping

InstantMesh inference at its core (information flow):

```
Input image
 ├─ (A) Background removal   rembg/BiRefNet
 ├─ (B) Multi-view diffusion Zero123++ UNet + VAE + Euler scheduler  → 6 views
 └─ (C) Reconstruction LRM
      ├─ DINO ViT-B/16 features
      ├─ TriplaneTransformer → triplane
      ├─ FlexiCubes geometry
      └─ NeuralRender rendering
```

| ID | Component | Main ops | CUDA | Vulkan | Porting difficulty | Notes |
|----|------|---------|:----:|:------:|:--------:|------|
| A | rembg(BiRefNet) | 2D conv / swin | ✅ | ✅ (needs extra ops) | Medium | Reuse trellis `RMBG-2.0-GGML` directly |
| B | Zero123++ UNet | 2D conv / attn | ✅ | ✅ | High | Reference: stable-diffusion.cpp |
| B | VAE decoder | 2D conv/up | ✅ | ✅ | Medium | Same as above |
| B | scheduler | Pure CPU numerics | ✅ | ✅ | Low | Euler, no tensor backend |
| C | DINO ViT-B | 2D transformer | ✅ | ✅ | Low | Reference: trellis DINOv3 |
| C | TriplaneTransformer | 2D attn | ✅ | ✅ | Low | 16 layers, 1024 dims |
| C | FlexiCubes geometry | 3D SDF/deform | ⚠️ | ⚠️ | High | Missing voxel ops, pin to CPU |
| C | NeuralRender | ray-marching | ⚠️ | ⚠️ | High | Custom sampling op or pin to CPU |
| C | UV / mesh export | Pure CPU | — | — | Low | mesh_export |

**Key discipline for backend consistency**: any op whose CUDA and Vulkan behaviors
differ is uniformly pinned to CPU (e.g. 3D conv) instead of letting each side run its
own path → guarantees accuracy consistency across the three backends.

---

## 3. Accuracy Alignment (parity)

Reuse the trellis **depth-anything-style layered verification**:

1. **Reference container**: `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` with official
   pip dependencies only; 3D sparse ops are monkeypatched to pure-torch gather-GEMM
   (runs on CPU and doubles as the porting spec).
2. **Reference activation dump**: Python forward hooks write each layer's activations
   to `.gguf` + a manifest, read directly by C++ via the ggml API.
3. **Layer-by-layer comparison on the C++ side**: `cpp_ggml/tests/parity.hpp` uses an
   `atol + rtol*|ref|` threshold (default 2e-3); a whole layer is skipped via
   `SKIP_RETURN_CODE 77` when fixtures are missing.
4. **Joint three-backend verification**: the same input runs on CPU/CUDA/Vulkan; the
   three are compared against each other and each against PyTorch. f16/q8 vs f32 error
   must stay within quantization tolerance (see §5).

---

## 4. Performance Targets

| Backend | Target | Reference |
|------|------|------|
| CUDA | **Faster than PyTorch CUDA** | trellis measured ggml CUDA 142.7s vs PyTorch 317.9s (2.2×) |
| Vulkan | **Faster than PyTorch CUDA** | trellis measured ggml Vulkan 136.1s (2.3×) |
| CPU | Just needs to run (no hard speed requirement) | Used for accuracy regression |

Key performance decisions:
- Always use `ggml_flash_attn_ext` for attention (bit-identical to full softmax,
  including on CPU).
- 2D conv / transformer / diffusion run on GPU; 3D conv decoding is pinned to CPU as
  in trellis (its GPU version is immature; consistency first).
- Use `TRELLIS2_TIMING`-style per-stage wall-clock timing to locate bottlenecks and
  avoid optimizing the wrong stage.

> Note: beating PyTorch comes mainly from two things — (1) native CUDA/Vulkan kernels
> for diffusion/Transformer; (2) replacing PyTorch's pure-Python sparse loops for 3D
> sparse ops with native kernels (trellis's 3D conv reaches 20×).

---

## 5. Multi-Precision GGUF (f32 / f16 / q8)

### 5.1 Conversion targets
**Every sub-model** (BiRefNet, Zero123++ UNet, VAE, DINO, TriplaneTransformer, plus
FlexiCubes/NeuralRender if they carry weights) produces three GGUF variants:

| Precision | ggml type | Use |
|------|-----------|------|
| f32 | `GGML_TYPE_F32` | Accuracy baseline, double-precision regression |
| f16 | `GGML_TYPE_F16` | Default inference, GPU-friendly |
| q8 | `GGML_TYPE_Q8_0` | Low VRAM/bandwidth, CPU and mobile |

### 5.2 Requirements
- Write `general.architecture` and per-sub-model hparams KV into the GGUF so the C++
  loader can dispatch by architecture.
- Weight tensor names map one-to-one to the PyTorch state_dict (with the
  `lrm_generator.` prefix stripped).
- Conversion scripts are idempotent and re-runnable; quantize f16 → q8_0 in the
  `quantize_to_q8.py` style.

### 5.3 Accuracy tolerances
- f16 vs f32: per-layer `atol+rtol*|ref|` threshold relaxed to ~1e-2 (half-precision
  tolerance).
- q8 vs f32: ~1e-2 to 5e-2. **All three backends must share the same quantized
  weights**; per-backend quantization is forbidden → keeps results comparable across
  backends.

---

## 6. Project Layout

```
InstantMesh-GGML/
├── run.py / app.py / train.py / configs/ / zero123plus/ / src/  # official Python InstantMesh (kept as-is; src/ keeps only .py)
└── cpp_ggml/                 # ★ the C++ / GGML world (all C++-related code lives here)
    ├── CMakeLists.txt        # Build: vendored ggml + dual backends (CUDA/Vulkan compile-time options)
    ├── src/
    │   ├── core/             # backend.hpp/.cpp runtime detection (CPU/CUDA/Vulkan), gguf_io.hpp/.cpp GGUF loading
    │   ├── models/           # dino / lrm_transformer / synthesizer / flexicubes / texture_map(+xatlas)
    │   ├── tools/            # C++ CLI tools (instantmesh.cpp: multi-view input -> OBJ + vertex colors/texture)
    │   └── utils/stb_image_write.h
    ├── convert/              # GGUF converters (Python, conversion only; convert_*.py)
    ├── tests/                # parity regression (test_*.cpp, ctest)
    ├── benchmarks/           # parity checks + performance benchmarks + results/ artifacts
    ├── models/gguf/          # f32/f16/q8 GGUF weights (not committed)
    └── third_party/ggml/     # ggml v0.21.0 submodule (with the flash_attn head_dim=64 fix)
```

---

## 7. Implementation Order (each step deliverable and verifiable)

1. **Infrastructure**: `cpp_ggml/CMakeLists.txt` (`-DGGML_CUDA=ON -DGGML_VULKAN=ON`),
   `cpp_ggml/src/core/backend` detection, reference container and weights, parity
   framework. → **delivered in this phase**
2. **DINO port + parity**: the DINO/Transformer part of image→triplane runs end-to-end
   in C++.
3. **Zero123++ diffusion + VAE + scheduler port + parity**: single image → multi-view.
4. **rembg(BiRefNet) integration**: reuse the trellis submodule.
5. **Geometry/rendering**: FlexiCubes + NeuralRender; pin 3D ops to CPU first for
   consistency.
6. **Performance tuning**: per-stage timing, flash-attn, GPU-izing 2D ops, adding 3D
   ops when necessary.
7. **Full GGUF three-precision set** + joint three-backend regression + benchmark
   comparison.

---

## 8. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|------|------|
| Missing 3D ops (voxel/ray-march) in backends | CUDA/Vulkan can't be fully GPU | Pin to CPU first for consistency; add custom ops later |
| High effort for custom ops (Vulkan shaders) | Incomplete Vulkan coverage | Reuse trellis `ggml-rmbg-ops.patch`; write each op for both backends with parity |
| Quantization introduces cross-backend error | Accuracy claims break | Share the same quantized weights across backends + per-precision thresholds |
| ggml upstream changes | Regression on both backends | **Pin the submodule version** + maintain the `patches/` directory |
| Reference generation depends on GPU/VRAM | Parity can't run | Staged dumps + monkeypatched sparse ops (CPU-capable) |

---

## 9. Acceptance Criteria

- [ ] PyTorch output from `python run.py` and C++ `instantmesh` output (same seed/input)
      agree on mesh geometry (tolerance allowed).
- [ ] Both CUDA and Vulkan inference latency are below the PyTorch CUDA baseline.
- [ ] Every sub-model has f32/f16/q8 GGUF files with a correct `general.architecture`.
- [ ] Layer-by-layer parity passes on CPU/CUDA/Vulkan at f32; f16/q8 pass within their
      tolerances.
- [ ] Zero PyTorch runtime dependency (`ldd` shows no torch, no `import torch`).

---

## 10. Current Implementation Status

Completed (in this repo):

- ✅ `docs/PLAN.md`: pure C++ integration, accuracy alignment, performance targets,
  f32/f16/q8 multi-precision plan.
- ✅ **C++ project skeleton**: [CMakeLists.txt](../cpp_ggml/CMakeLists.txt) vendors
  `third_party/ggml`; backend auto-detection in [src/core/backend.hpp](../cpp_ggml/src/core/backend.hpp);
  GGUF loading in [src/core/gguf_io.hpp](../cpp_ggml/src/core/gguf_io.hpp).
- ✅ **GGUF conversion framework** (`cpp_ggml/convert/`): `convert_common.py`
  (f32/f16/q8 + per-tensor type policy), `convert_lrm.py` (splits the LRM into
  dino / lrm_transformer / geometry / synthesizer), `convert_all.py` (batched
  three-precision). Fully verified for f32/f16/q8 with a synthetic ckpt, and the C++
  loader reads back `general.architecture` and tensors correctly.
- ✅ **Backend verification**: CPU build passes; **CUDA build passes**, detecting the
  RTX 3060 at runtime and loading GGUF weights on the GPU.
- ✅ **Vulkan backend**: see the note below; the dual-backend (CUDA+Vulkan) build passes
  and both GPU backends initialize the RTX 3060 at runtime.
- ✅ **DINO encoder port**: [src/models/dino.hpp](../cpp_ggml/src/models/dino.hpp)/[dino.cpp](../cpp_ggml/src/models/dino.cpp)
  — full adaLN DiT graph: patch conv → cls → pos-enc → 12 layers (adaLN modulation +
  flash attention + qkv-bias + GELU MLP), running on any of CPU/CUDA/Vulkan; CLI at
  [src/tools/dino.cpp](../cpp_ggml/src/tools/dino.cpp).
- ✅ **ggml upgraded to v0.18.1 with flash_attn enabled**: submodule bumped from v0.18.0
  to v0.18.1 (`git submodule update cpp_ggml/third_party/ggml`). Replaced the **manual
  softmax attention** (`mul_mat`+`soft_max_ext`+`mul_mat`) in DINO/TriplaneTransformer
  with `ggml_flash_attn_ext`. v0.18.1 fixed the v0.18.0 flash_attn bug at head_dim=64 —
  verified with an isolated test on three real configurations (DINO self-attn
  hd=64/nh=12/seq=197, LRM self-attn hd=64/nh=16/seq=3072 tiled path, LRM cross-attn
  q=3072/kv=197) that flash_attn matches manual softmax element-wise (corr=1.0,
  maxdiff≈1e-7). After the swap, three-precision CPU parity is unchanged (DINO
  rel≈1.2e-3, LRM f32/f16/q8 = 1.2e-5 / 1.7e-3 / 3.4e-2), and the flash path is
  significantly faster on GPUs.
- ✅ **TriplaneTransformer port**: [src/models/lrm_transformer.hpp](../cpp_ggml/src/models/lrm_transformer.hpp)/[lrm_transformer.cpp](../cpp_ggml/src/models/lrm_transformer.cpp)
  — 16 layers of 1024-dim BasicTransformerBlock (LayerNorm + cross-attention +
  self-attention + GELU MLP, all with residuals), adaLN modulation, final LayerNorm,
  transposed-conv upsampling into the triplane; CLI at
  [src/tools/lrm_transformer.cpp](../cpp_ggml/src/tools/lrm_transformer.cpp).
  Three-precision CPU parity all pass (rel≈1.1e-5 / 1.7e-3 / 3.4e-2). Key pitfalls:
  - Attention uses `ggml_flash_attn_ext` (from v0.18.1 on; before that v0.18.0 had the
    head_dim=64 bug, forcing a manual softmax fallback).
  - MLP uses `ggml_gelu_erf` (PyTorch `nn.GELU()` defaults to the exact erf); otherwise
    the tanh approximation accumulates error layer by layer.
  - `ggml_conv_transpose_2d_p0` does **not support batch>1** (it only computes the
    batch-0 slice): run per-plane with batch=1 and `ggml_concat` along the batch dim;
    it also requires F32 input, so activations of quantized models must be
    `ggml_cpy`-converted to F32 first.
  - Activations always run in F32 (quantization applies to weights only): the initial
    `pos_embed` must be converted to F32 if F16, otherwise later
    `ggml_norm`/`conv_transpose` assertions fail.
- ✅ **Geometry prediction / OSGDecoder port**: [src/models/synthesizer.hpp](../cpp_ggml/src/models/synthesizer.hpp)/[synthesizer.cpp](../cpp_ggml/src/models/synthesizer.cpp)
  — the neural part of FlexiCubes (with weights): host-side triplane bilinear sampling
  (`grid_sample` align_corners=False, 3 planes projected as (x,y)/(x,z)/(z,y)) +
  gathering 8 cube-corner features, then a single ggml graph runs three 4-layer ReLU
  MLPs for sdf / deformation / weight; CLI at
  [src/tools/synthesizer.cpp](../cpp_ggml/src/tools/synthesizer.cpp), reference in
  [convert/parity_synth.py](../cpp_ggml/convert/parity_synth.py). At 64×64 planes +
  grid_res=64, three-precision CPU parity: f32 rel≈3e-7, f16≈2-7e-4, q8 acceptable
  (sdf sign-flip rate 0.003%). Key points:
  - Sampling is a fixed geometric operation (no weights), implemented in host C++;
    `px=(nx+1)/2*W-0.5`, 4-neighbor weighting, zero padding out of bounds.
  - GGUF tensor names carry the `decoder.` prefix (after stripping `synthesizer.`);
    `get_t` needs it appended.
  - The OSGDecoder weight network takes the 8 gathered cube-corner features as input
    (`decoder.net_weight.0.weight` input 8*3C=1920) and multiplies the output by 0.1.
- ✅ **FlexiCubes mesh extraction port**: [src/models/flexicubes.hpp](../cpp_ggml/src/models/flexicubes.hpp)/[flexicubes.cpp](../cpp_ggml/src/models/flexicubes.cpp),
  table header [flexicubes_tables.hpp](../cpp_ggml/src/models/flexicubes_tables.hpp)
  (generated by [convert/gen_flexicubes_tables.py](../cpp_ggml/convert/gen_flexicubes_tables.py)),
  CLI at [src/tools/flexicubes.cpp](../cpp_ggml/src/tools/flexicubes.cpp), reference in
  [convert/parity_flexicubes.py](../cpp_ggml/convert/parity_flexicubes.py). Pure C++
  fixed-lookup-table algorithm (no weights): surface cube selection → dual vertices
  (with `_linear_interp` edge interpolation) → triangulation. Parity: vertex
  max_abs≈6e-8 (float32 limit), faces exactly identical. Key pitfalls:
  - `DMC_TABLE` is a `[256][4][7]` 3D array, indexed as `DMC_TABLE[case][v][e]`.
  - Dual vertex `ue = (x0*w1 - x1*w0)/(w1 - w0)` (w0↔v0, w1↔v1); misbinding the
    weights causes ~9% vertex error.
- ✅ **End-to-end pipeline integration**: [src/tools/instantmesh.cpp](../cpp_ggml/src/tools/instantmesh.cpp)
  chains DINO → TriplaneTransformer → OSGDecoder → FlexiCubes into a single command
  `image → mesh.obj`; input preprocessing comes from
  [convert/prep_input.py](../cpp_ggml/convert/prep_input.py) (multi-view images
  `[V,3,224,224]` + cameras `[V,16]`). Host C++ has built-in `construct_voxel_grid`
  (voxel grid + deduplicated cube indices), `center_boundary_index`, default
  zero123plus cameras (radius=4.0/fov=30), deformation tanh normalization and the
  empty-shape sdf fix. Fixed multi-view batch related assertions:
  - CLS token must be `ggml_repeat`-broadcast to the batch dim before `ggml_concat`.
  - `ggml_flash_attn_ext` output is non-contiguous for batch>1; needs `ggml_cont`
    before reshape.
  - The four adaLN `ggml_view_2d` segments are padded non-contiguous views; `modulate`
    needs `ggml_cont` before reshaping to `[hidden,1,B]` (batch=1 slipped through only
    because `ggml_is_contiguous` skips the check for `ne==1` dims).
  - Runs on real photos: `sdf` distribution is sensible (mostly positive, locally
    negative forming the surface) and produces a mesh with a converging bbox; mesh
    quality is limited by the input (replicated single view can't replace Zero123++
    multi-view generation).
- ✅ **Vertex colors (net_rgb)**: [src/models/synthesizer.cpp](../cpp_ggml/src/models/synthesizer.cpp)
  adds `synthesizer_texture_forward`, using the synthesizer's `net_rgb` branch to
  sample the triplane per mesh vertex and compute RGB (`sigmoid` + MipNeRF clamp),
  same triplane as geometry. Parity (rgb comparison in
  [convert/parity_synth.py](../cpp_ggml/convert/parity_synth.py)): `max_abs≈3.3e-7`.
  OBJ output looks like `v x y z r g b` (RGB ∈ [0,1]).
- ✅ **End-to-end color/geometry validation (cpp_ggml/benchmarks/)**: added
  [benchmarks/pytorch_reference.py](../cpp_ggml/benchmarks/pytorch_reference.py)
  (vertex-color reference sampled on the official `LRMGenerator`),
  [benchmarks/run_bench.sh](../cpp_ggml/benchmarks/run_bench.sh) (CPU/CUDA/Vulkan ×
  f32/f16/q8 end-to-end benchmark), [benchmarks/analyze.py](../cpp_ggml/benchmarks/analyze.py)
  (report + charts), [benchmarks/render_meshes.py](../cpp_ggml/benchmarks/render_meshes.py)
  (comparison rendering), [benchmarks/color_parity_e2e.py](../cpp_ggml/benchmarks/color_parity_e2e.py)
  (compares both sides' net_rgb on identical mesh vertices). Conclusion: **ggml vs
  PyTorch vertex colors Δmean≈0.002, vertex counts essentially identical** (e.g.
  blue_cat 14550 vs 14558).
- ✅ **Input normalization bug fix**: `benchmarks/mv/<img>/image.bin` is already an
  ImageNet-normalized tensor (now under `cpp_ggml/benchmarks/mv/`); the PyTorch
  reference previously normalized it again via `ViTImageProcessor` (double
  normalization), which pushed the reference triplane/mesh/colors far from ggml
  (Δmean≈0.25). After the fix the reference feeds DINO the same normalized tensor
  directly, dropping Δmean to ~0.002. **The ggml side was correct all along.**
- ✅ **Mesh resolution alignment**: since an RTX 3060 (12GB) cannot fit grid_res=128
  (~18GB needed), ggml and the PyTorch reference both use **grid_res=88** for
  like-for-like comparison; `--grid-res` is controlled by
  [benchmarks/run_bench.sh](../cpp_ggml/benchmarks/run_bench.sh).
- ✅ **Texture map export (baking)**: [src/models/texture_map.cpp](../cpp_ggml/src/models/texture_map.cpp)
  (xatlas UV unwrapping + multi-view texture baking) is implemented; the CLI
  `--export-texmap` matches the official `--export_texmap`;
  [test_texture_map](../cpp_ggml/tests/test_texture_map.cpp) parity passes; GLB
  packaging is not implemented (outputs OBJ+MTL+PNG).
- ✅ **Official feature alignment audit (PBR/baking)**: official InstantMesh has
  **no PBR** (no metallic/roughness/normal-map path anywhere in the Python code), so
  nothing to align; the official "baking" is exactly the `--export_texmap` texture
  baking, which C++ has aligned (see above); official capabilities not yet covered by
  C++ (rembg foreground segmentation, Zero123++ multi-view diffusion, NeuralRender,
  `--save_video`, NeRF variants) are tracked in the TODO below.
- ✅ **Directory restructuring (cpp_ggml/)**: all C++/GGML-related parts
  (CMakeLists.txt, src C++ subtree, tests, convert, benchmarks, models/gguf,
  third_party/ggml submodule) moved under `cpp_ggml/`, keeping the official Python
  InstantMesh at the repo root. C++ relative include paths unchanged; Python script
  path roots split into `ROOT` (→ cpp_ggml/) and `REPO_ROOT` (→ repo root for official
  src/, configs/, examples/).
- ✅ **ggml patch mechanism in place (§8 mitigation)**:
  [cpp_ggml/patches/](../cpp_ggml/patches/README.md) holds all adaptations to the ggml
  submodule, auto-applied by CMake at configure time (idempotent skip when already
  applied, error on conflict), making `clone → submodule update → build` reproducible.
  Currently includes `ggml-rmbg-ops.patch` (RMBG custom ops, v0.21.0 baseline); a
  submodule existence check was also added to the build. Beginner's guide in
  [cpp_ggml/README.md](../cpp_ggml/README.md); model metrics/download in
  [cpp_ggml/models/MODEL_CARD.md](../cpp_ggml/models/MODEL_CARD.md) (weights hosted on
  Hugging Face `Asher-1/InstantMeshGGuf`).
- ✅ **ggml upgraded to v0.21.0**: submodule bumped from v0.18.1 to v0.21.0
  (`8599e0ea`). API audit: v0.18.1→v0.21.0 changes in `ggml.h`/`ggml-backend.h` are
  additions only (`ggml_rope_set_offset`, `ggml_build_forward_order`, `mmap_support`
  field) — no removals/signature changes, zero adaptation needed for this repo's ~40
  ggml call sites. The RMBG patch was replayed with `git apply --3way`; only
  `ggml-vulkan.cpp` submit logic conflicted (upstream already includes the 0.20.1
  submission fix, upstream version kept), and the patch was re-exported with v0.21.0
  as baseline. Verified: CPU/CUDA builds + ctest pass (only the in-development
  test_scheduler fails, pure-CPU numeric code unrelated to ggml); end-to-end
  cute_horse CUDA f32 is **bit-identical** to v0.18.1 (RMSE=0, vertex count
  19328=19328), f16 relative RMSE 1.33% (sign flips 0.07%, same order as quantization
  noise, vertex count diff 0.13%), geometry preserved.
- ✅ **f16 drift re-verification (4 demos, repeated)**: cross-version f16 relative
  RMSE over blue_cat/cute_horse/fox/robot = 0.59% / 1.33% / 0.47% / 0.58% (mean
  **0.74%**, all ≤ the f16 quantization-noise baseline of 0.81%); sign flips
  0.037–0.071% all below the v0.18.1 f16 self-baseline 0.094%; within-v0.21.0
  f16-vs-f32 deviation = 0.83% ≈ historical baseline 0.81% (f16 path health
  unchanged — actually slightly closer to f32 than v0.18.1's 1.21%); the
  cross-version drift matches two independent roundings of the same math
  (expected orthogonal composition √(0.83²+1.21²)≈1.47% ≥ observed 1.33%). Root
  cause confirmed upstream: 14 CUDA kernel commits between v0.18.1..v0.21.0
  (mvq→MMQ crossover retuning, MMVQ nwarps, static cuBLAS workspaces, cpy kernel
  launch fix) change fp16 accumulation order — a re-rounding, not a semantic
  change (f32 bit-identical proves this). Vertex-level Chamfer displacement vs
  v0.18.1: mean 0.042–0.061% of bbox diagonal, vertex count diff <0.2%. Side-by-
  side renders: `build-cuda/v021_parity_render.png` (top v0.18.1 / bottom
  v0.21.0, visually indistinguishable). Conclusion: **not a defect** — expected
  fp16 re-rounding from upstream kernel retuning; the f16 recommendation in the
  model card is unaffected.

### Status / TODO

- ✅ **Zero123++ multi-view diffusion + VAE + scheduler** ported and parity-verified:
  [src/tools/zero123pp.cpp](../cpp_ggml/src/tools/zero123pp.cpp) chains CLIP vision →
  scheduler → UNet → VAE decode; per-component parity via ctest fixtures
  (`test_clip_vision`, `test_scheduler`, `test_unet`, `test_vae`) all pass
  (UNet f32/f16 max_abs≈7e-4/8.8e-4, VAE encode 2.4e-4 / decode 3e-3).
- ✅ **rembg(BiRefNet) foreground segmentation** integrated via ggml custom ops
  (see `ggml-rmbg-ops.patch`) with the `rembg` tool; not yet chain-integrated into
  the single-command instantmesh pipeline.
- ❌ **NeuralRender rendering** (ray-marching) not ported yet.
- ❌ **GLB packaging export** not implemented (texture baking is implemented, outputs
  OBJ+MTL+PNG).
- ⏳ **Vulkan full parity**: can initialize and run on GPU, but full layer-by-layer
  parity regression for the three precisions is pending (CUDA is aligned).
- ⏳ Larger mesh resolution (grid_res=128) unverified due to VRAM limits.

Build commands:

```bash
# CPU
cmake -B cpp_ggml/build -S cpp_ggml -DINSTANTMESH_BUILD_TESTS=ON && cmake --build cpp_ggml/build -j
# CUDA (recommended GPU path)
cmake -B cpp_ggml/build-cuda -S cpp_ggml -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build cpp_ggml/build-cuda -j
# Dual backend (CUDA + Vulkan)
cmake -B cpp_ggml/build-gpu -S cpp_ggml -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build cpp_ggml/build-gpu -j
```

### Vulkan Build Notes (resolved)

Local toolchain: the Vulkan SDK lives in `~/VulkanSDK`; the working glslc is
`/usr/local/bin/glslc` (the glslc bundled with the VulkanSDK is a 2026.x build needing
a newer glibc and won't run). glslc must be specified explicitly at configure time:

```bash
export VULKAN_SDK=$HOME/VulkanSDK/1.4.350.0/x86_64
cmake -B cpp_ggml/build-gpu -S cpp_ggml -DGGML_CUDA=ON -DGGML_VULKAN=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DVulkan_GLSLC_EXECUTABLE=/usr/local/bin/glslc
cmake --build cpp_ggml/build-gpu -j
```

**Verified locally**: the dual-backend build passes; both GPU backends initialize the
RTX 3060 correctly at runtime — both the CUDA and Vulkan paths work. The three-backend
(CPU/CUDA/Vulkan) auto-detection logic passes in `test_backend`.
