# patches/ — ggml Submodule Adaptation Patches

This directory holds **all local adaptation changes** to `third_party/ggml` (the ggml
submodule), one `.patch` file per patch (generated via `git format-patch` / `git diff`;
lexicographic filename order is the apply order, naming convention `NNNN-description.patch`).

## Why This Directory Exists

ggml is a submodule pinned to a fixed tag of the official
[ggml-org/ggml](https://github.com/ggml-org/ggml). Any uncommitted modification inside
the submodule is **not** distributed with this repo — other developers always get clean
official sources from `git submodule update --init`. Therefore:

> **Any modification to the ggml sources must be captured as a `.patch` file in this directory.**
> CMake automatically applies them in order at configure time (idempotent skip if already
> applied); no manual step is needed after clone.

## Current Status

| Patch | Content | Baseline |
|---|---|---|
| `ggml-rmbg-ops.patch` | ggml custom ops required by the rembg/BiRefNet foreground segmentation (RMBG-2.0) port: `GGML_MAX_NAME` 64→128, CUDA kernels (deform-im2col / affine-relu / gather / swin-qkv-layout / swin-windows) with matching Vulkan shaders, plus shader-generator registration. Also carries three explicit ggml-vulkan precision APIs (see below) and an `IM_GN_DEBUG`-gated debug print in `src/ggml-cpu/ops.cpp` group-norm (used during UNet parity debugging). | Official v0.21.0 (`8599e0ea`) |

> History:
> - The flash-attn head_dim=64 issue (v0.18.0) was fixed upstream in v0.18.1; resolved
>   via a submodule version bump, no patch needed.
> - During the v0.18.1 → v0.21.0 upgrade, this patch was replayed with `git apply --3way`;
>   the only conflict was in the `ggml-vulkan.cpp` submit logic (v0.21.0 refactored
>   `submit_after` into a parameterized lambda and added serialize diagnostics; the
>   patch-side "0.20.1 submission fix" already exists upstream). The upstream version
>   was kept at the conflict site, and all RMBG op code merged cleanly. The patch file
>   has been re-exported with v0.21.0 as the baseline.

The current `third_party/ggml` worktree = v0.21.0 + all patches above. CMake verifies
automatically at configure time: missing patch → auto apply; already applied → idempotent
skip; conflict → error and abort.

## Known ggml-vulkan Precision Pitfalls (upstream behavior — model graphs must compensate)

Two upstream ggml-vulkan behaviors silently quantize F32 data during backend parity
debugging (full analysis in `docs/ALIGNMENT.md`, "Vulkan f16 traps"). Neither is a
patch bug, but the patch's dispatch hooks interact with both, so record them here:

1. **F32 matmul shaders stage through F16.** Every `matmul_f32_*` SPIR-V variant
   (the coopmat `_cm1` shader AND the fp16-compiled scalar fallback in
   `ggml_vk_load_shaders`'s `else if (device->fp16)` branch) is generated with
   `FLOAT_TYPE=float16_t` — baked in at build time by `vulkan-shaders-gen`, not
   changeable at runtime. Only the `_fp32` variants (used by the
   `!coopmat && !fp16` branch) are exact. Escape hatches:
   - Explicit API `ggml_backend_vk_set_f32_matmul_exact(bool)` (added by this
     patch, default `true`): F32×F32→F32 matmuls are routed to the exact FP32
     scalar pipelines while coopmat stays available for f16/quantized matmuls
     (which always keep tensor cores). Set `false` to trade exactness for
     tensor-core speed on large f32 matmuls. The rembg `model_loader.cpp`
     calls this instead of env vars.
   - Diagnostic env `GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_F16=1` (both
     required — disabling only coopmat lands in the still-fp16 scalar branch).
   Measured: isolated mul_mat error 4.6e-3 → 1.8e-5; CLIP patch conv 2.1e-4 → 1.0e-6.
2. **Non-contiguous matmul operands are copied to F16.** `ggml_vk_mul_mat_q_f16`
   converts x/y operands that fail `ggml_vk_dim01_contiguous` to F16 before
   dispatch. Per-head strided `ggml_view_3d` or un-materialized
   `reshape+permute` used directly as a matmul operand hits this (symptom: a
   single attention output at ~1e-4 instead of ~1e-6 vs torch fp32).
   Mitigation: materialize every matmul operand with `ggml_cont()`
   (clip_vision `qr`, unet `qh/kh`, dino `to_head`).

When changing anything in these dispatch paths, re-verify with
`tests/test_clip_vision vulkan` under the escape-hatch env: expect embeds parity
~4e-6 with all staged dumps at torch-fp32 rounding level.

Additional patch-provided explicit APIs (all weak-linked by callers, CPU-only /
CUDA-only builds stay linkable):

- `ggml_backend_vk_set_fp16(bool)` — create Vulkan devices with fp16
  storage/compute disabled (all shaders fp32, f16 weights dequantized per
  matmul). Long sampling runs need `false`: f16-weight matmuls stage f32
  activations through f16 and the error compounds step over step. Latched at
  device creation (`ggml_vk_get_device`, reached via `ggml_backend_dev_init`) —
  must be called before the first device init. `src/core/backend.cpp` applies
  `BackendInitOptions::vulkan_fp16` (default `false`) there for every
  InstantMesh tool. The `GGML_VK_DISABLE_F16` env remains supported and ANDs
  with the flag.
  Threading/multi-device contract (matches upstream env-read semantics): the
  flag is a plain global read once per device at creation; it never mutates
  already-created devices, so repeated sequential `init_best_backend` calls
  cannot cross-contaminate. Concurrent backend initialization from multiple
  threads is NOT supported (upstream `ggml_backend_dev_init` is not
  thread-safe either — serialize it, as `getenv`-based config also requires).
  One global governs every Vulkan device created after the call; per-device
  control is only possible with sequential setter + `dev_init` pairs.

## How to Generate a New Patch

```bash
# After making and verifying changes inside the submodule, export the worktree diff.
# IMPORTANT: `git add -N` every NEW (untracked) file first — plain `git diff`
# silently omits untracked files, which would drop them from the patch.
cd cpp_ggml/third_party/ggml
git add -N $(git ls-files --others --exclude-standard)
git diff > ../../patches/0001-<short-desc>.patch

# Verify idempotent application (on a clean submodule):
git -C cpp_ggml/third_party/ggml checkout -- .
rm -rf cpp_ggml/build && cmake -B cpp_ggml/build -S cpp_ggml   # should print "Applied ggml patch"
```

When committing, include both the patch file(s) and any gitlink change of
`third_party/ggml` (if applicable).
