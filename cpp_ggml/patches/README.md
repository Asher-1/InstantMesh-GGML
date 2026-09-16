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
| `ggml-rmbg-ops.patch` | ggml custom ops required by the rembg/BiRefNet foreground segmentation (RMBG-2.0) port: `GGML_MAX_NAME` 64→128, CUDA kernels (deform-im2col / affine-relu / gather / swin-qkv-layout / swin-windows) with matching Vulkan shaders, plus shader-generator registration | Official v0.21.0 (`8599e0ea`) |

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

## How to Generate a New Patch

```bash
# After making and verifying changes inside the submodule, export the worktree diff:
cd cpp_ggml/third_party/ggml
git diff > ../../patches/0001-<short-desc>.patch

# Verify idempotent application (on a clean submodule):
git -C cpp_ggml/third_party/ggml checkout -- .
rm -rf cpp_ggml/build && cmake -B cpp_ggml/build -S cpp_ggml   # should print "Applied ggml patch"
```

When committing, include both the patch file(s) and any gitlink change of
`third_party/ggml` (if applicable).
