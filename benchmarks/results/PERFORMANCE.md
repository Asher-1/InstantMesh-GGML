### ggml (CUDA, per-stage inference) vs PyTorch (CUDA) — wall seconds

| image | ggml f32 | ggml f16 | ggml q8 | PyTorch | ggml f16 / PyTorch |
|-------|----------|----------|---------|---------|--------------------|
| blue_cat | 2.50 | 2.51 | 2.46 | 1.37 | 1.8x |
| cute_horse | 2.48 | 2.46 | 2.52 | 1.34 | 1.8x |
| fox | 2.51 | 2.44 | 2.52 | 1.34 | 1.8x |
| robot | 2.51 | 2.50 | 2.46 | 1.35 | 1.8x |

Mean ggml CUDA f16 total inference: 2.48s vs PyTorch mean 1.35s.

### ggml full-pipeline wall time (incl. model load + mesh export) — seconds

| image | backend | f32 | f16 | q8 |
|-------|---------|-----|-----|----|
| blue_cat | cpu | 25.31 | 35.41 | 22.68 |
| blue_cat | cuda | 3.74 | 3.14 | 3.04 |
| blue_cat | vulkan | 4.54 | 3.95 | 4.11 |
| blue_cat | PyTorch | -- | -- | -- | (CUDA 1.37s)
| cute_horse | cpu | 18.16 | 31.35 | 25.43 |
| cute_horse | cuda | 4.43 | 3.28 | 2.98 |
| cute_horse | vulkan | 4.34 | 3.82 | 3.57 |
| cute_horse | PyTorch | -- | -- | -- | (CUDA 1.34s)
| fox | cpu | 19.41 | 31.09 | 23.06 |
| fox | cuda | 3.88 | 3.33 | 3.05 |
| fox | vulkan | 4.17 | 3.85 | 3.73 |
| fox | PyTorch | -- | -- | -- | (CUDA 1.34s)
| robot | cpu | 23.75 | 34.27 | 28.04 |
| robot | cuda | 3.68 | 3.20 | 3.09 |
| robot | vulkan | 4.33 | 4.07 | 3.68 |
| robot | PyTorch | -- | -- | -- | (CUDA 1.35s)
