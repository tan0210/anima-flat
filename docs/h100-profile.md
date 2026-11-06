# H100 NVL Profile

Inference baseline on an NVIDIA H100 NVL, captured after the fused forward pass was restructured into a per-block helper and the backward kernels were added. Raw JSON output and Chrome traces are available in `profile_h100/`.

```bash
python -m src.scripts.profile --fp8 --sizes 512 1024 --profile-size 1024 --out profile_h100
```

Absolute timings are specific to this GPU and host. For cross-GPU comparisons, use the reported ratios rather than absolute milliseconds.

## Forward Pass

One denoiser evaluation at batch size 2.

| size | implementation      |     ms | speedup vs Diffusers | TFLOP/s |
| ---- | ------------------- | -----: | -------------------: | ------: |
| 512  | Diffusers           |  40.93 |                1.00x |     188 |
| 512  | Flat (BF16)         |  25.65 |                1.60x |     300 |
| 512  | Flat (FP8)          |  30.40 |                1.35x |     253 |
| 512  | Flat reference path |  29.60 |                1.38x |     260 |
| 1024 | Diffusers           | 164.30 |                1.00x |     218 |
| 1024 | Flat (BF16)         |  83.75 |                1.96x |     428 |
| 1024 | Flat (FP8)          |  66.14 |                2.48x |     542 |
| 1024 | Flat reference path | 129.29 |                1.27x |     277 |

At 1024×1024, FP8 is **1.27x faster than BF16**, while the fused path is **1.54x faster than the reference path**.

## End-to-End

20 sampling steps, including text encoding and VAE decoding.

| size | implementation |       ms | speedup vs Diffusers |
| ---- | -------------- | -------: | -------------------: |
| 512  | Diffusers      | 1,746.36 |                1.00x |
| 512  | Flat (BF16)    |   586.20 |                2.98x |
| 512  | Flat (FP8)     |   666.70 |                2.62x |
| 1024 | Diffusers      | 3,439.98 |                1.00x |
| 1024 | Flat (BF16)    | 1,863.50 |                1.85x |
| 1024 | Flat (FP8)     | 1,520.84 |                2.26x |

### BF16 Stage Breakdown

| size |   text encode |           denoise |    VAE decode |
| ---- | ------------: | ----------------: | ------------: |
| 512  | 51.85 ms (9%) |   507.05 ms (86%) | 22.82 ms (4%) |
| 1024 | 63.09 ms (3%) | 1,762.75 ms (95%) | 90.49 ms (5%) |

Denoising dominates end-to-end runtime, accounting for 86% of the 512×512 run and 95% of the 1024×1024 run.

## Kernel Breakdown at 1024×1024

| implementation | ms per call | GPU kernel ms | busy | launches |
| -------------- | ----------: | ------------: | ---: | -------: |
| Diffusers      |      164.30 |        161.73 |  98% |    3,009 |
| Flat (BF16)    |       83.75 |         80.49 |  96% |      535 |
| Flat (FP8)     |       66.14 |         63.30 |  96% |      647 |

### Kernel Categories

| category             | Diffusers | Flat (BF16) | Flat (FP8) |
| -------------------- | --------: | ----------: | ---------: |
| attention            |  18.90 ms |    20.13 ms |   19.91 ms |
| GEMM                 |  51.51 ms |    54.50 ms |   32.25 ms |
| our Triton kernels   |       n/a |     5.67 ms |   10.92 ms |
| norm and elementwise |  64.86 ms |     0.04 ms |    0.04 ms |
| copy and layout      |  26.46 ms |     0.17 ms |    0.19 ms |

The main difference between the Flat implementations is GEMM time: FP8 reduces GEMM time from 54.50 ms to 32.25 ms, more than offsetting the additional Triton-kernel cost.

### Top Kernels — Flat (FP8)

| kernel                                                      | calls |    ms | avg µs | share |
| ----------------------------------------------------------- | ----: | ----: | -----: | ----: |
| `nvjet_sm90_qqtst_128x128_128x6_..._ovscale_TNT` (FP8 GEMM) |   168 | 31.82 |  189.4 | 50.3% |
| FlashAttention-4 forward                                    |    56 | 19.91 |  355.6 | 31.5% |
| `_rowquant_kernel`                                          |    84 |  4.96 |   59.0 |  7.8% |
| `_residual_adaln_fp8_kernel`                                |    84 |  3.88 |   46.2 |  6.1% |
| `_qkv_norm_rope_kernel`                                     |    28 |  1.45 |   51.8 |  2.3% |
| `_head_norm_kernel`                                         |    28 |  0.59 |   21.1 |  0.9% |

### Top Kernels — Flat (BF16)

| kernel                                       | calls |    ms | avg µs | share |
| -------------------------------------------- | ----: | ----: | -----: | ----: |
| `nvjet_sm90_tst_256x128_64x4_..._TNT` (GEMM) |   113 | 25.76 |  227.9 | 32.0% |
| FlashAttention-4 forward                     |    56 | 20.13 |  359.5 | 25.0% |
| QuACK `linear_act` (up projection with GELU) |    28 | 16.77 |  598.9 | 20.8% |
| `nvjet_sm90_tst_192x192_64x4_..._TNN` (GEMM) |    28 | 11.56 |  412.7 | 14.4% |
| `_residual_adaln_kernel`                     |    85 |  3.60 |   42.4 |  4.5% |
| `_qkv_norm_rope_kernel`                      |    28 |  1.45 |   51.8 |  1.8% |
| `_head_norm_kernel`                          |    28 |  0.61 |   21.9 |  0.8% |

## Notes

- **FP8 is slower than BF16 at 512×512 in this run.** Forward time is 30.40 ms vs. 25.65 ms, and end-to-end time is 666.70 ms vs. 586.20 ms. FP8 is faster at 1024×1024. The 512×512 result has not been diagnosed. Because the profile does not record GPU busy time at 512×512, host launch overhead cannot be separated from kernel time. Rerun on a quiet host before drawing conclusions.

- **The cross-GPU ratios are consistent with the H200 run.** At 1024×1024, FP8 is 1.27x faster than BF16 on H100 NVL vs. 1.26x on H200. Flat BF16 is 1.96x faster than Diffusers on H100 NVL vs. 2.08x on H200.

- **There is no pre-restructure H100 baseline.** This run therefore cannot establish whether restructuring the fused forward pass into a per-block helper changed forward performance. Future H100 runs should be compared against this baseline.

- **GEMM and attention dominate kernel time.** Together they account for 82% of FP8 kernel time and 93% of BF16 kernel time. The custom Triton kernels account for 17% of FP8 forward time.
