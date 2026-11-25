# Fork Optimizations Summary (`opt/2gpu-fast`)

This document summarizes the performance, multi-GPU placement, and VRAM optimizations added in this fork on top of `849bf6392` (`unslothai/llama.cpp#433`).

Target testbed: Windows (WDDM), `CUDA0` RTX 4070 Ti SUPER (16 GiB, 672 GB/s, display GPU) + `CUDA1` RTX 4060 Ti (16 GiB, 288 GB/s, PCIe 3.0 x8), running `Inkling-2-Mini-UD-IQ4_XS.gguf` (24 physical -> 48 logical layers with `[6, 18)` looped 3x, 128-expert top-8 MoE) + `mmproj-F16.gguf` on `CUDA1` with `q8_0` KV cache up to 256k context.

## Summary of Changes

### 1. Loop-Aware 2-GPU Split & DFlash Reserve Cap (`9ad0be05f`)
- Files: `src/llama-model.cpp`, `src/llama-model.h`, `src/llama-context.cpp`, `src/models/models.h`
- Added `loop_core_range()` so automatic layer splitting (`-sm layer`, default auto `-ts`) keeps the entire 3x looped core `[6, 18)` intact on the faster primary GPU (`CUDA0`), offloads prelude `[0, 6)` to `CUDA1`, and peels postlude tail layers from the top onto `CUDA1` to preserve KV cache, compute buffer, and WDDM headroom on `CUDA0`.
- Kept small `LLM_ARCH_DFLASH` draft models unbroken on the GPU with the most free VRAM and capped DFlash prompt-processing graph reservation to draft-block size rather than full `n_tokens`.
- Note: Passing a manual non-zero `-ts` disables the loop-aware layout and falls back to a single contiguous split. To shift an extra postlude layer to `CUDA1` without breaking the loop core, use `-ot "blk\.19\..*=CUDA1"`.

### 2. Inkling-2 2-GPU Graph Cleanup & Batched H2D Input Copies (`49dcf342e`)
- Files: `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, `src/models/models.h`, `ggml/src/ggml-backend.cpp`
- Fixed `stack()` / `mix()` device assignment for `il < 0` (`model.dev_output()`) and passed `il` into loop `mix(bank, bank_out, loop.msg_query, il)` to avoid redundant cross-GPU ping-pong copies.
- Cached `view_3d` tensors across splits, expanded `l_out` at cross-GPU boundaries, and grouped `wqkv_gate` with `wq` / `attn_rel` / `sinks` to maximize `Q8_1` `src1_cache` reuse.
- Padded the projected relative-bias table instead of `attn_rel_bias` (enabling `pad_f32` elision in `fattn-rel.cu`).
- Added a batched per-split host input transfer path (`set_tensor_n` / grouped `GGML_TENSOR_FLAG_INPUT` copies) in `ggml_backend_sched`, cutting WDDM H2D synchronization stalls per token from 21 down to 3.

### 3. Ada (`sm_89`) & Inkling-2 CUDA Kernel Fusions and Vectorization (`58427ca83`)
- Files: `ggml/src/ggml-cuda/depth-mix.cu`, `elem-fuse.cu`, `elem-fuse.cuh`, `fattn-rel.cu`, `fattn-common.cuh`, `ggml-cuda.cu`, `situ.cu`, `moe-weighted-reduction.cu`
- **`depth-mix.cu`**: Added `nq = 4`, low-register odd-`n` `mm2`, and Form (c) cached-stack fusion (`[REPEAT,] MUL_MAT, SOFT_MAX, MUL_MAT` with `depth_mix_is_transpose_stack`), fusing all 24/24 `loop.msg_query` mixes and eliminating cuBLAS `gemmSN_TN_kernel` + `splitKreduce` launches.
- **`fattn-rel.cu` & `fattn-common.cuh`**: Added `Q8_0` (`D = 64/128/256`) KV support to `rel_bias_fa` and elided all 48 `pad_f32` kernels/token via a `tab_nrows` bounds check.
- **`elem-fuse.cu` & `ggml-cuda.cu`**:
  - Enabled `ADD + RMS_NORM` fusion on Ada (`sm_89`).
  - Added `ctrl_mmvf_gate_f32_vec4` (fuses `ctrl_forget` + `ctrl_gain` + double `tanh_mul_addsub` for `T <= 3`).
  - Added `ctrl_gate_rms_norm_mul_f32` (fuses double `tanh_mul_addsub` + `RMS_NORM` + `MUL(w)` for `T > 3`).
  - Added `concat_mmvf_f32` (fuses `CONCAT(dim=0)` + `MUL_MAT` for `T <= 3`).
- **`situ.cu` & `moe-weighted-reduction.cu`**: Added `float4` vectorized paths (`situ_f32` -27% on `pp512`, `moe_weighted_reduction` -9% median).

### 4. Backend Sampling (`-bs`) Full-Vocab D2H Copy Elimination (`a3defaf57`)
- Files: `include/llama.h`, `src/llama.cpp`, `src/llama-sampler.h`, `src/llama-sampler.cpp`, `common/sampling.cpp`
- Added `llama_sampler_chain_params::skip_distributions` (enabled automatically when `n_probs == 0`, `n_outputs_max_per_seq == 1`, and GPU sampling produces `data->sampled`).
- Eliminates 1.6-2.4 MB/token of unused full-vocab `logits` / `probs` / `candidates` D2H copies over PCIe 3.0 x8, transferring only the 4-byte sampled token ID (`+8.8 t/s` at `temp=1.0`).
- Automatically falls back when `logprobs` (`n_probs > 0`) or CPU-only samplers (grammar, penalties, DRY, etc.) are requested.

## Measured Performance

### `llama-server` Real Chat (~7,550 tokens, `-c 200000 -ub 512 -ctk q8_0 -ctv q8_0 --mmproj-device CUDA1`)

| Build / Stage | Prefill (`pp`) | Decode (`tg`) | Notes |
|---|---:|---:|---|
| Upstream baseline (`849bf6392`) | `638.05 t/s` | `32.07 t/s` | Split through looped core + WDDM overhead |
| After loop-aware split (`9ad0be05f`) | `1,348 t/s` | `56.6 t/s` | Loop core `[6, 18)` intact on `CUDA0` |
| Current `HEAD` + `-bs` (`temp=1.0, top_p=1.0`) | **`1,823.13 t/s`** | **`69.57 t/s`** | **2.86x pp, 2.17x tg**, 0 MiB WDDM spill |

### `llama-bench` (`-ctk q8_0 -ctv q8_0 -fa 1 -ngl 99 -sm layer -ts 0 -lm none -r 3`)

| Test | After `9ad0be05f` | Current `HEAD` | Gain |
|---|---:|---:|---:|
| `pp512` | `1,968.1 t/s` | **`2,457.67 +/- 29.65 t/s`** | `+24.9%` |
| `tg128` | `61.46 t/s` | **`69.18 +/- 0.28 t/s`** | `+12.6%` |
| `pp512 @ d8192` | `1,607.8 t/s` | **`1,937.03 +/- 75.37 t/s`** | `+20.5%` |
| `tg128 @ d8192` | `59.58 t/s` | **`66.14 +/- 0.21 t/s`** | `+11.0%` |

### Long-Context Stress Tests (`llama-server`, `mmproj` on `CUDA1`, `q8_0` KV, `-ub 512 -bs`)

| Context | Prompt + Gen | Prefill Avg | Decode @ Full Depth | GPU0 Min Free | GPU1 Min Free | WDDM Spill | Graph Reallocs (same/changed) |
|---|---|---:|---:|---:|---:|---:|---|
| **200k** (`-c 200000`) | `199,800 + 128` | `792.91 t/s` (`252s`) | `35.41 t/s` | `2,228 MiB` | `7,657 MiB` | `0 MiB` | `388 / 0` |
| **256k** (`-c 262144`) | `261,950 + 128` | `666.49 t/s` (`393s`) | `30.16 t/s` | `1,128 MiB` | `7,157 MiB` | `0 MiB` | `509 / 0` |

## Correctness & Accuracy Verification

Verified against completely unfused CUDA execution (`GGML_CUDA_DISABLE_FUSION=1`) on `Inkling-2-Mini-UD-IQ4_XS.gguf` (`-ctk q8_0 -ctv q8_0 -fa on`):

- **Perplexity (`llama-perplexity`)**:
  - Prefill path (`src/llama-model.cpp`, 4,096 tokens, `-ub 512`): `1.3549 +/- 0.0354` (all fusions on) vs `1.3548 +/- 0.0353` (all fusions off, `delta +0.0001`).
  - English docs (`docs/build.md`, 4,096 tokens, `-ub 512`): `4.3967 +/- 0.2167` (all on) vs `4.4062 +/- 0.2173` (all off, `delta -0.0095`).
  - Decode-only path (`-b 1 -ub 1`, 512 tokens, exercising all `T <= 3` kernels): `1.6107 +/- 0.1301` (all on, bit-deterministic across repeats) vs `1.6050 +/- 0.1287` (all off, `delta +0.0057`).
- **Bit-exact components (`0.000e+00` diff)**: `rel_bias_fa` Q8_0 + `pad` elision (verified `0.000e+00` across 16 decode steps and identical PPL), `situ_f32_vec4`, `moe_weighted_reduction_f32_vec4`, `set_tensor_n` batching, `-bs` `skip_distributions` (128/128 exact greedy match), and `depth_mix` for `n = 2, nq <= 3` (`0` differing bytes under `GGML_CUDA_DEPTH_MIX_SHADOW=1`).
- **1-ULP FP32 reassociation**: `depth_mix` (`nq = 4`, `n >= 3`), `elem-fuse` control kernels, and upstream `ADD + RMS_NORM` on `sm_89` differ only by IEEE-754 reduction order (`16/16` top-1 agreement at 1,500 tokens, `90+` matching greedy tokens before near-tie split).

## Recommended Launch Command

Template scripts are included at the repository root: copy `example.build.ps1` -> `build.ps1` and `example.run.ps1` -> `run.ps1`, then fill in your local CUDA/model paths.

```powershell
llama-server.exe `
    -m Inkling-2-Mini-UD-IQ4_XS.gguf `
    --mmproj mmproj-F16.gguf --mmproj-device CUDA1 `
    -c 200000 -ub 512 -np 1 -ctk q8_0 -ctv q8_0 `
    -ngl 99 -fa on -fit off -lm none -bs `
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
```

## Hardware Switching Guide

The C++/CUDA changes are portable across single-GPU and multi-GPU NVIDIA setups and do not hardcode 2 GPUs or specific card models.

### Switching to a Single GPU (e.g., 24 GB RTX 3090 / 4090 or 32 GB RTX 5090)
- **Loop-aware split auto-disables**: `src/llama-model.cpp` gates custom multi-GPU placement on `n_devices() >= 2`, so single-GPU runs place everything on `CUDA0` with no extra flags.
- **Single-GPU speedups stay active**: `Q8_0` `rel_bias_fa` + 48 `pad_f32` elisions, `elem-fuse.cu` control kernels, `depth_mix` Form (c), `float4` `situ` / `moe_weighted_reduction`, batched `set_tensor_n` H2D copies, and `-bs` `skip_distributions` all remain active on 1 GPU.
- **VRAM sizing on 24 GB**: `Inkling-2-Mini-UD-IQ4_XS` (~14.5 GB) + `mmproj-F16` (~0.8 GB) + desktop (~1 GB) leaves ~7.5 GB for KV + compute. Expect ~160k-200k `q8_0` KV to fit cleanly; 256k on 24 GB needs `q4_0` KV (`-ctk q4_0 -ctv q4_0`) or disabling `mmproj`.

### Switching CUDA Architectures
- **Ada (`sm_89`, RTX 40xx), Hopper (`sm_90`), Blackwell (`sm_100` / `sm_120`, RTX 50xx)**: 100% of optimizations stay enabled (`cc >= GGML_CUDA_CC_ADA_LOVELACE`). Swapping between Ada GPUs (e.g., 4070 Ti SUPER -> 4090) does not even require a rebuild.
- **Ampere (`sm_86` RTX 30xx / `sm_80` A100) or Turing (`sm_75`)**: works cleanly after rebuilding for that architecture. Two arch guards automatically disable `ADD + RMS_NORM` fusion and decode-size `depth_mix` (`T <= 8`) on `cc < 89`, while prefill `depth_mix`, `fattn-rel` `Q8_0`, `elem-fuse`, `situ`, `moe_weighted_reduction`, `set_tensor_n`, and `-bs` remain active.

### Script Updates When Swapping GPUs
1. **`build.ps1`**: set `-DCMAKE_CUDA_ARCHITECTURES` to your target arch (e.g., `86` for RTX 3090, `89` for RTX 4090, `120` for RTX 5090, or `native` to auto-detect the installed GPU), then run `.\build.ps1 -reconfigure`.
2. **`run.ps1`**: on a single-GPU machine, remove `--mmproj-device CUDA1` (or change it to `CUDA0`).

Debug toggles for isolating individual CUDA fusions:
- `GGML_CUDA_DISABLE_FUSION=1` (disable all fusions)
- `GGML_CUDA_DISABLE_CTRL_FUSION=1`
- `GGML_CUDA_DISABLE_DEPTH_MIX_FUSION=1`
- `GGML_CUDA_DISABLE_ADD_RMS_NORM_FUSION=1`
- `GGML_CUDA_DISABLE_REL_BIAS_FA=1`
