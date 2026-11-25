# Fork Optimizations Summary (`opt/2gpu-fast`)

This document summarizes the performance, multi-GPU placement, and VRAM optimizations added in this fork on top of `849bf6392` (`unslothai/llama.cpp#433`).

Target testbed: Windows (WDDM), `CUDA0` RTX 4070 Ti SUPER (16 GiB, 672 GB/s, display GPU) + `CUDA1` RTX 4060 Ti (16 GiB, 288 GB/s, PCIe 3.0 x1), running `Inkling-2-Mini-UD-IQ4_XS.gguf` (24 physical -> 48 logical layers with `[6, 18)` looped 3x, 128-expert top-8 MoE) + `mmproj-F16.gguf` on `CUDA1` with `q8_0` KV cache up to 256k context.

## Summary of Changes

### 1. Loop-Aware 2-GPU Split & DFlash Reserve Cap (`9ad0be05f`)
- Files: `src/llama-model.cpp`, `src/llama-model.h`, `src/llama-context.cpp`, `src/models/models.h`
- Added `loop_core_range()` so automatic layer splitting (`-sm layer`, default auto `-ts`) keeps the entire 3x looped core `[6, 18)` intact on the faster primary GPU (`CUDA0`), offloads prelude `[0, 6)` to `CUDA1`, and peels postlude tail layers from the top onto `CUDA1` to preserve KV cache, compute buffer, and WDDM headroom on `CUDA0`.
- Kept small `LLM_ARCH_DFLASH` draft models unbroken on the GPU with the most free VRAM and capped DFlash prompt-processing graph reservation to draft-block size rather than full `n_tokens`.
- Note: Passing a manual non-zero `-ts` disables the loop-aware layout and falls back to a single contiguous split. To shift an extra postlude layer to `CUDA1` without breaking the loop core, use `-ot "blk\.19\..*=CUDA1"`.

### 2. Inkling-2 2-GPU Graph Cleanup & Batched H2D Input Copies (`49dcf342e`)
- Files: `src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, `src/models/models.h`, `ggml/src/ggml-backend.cpp`
- Fixed `stack()` / `mix()` device assignment for `il < 0` (`model.dev_output()`) and passed `il` into loop `mix(bank, bank_out, loop.msg_query, il)` to avoid redundant cross-GPU ping-pong copies.
- Expanded `l_out` at cross-GPU boundaries and grouped `wqkv_gate` with `wq` / `attn_rel` / `sinks` to maximize `Q8_1` `src1_cache` reuse.
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
- Eliminates 1.6-2.4 MB/token of unused full-vocab `logits` / `probs` / `candidates` D2H copies over PCIe, transferring only the 4-byte sampled token ID (`+8.8 t/s` at `temp=1.0`).
- Automatically falls back when `logprobs` (`n_probs > 0`) or CPU-only samplers (penalties, DRY, etc.) are requested.

### 5. Tier 1: Primary `lm_head`, Maskless `Q8_0` Banded Decode, & Lazy Grammar `-bs` (`432a556a6`)
- Files: `src/llama-model.cpp`, `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-graph.cpp`, `src/llama-graph.h`, `src/models/inkling2.cpp`, `ggml/src/ggml.c`, `ggml/src/ggml-cuda/fattn-banded.cu`, `ggml/src/ggml-cuda/fattn.cu`, `common/sampling.cpp`
- Kept `output.weight` (`lm_head`) on the faster primary GPU (`CUDA0`) instead of `CUDA1`, shifting an extra postlude layer to `CUDA1` to preserve `CUDA0` KV headroom.
- Added maskless contiguous single-sequence `Q8_0` banded FlashAttention decode (`attn_inp_n_kv_valid`, passing a 4-byte valid count instead of full mask/rel-pos/shift buffers) and tuned `fattn-banded.cu` for Ada `Q8_0` decode.
- Updated `common/sampling.cpp` so backend sampling (`-bs`) stays active alongside lazy grammars (such as tool calling), falling back to CPU grammar sampling only when a lazy trigger activates.

### 6. Tier 2: Inkling-2 `Q8_1`/FWHT/KV Fusions, Warp-Per-Row `Q8_0` GEMV, & 2-GPU Graph Cleanup (`94f475d74`)
- Files: `ggml/src/ggml-cuda/{clip-rows.cu,clip-rows.cuh,common.cuh,depth-mix.cu,elem-fuse.cu,elem-fuse.cuh,fwht.cu,fwht.cuh,ggml-cuda.cu,mmvf.cu,mmvq.cu,norm.cu,norm.cuh,situ.cu,situ.cuh,vecdotq.cuh}`, `src/{llama-context.cpp,llama-graph.cpp,models/inkling2.cpp,models/models.h}`
- **Quantized & F32 GEMV / small-GEMM kernels**:
  - `vecdotq.cuh`: hoisted shared `block_q8_1` scale and packed 16-bit FP8 scale conversion in `vec_dot_nvfp4_q8_1`.
  - `mmvq.cu`: decoupled NVFP4 `x_scale` from `has_fusion` (`REG: 107 -> 69`), added zero-shared-mem `mul_mat_vec_q_warp<GGML_TYPE_Q8_0, 1>` (Turing+), and added LIFO-safe read-only `ggml_cuda_src1_cache_peek` for `MUL_MAT_ID`.
  - `elem-fuse.cu` & `mmvf.cu`: added `mul_mat_f32_small_k` (eliminates `48x/tok` cuBLAS split-K launches) and `mul_mat_vec_f32_vec4<512>` for few-row long-K F32 matvecs (MoE router) on `nsm >= 48` GPUs.
- **Producer-side `Q8_1` side-outputs (`quantize_q8_1`: `249x -> 1x/tok`)**:
  - Added 24-node lookahead `ggml_cuda_src1_cache_prepare_q8_1` in `ggml-cuda.cu` and `Q8_1` side-output paths in `rms_norm_mul_f32_q8_1` (`norm.cu`), `situ_f32_q8_1` (`situ.cu`), and `fwht_sigmoid_mul_f32` (`fwht.cu`); tuned `rms_norm_f32` to `<256, 4>` / `<512, 2>`.
- **Head-wise FWHT / KV-store / attention-gate fusions (`fwht_cuda`: `192x -> 0`, `set_rows_quant`: `96x -> 0`, `sigmoid`: `48x -> 0`)**:
  - `clip-rows.cu`: added optional per-token `tau` scaling and `clip_rows_fwht_128_kernel` (`Qcur` F32 and `Kcur` `Q8_0` `set_rows`).
  - `fwht.cu`: added `rms_norm_128_fwht_set_rows_q8_0_kernel` (`Vcur`) and `fwht_sigmoid_mul_f32` (`kqv_out` + optional `Q8_1` side-output).
- **2-GPU graph, fusion allocator deps, & progressive CUDA graph chunking**:
  - Removed `states_t` concats and duplicate `view_3d` peer copies in `inkling2.cpp`, pinned `res_norm` to the target layer/output device in `llama-context.cpp` (cutting `SPLIT #3`/`#4` peer copies `200.6 KB -> 36 KB`), and ordered K/V/Q prep and KV stores (`llama-graph.cpp`) for contiguous fusion + `src1_cache` hits.
  - Added `depth-mix.cu` Form (b) `CONT(PERMUTE)` support and small-batch `add_alloc_dep` guards in `ggml_backend_cuda_graph_optimize` (`rel_bias_fa` `80/80` fused, NVFP4 MoE scale `48/48` fused).
  - Switched default CUDA graph chunking to progressive sizing (`128, 384, 1152, ...`) to overlap early GPU execution with WDDM submission latency.

## Measured Performance

### `llama-bench` (`-ctk q8_0 -ctv q8_0 -fa 1 -ngl 99 -sm layer -ts 0 -lm none -ub 512 -r 3`)

| Test | After `9ad0be05f` | Pre-Tier 1 (`a3defaf57`) | Tier 1 (`432a556a6`) | Tier 2 (`94f475d74`) | Gain vs `9ad0be05f` |
|---|---:|---:|---:|---:|---:|
| `pp512` | `1,968.1 t/s` | `2,457.67 +/- 29.65 t/s` | - | **`2,635.34 +/- 37.37 t/s`** | **`+33.9%`** |
| `tg128` | `61.46 t/s` | `69.18 +/- 0.28 t/s` | `73.37 +/- 3.20 t/s` | **`83.49 +/- 3.61 t/s`** | **`+35.8%`** |
| `pp512 @ d8192` | `1,607.8 t/s` | `1,937.03 +/- 75.37 t/s` | - | **`2,024.36 +/- 75.69 t/s`** | **`+25.9%`** |
| `tg128 @ d8192` | `59.58 t/s` | `66.14 +/- 0.21 t/s` | `72.05 +/- 1.36 t/s` | **`81.42 +/- 0.55 t/s`** | **`+36.7%`** |
| `pp512 @ d32768` | - | - | - | **`1,474.13 +/- 52.29 t/s`** | - |
| `tg128 @ d32768` | - | - | `66.93 +/- 1.06 t/s` | **`76.00 +/- 0.41 t/s`** | **`+13.6%` vs Tier 1** |
| `tg64 @ d131072` | - | - | - | **`57.93 +/- 1.10 t/s`** | - |

### Nsight Systems Steady-State Decode (`llama-completion`, 20-token median, `-c 8192 -bs`)

| Stage | Step Time | Throughput | Kernels / Token (`dev0` + `dev1`) | Cross-GPU Gaps |
|---|---:|---:|---:|---:|
| Tier 1 (`432a556a6`) | `13.389 ms/tok` | `74.69 t/s` | `2,747` (`2240` + `507`) | `1,125.7 us` |
| Tier 2 (`94f475d74`) | **`11.896 ms/tok`** | **`84.06 t/s`** | **`1,998` (`1642` + `356`)** | **`712.8 us`** |

### `llama-server` End-to-End (`-ctk q8_0 -ctv q8_0 -ub 512 -bs --mmproj-device CUDA1`)

| Build / Stage | Workload & Context | Prefill (`pp`) | Decode (`tg`) | Notes |
|---|---|---:|---:|---|
| Upstream baseline (`849bf6392`) | ~7,550-tok chat (`-c 200000`) | `638.05 t/s` | `32.07 t/s` | Split through looped core + WDDM overhead |
| After loop-aware split (`9ad0be05f`) | ~7,550-tok chat (`-c 200000`) | `1,348 t/s` | `56.6 t/s` | Loop core `[6, 18)` intact on `CUDA0` |
| Pre-Tier 1 (`a3defaf57`) | ~7,550-tok chat (`-c 200000`) | `1,823.13 t/s` | `69.57 t/s` | `0 MiB` WDDM spill |
| Tier 2 (`94f475d74`) | 2,209-tok prompt + `tools` (`-c 262144`) | `1,511.09 t/s` | **`80.34 t/s`** (`80.71 t/s` `tg_3s`) | Lazy grammar + `-bs`, `CUDA0` `888 MiB` free / `CUDA1` `6,875 MiB` free, `0 MiB` spill, `0` size-change reallocs |

### Long-Context Stress Tests (`llama-server`, `mmproj` on `CUDA1`, `q8_0` KV, `-ub 512 -bs`, measured at `a3defaf57`)

| Context | Prompt + Gen | Prefill Avg | Decode @ Full Depth | GPU0 Min Free | GPU1 Min Free | WDDM Spill | Graph Reallocs (same/changed) |
|---|---|---:|---:|---:|---:|---:|---|
| **200k** (`-c 200000`) | `199,800 + 128` | `792.91 t/s` (`252s`) | `35.41 t/s` | `2,228 MiB` | `7,657 MiB` | `0 MiB` | `388 / 0` |
| **256k** (`-c 262144`) | `261,950 + 128` | `666.49 t/s` (`393s`) | `30.16 t/s` | `1,128 MiB` | `7,157 MiB` | `0 MiB` | `509 / 0` |

## Correctness & Accuracy Verification

Verified on `94f475d74` against completely unfused CUDA execution (`GGML_CUDA_DISABLE_FUSION=1`) on `Inkling-2-Mini-UD-IQ4_XS.gguf` (`-ctk q8_0 -ctv q8_0 -fa on`):

- **CUDA backend unit tests (`test-backend-ops`)**: **18,006 / 18,006 passed** across both `CUDA0` and `CUDA1` (`MUL_MAT`, `MUL_MAT_ID`, `RMS_NORM`, `SET_ROWS`, `CONCAT`, `FLASH_ATTN_EXT`).
- **Greedy 128-token parity (`--temp 0 -bs -no-cnv`)**: **100% bit-exact** on both short (`647/647` chars) and >128-token (`1,417/1,417` chars) prompts vs `GGML_CUDA_DISABLE_FUSION=1`.
- **Perplexity (`llama-perplexity`)**:
  - Prefill (`CONTRIBUTING.md`, `-c 512 -b 512 -ub 512 --chunks 2`): `4.2444 +/- 0.4206` (all fusions on) vs `4.2633 +/- 0.4247` (all fusions off), and bit-identical `4.2444` with `GGML_CUDA_DISABLE_Q8_1_REUSE=1`, `GGML_CUDA_DISABLE_SITU_FUSION=1`, `GGML_CUDA_DISABLE_SHORT_CONV_FUSION=1`, and `GGML_CUDA_DISABLE_REL_BIAS_FA_FUSION=1`.
  - Decode-only (`CONTRIBUTING.md`, `-c 128 -b 1 -ub 1 --chunks 1`): `5.7278 +/- 1.4620` (all on) vs `5.7179 +/- 1.4511` (all off).
  - Pre-Tier 1 4k-token baseline (`src/llama-model.cpp`: `1.3549` vs `1.3548`; `docs/build.md`: `4.3967` vs `4.4062`).
- **1-ULP FP32 reassociation**: `depth_mix` (`nq = 4`, `n >= 3`), `elem-fuse` control kernels, `ADD + RMS_NORM`, and warp-reduced `clip_rows_fwht_128` / `rms_norm_128_fwht_set_rows_q8_0` differ only by IEEE-754 reduction order.

## Recommended Launch Command

Template scripts are included at the repository root: copy `example.build.ps1` -> `build.ps1` and `example.run.ps1` -> `run.ps1`, then fill in your local CUDA/model paths.

```powershell
llama-server.exe `
    -m Inkling-2-Mini-UD-IQ4_XS.gguf `
    --mmproj mmproj-F16.gguf --mmproj-device CUDA1 `
    -c 262144 -ub 512 -np 1 -ctk q8_0 -ctv q8_0 `
    -ngl 99 -fa on -fit off -lm none -bs `
    --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0
```

## Hardware Switching Guide

The C++/CUDA changes are portable across single-GPU and multi-GPU NVIDIA setups and do not hardcode 2 GPUs or specific card models.

### Switching to a Single GPU (e.g., 24 GB RTX 3090 / 4090 or 32 GB RTX 5090)
- **Loop-aware split auto-disables**: `src/llama-model.cpp` gates custom multi-GPU placement on `n_devices() >= 2`, so single-GPU runs place everything on `CUDA0` with no extra flags.
- **Single-GPU speedups stay active**: `Q8_0` `rel_bias_fa` + `pad` elision, `elem-fuse.cu` kernels, `depth_mix` Form (b)/(c), producer-side `Q8_1` side-outputs, FWHT/KV fusions, `mul_mat_vec_q_warp<Q8_0>`, NVFP4 `vecdotq`/`x_scale`, batched `set_tensor_n` H2D copies, and `-bs` `skip_distributions` + lazy grammar support all remain active on 1 GPU.
- **VRAM sizing on 24 GB**: `Inkling-2-Mini-UD-IQ4_XS` (~14.5 GB) + `mmproj-F16` (~0.8 GB) + desktop (~1 GB) leaves ~7.5 GB for KV + compute. Expect ~160k-200k `q8_0` KV to fit cleanly; 256k on 24 GB needs `q4_0` KV (`-ctk q4_0 -ctv q4_0`) or disabling `mmproj`.

### Higher-Precision Quants & Looped-Core Split Sizing (2 GPUs)
- **Quant compatibility**: all Tier 1 and Tier 2 graph/attention/FWHT/`Q8_1`-cache/controller/sampling optimizations work across any quantized weight type (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `UD-*_XL`, etc.). `mul_mat_vec_q_warp<Q8_0, 1>` covers even more layers on higher quants, while only the two `NVFP4` helpers (`vecdotq.cuh`, `mmvq.cu` `x_scale`) are specific to `NVFP4` MoE weights.
- **16 + 16 GiB auto-split thresholds (`src/llama-model.cpp`)**: `primary_kv_cap` reserves up to `5.5 GiB` on `CUDA0` for 256k `q8_0` KV + compute + WDDM (~`9.5 GiB` weight cap on a 16 GiB display GPU; ~`8.7 GiB` with `lm_head` compute headroom):
  - **`~15-17.5 GiB` quants** (e.g., `UD-IQ4_XS`, `Q4_K_M`, `UD-Q4_K_XL`, core `[6, 18)` ~`6.8-8.3 GiB`): entire looped core `[6, 18)` + `lm_head` stay on `CUDA0`; more postlude layers automatically peel to `CUDA1`.
  - **`~18-20 GiB` quants** (e.g., `Q5_K_M`, `UD-Q5_K_XL`, core ~`8.4-9.5 GiB`): looped core `[6, 18)` stays intact on `CUDA0`, while `lm_head` spills to `CUDA1` (minor `~0.6 ms/tok` penalty).
  - **`>20 GiB` quants** (e.g., `Q6_K` ~`21.5 GiB`, `Q8_0` ~`27.5 GiB`, core ~`10.5-13.5 GiB`): `bytes_core` exceeds `primary_kv_cap`, so auto-split falls back to a contiguous split through the middle of `[6, 18)`. On a 2-GPU setup with a slower secondary card (`288 GB/s` vs `672 GB/s`) or WDDM sync latency, splitting inside the 3x looped core multiplies the slower GPU's bandwidth penalty 3x and adds 4 extra cross-GPU sync boundaries per token. To keep `[6, 18)` unbroken on 16 + 16 GiB at smaller context (`-c 65536`/`131072`), override placement with `-ot` or relax the `primary_kv_cap` fallback in `src/llama-model.cpp`.
- **Upgrading GPUs**: on a single 24-32 GB card there is no split at all; on a 2-GPU setup with a 24+ GB primary card (`primary_kv_cap` ~`18.5 GiB`), `[6, 18)` + `lm_head` stay intact on `CUDA0` all the way through `Q8_0`, and a faster-VRAM secondary card on a wider PCIe link further minimizes any remaining 2-GPU overhead.

### Switching CUDA Architectures
- **Ada (`sm_89`, RTX 40xx), Hopper (`sm_90`), Blackwell (`sm_100` / `sm_120`, RTX 50xx)**: 100% of optimizations stay enabled (`cc >= GGML_CUDA_CC_ADA_LOVELACE`). Swapping between Ada GPUs (e.g., 4070 Ti SUPER -> 4090) does not require a rebuild.
- **Ampere (`sm_86` RTX 30xx / `sm_80` A100) or Turing (`sm_75`)**: works cleanly after rebuilding for that architecture. Arch guards disable `ADD + RMS_NORM` fusion, decode-size `depth_mix` (`T <= 8`), and FP8 packed NVFP4 scales on `cc < 89`, while `mul_mat_vec_q_warp<Q8_0>`, FWHT/KV fusions, producer-side `Q8_1` caching, `fattn-rel`/`fattn-banded` `Q8_0`, `elem-fuse`, `situ`, `set_tensor_n`, and `-bs` remain active.

### Caveats & Scope
- **NVIDIA CUDA only (not AMD HIP `wave64`)**: `clip_rows_fwht_128_kernel` and `rms_norm_128_fwht_set_rows_q8_0_kernel` have `static_assert(warp_size == 32)` and will not compile on 64-lane AMD HIP targets without an `if constexpr (warp_size == 32)` guard.
- **Ada / WDDM tuning**: `mul_mat_vec_q_warp<Q8_0>` is enabled for Turing+ NVIDIA desktop GPUs (`table_id == MMVQ_PARAMETERS_GENERIC`) and `mul_mat_vec_f32_vec4<512>` is gated on `nsm >= 48`; both were tuned on Ada (`sm_89`) and not benchmarked on non-Ada cards. Progressive CUDA graph chunking (`128, 384, 1152, ...`) targets Windows WDDM submission latency; on Linux, `GGML_CUDA_GRAPH_CHUNK_SIZE=0` (single graph) may match or beat it.
- **Inkling-2 & `Q8_0` KV specificity**: the FWHT/KV/SiTU fusions (`clip_rows_fwht_128`, `rms_norm_128_fwht_set_rows_q8_0`, `fwht_sigmoid_mul`, `situ_f32_q8_1`) specifically target Inkling-2 with `Q8_0` KV cache. Other models or KV types (e.g. `f16`/`q4_0`) cleanly fall back to the unfused paths while keeping the general CUDA optimizations.
- **`rms_norm_mul_f32_q8_1` width cap**: producer-side `Q8_1` fusion on `RMS_NORM * w` handles single-row F32 tensors with `ncols_padded <= 4096`; wider models (`n_embd > 4096`) fall back cleanly to the standard two-kernel path.
- **Decode scratch VRAM**: the small-batch (`<= 8` tokens) `add_alloc_dep` guards keep a few small decode buffers (~tens of KB) alive slightly longer to avoid aliasing that would block fusion; prefill peak VRAM is unaffected.
- **Per-fusion toggles**: `RMS_NORM(128)+FWHT+SET_ROWS` and `FWHT+SIGMOID+MUL` toggle via the master `GGML_CUDA_DISABLE_FUSION=1` flag and do not yet have individual disable env vars.

### Script Updates When Swapping GPUs
1. **`build.ps1`**: set `-DCMAKE_CUDA_ARCHITECTURES` to your target arch (e.g., `86` for RTX 3090, `89` for RTX 4090, `120` for RTX 5090, or `native` to auto-detect the installed GPU), then run `.\build.ps1 -reconfigure`.
2. **`run.ps1`**: on a single-GPU machine, remove `--mmproj-device CUDA1` (or change it to `CUDA0`).

Debug toggles for isolating individual optimizations:
- `GGML_CUDA_DISABLE_FUSION=1` (disable all CUDA graph fusions)
- `GGML_CUDA_DISABLE_Q8_1_REUSE=1` (disable `src1_cache` and producer-side `Q8_1` side-outputs)
- `GGML_CUDA_DISABLE_CLIP_ROWS_FUSION=1`
- `GGML_CUDA_DISABLE_CTRL_FUSION=1`
- `GGML_CUDA_DISABLE_DEPTH_MIX_FUSION=1`
- `GGML_CUDA_DISABLE_ADD_RMS_NORM_FUSION=1`
- `GGML_CUDA_DISABLE_REL_BIAS_FA_FUSION=1`
- `GGML_CUDA_GRAPH_CHUNK_SIZE=<N>` (`-1` progressive default, `0` single whole graph, `512` legacy fixed chunks)
- `LLAMA_INKLING_DECODE_BANDED_Q8=0` (revert full-attention `Q8_0` decode from banded to dense)
