#include "clip-rows.cuh"
#include "ggml-common.h"

// one block per row; the sum of squares is accumulated exactly like reduce_row_f32 (reduce_rows.cuh) so that the
// result is bit-identical to SQR + SUM_ROWS for the same block size
template <bool has_w, bool has_tau>
static __global__ void clip_rows_f32(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ limit,
        const float * __restrict__ tau, float * __restrict__ dst, const int ncols, const int w_rows, const int tau_stride,
        const float limit_value, const float bias, const float lo, const float hi) {
    const int64_t row = blockIdx.x;
    const int     col = threadIdx.x;

    const float * xr = x + row*ncols;
    const float * wr = has_w ? w + int64_t(row % w_rows)*ncols : nullptr;

    auto value = [&](int i) {
        return has_w ? __fmul_rn(xr[i], wr[i]) : xr[i];
    };

    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    ggml_cuda_pdl_sync();
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                const float c = value(i);
                temp[j] = __fmul_rn(c, c);
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] = __fadd_rn(sum_temp[j], temp[j]);
        }
    }
    float sum = 0.0f;
    for (int j = 0; j < num_unroll; ++j) {
        sum = __fadd_rn(sum, sum_temp[j]);
    }

    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);
    ggml_cuda_pdl_lc();

    // SQRT, SCALE (scale 1: x*1 + bias rounds once either way), DIV, CLAMP as in their own kernels
    const float norm   = sqrtf(sum) + bias;
    const float factor = fminf(fmaxf((limit ? limit[row] : limit_value) / norm, lo), hi);
    const float t      = has_tau ? tau[row / tau_stride] : 1.0f;

    for (int i = col; i < ncols; i += blockDim.x) {
        const float y = __fmul_rn(value(i), factor);
        dst[row*ncols + i] = has_tau ? __fmul_rn(y, t) : y;
    }
}

// One warp per 128-element head: clip + optional tau + FWHT(128) + F32 store or Q8_0 set_rows scatter.
template <bool has_w, bool has_tau, bool to_q8_0>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void clip_rows_fwht_128_kernel(
        const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ limit,
        const float * __restrict__ tau, const int64_t * __restrict__ idxs, void * __restrict__ dst_ptr,
        const int64_t nrows, const int w_rows, const int n_heads, const int64_t idxs_s10, const int64_t dst_nb1,
        const float limit_value, const float bias, const float lo, const float hi) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32, "clip_rows_fwht_128_kernel requires 32-lane warps");

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= nrows) {
        return;
    }

    const int lane = threadIdx.x;
    const float * xr = x + r * 128;
    const float * wr = has_w ? w + int64_t(r % w_rows) * 128 : nullptr;

    float reg[4];
    float ss = 0.0f;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v = xr[i * 32 + lane];
        const float c = has_w ? __fmul_rn(v, wr[i * 32 + lane]) : v;
        reg[i] = c;
        ss += c * c;
    }

    ss = warp_reduce_sum<32>(ss);
    const float norm   = sqrtf(ss) + bias;
    const float factor = fminf(fmaxf((limit ? limit[r] : limit_value) / norm, lo), hi);
    const float t      = has_tau ? tau[r / n_heads] : 1.0f;
    const float scale  = factor * t * (1.0f / sqrtf(128.0f));

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reg[i] *= scale;
    }

    // FWHT(128) in registers
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, 32);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    {
        const float x0 = reg[0], y0 = reg[1];
        reg[0] = x0 + y0; reg[1] = x0 - y0;
        const float x1 = reg[2], y1 = reg[3];
        reg[2] = x1 + y1; reg[3] = x1 - y1;
    }
    {
        const float x0 = reg[0], y0 = reg[2];
        reg[0] = x0 + y0; reg[2] = x0 - y0;
        const float x1 = reg[1], y1 = reg[3];
        reg[1] = x1 + y1; reg[3] = x1 - y1;
    }

    ggml_cuda_pdl_lc();

    if constexpr (to_q8_0) {
        const int64_t tok     = r / n_heads;
        const int64_t head    = r % n_heads;
        const int64_t dst_row = idxs[tok * idxs_s10];
        block_q8_0 * dst_b = (block_q8_0 *) ((char *) dst_ptr + dst_row * dst_nb1) + head * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float amax = warp_reduce_max<32>(fabsf(reg[i]));
            const float d    = amax / 127.0f;
            const float id   = d ? 1.0f / d : 0.0f;
            if (lane == 0) {
                dst_b[i].d = d;
            }
            dst_b[i].qs[lane] = (int8_t) roundf(reg[i] * id);
        }
    } else {
        float * dst = (float *) dst_ptr + r * 128;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            dst[i * 32 + lane] = reg[i];
        }
    }
}

void ggml_cuda_op_clip_rows_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
        const ggml_tensor * limit, float limit_value, float bias, float lo, float hi,
        const ggml_tensor * tau, ggml_tensor * dst) {
    const int     ncols = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    // the block size of sum_rows_f32_cuda for this shape
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int block_size = (nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());

    const float * w_d     = w ? (const float *) w->data : nullptr;
    const float * limit_d = limit ? (const float *) limit->data : nullptr;
    const float * tau_d   = tau ? (const float *) tau->data : nullptr;
    const int     w_rows  = w ? (int) ggml_nrows(w) : 1;
    const int     tau_stride = (int) x->ne[1];

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, (const float *) x->data, w_d, limit_d, tau_d, (float *) dst->data,
            ncols, w_rows, tau_stride, limit_value, bias, lo, hi);
    };
    if (w) {
        tau ? launch(clip_rows_f32<true, true>) : launch(clip_rows_f32<true, false>);
    } else {
        tau ? launch(clip_rows_f32<false, true>) : launch(clip_rows_f32<false, false>);
    }
}

void ggml_cuda_op_clip_rows_fwht_128(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
        const ggml_tensor * limit, float limit_value, float bias, float lo, float hi,
        const ggml_tensor * tau, ggml_tensor * dst, const ggml_tensor * set_rows_idxs) {
    const int64_t nrows   = ggml_nrows(x);
    const int     n_heads = (int) x->ne[1];
    const int     w_rows  = w ? (int) ggml_nrows(w) : 1;

    constexpr int rows_per_block = 4;
    const int64_t num_blocks = (nrows + rows_per_block - 1) / rows_per_block;
    const ggml_cuda_kernel_launch_params launch_params(dim3(num_blocks, 1, 1), dim3(32, rows_per_block, 1), 0, ctx.stream());

    const float   * w_d     = w ? (const float *) w->data : nullptr;
    const float   * limit_d = limit ? (const float *) limit->data : nullptr;
    const float   * tau_d   = tau ? (const float *) tau->data : nullptr;
    const int64_t * idxs_d  = set_rows_idxs ? (const int64_t *) set_rows_idxs->data : nullptr;
    const int64_t   idxs_s  = set_rows_idxs ? (int64_t) (set_rows_idxs->nb[0] / sizeof(int64_t)) : 0;
    const int64_t   dst_nb1 = set_rows_idxs ? (int64_t) dst->nb[1] : 0;

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, (const float *) x->data, w_d, limit_d, tau_d, idxs_d, dst->data,
            nrows, w_rows, n_heads, idxs_s, dst_nb1, limit_value, bias, lo, hi);
    };
    if (set_rows_idxs) {
        if (w) {
            tau ? launch(clip_rows_fwht_128_kernel<true, true, true>) : launch(clip_rows_fwht_128_kernel<true, false, true>);
        } else {
            tau ? launch(clip_rows_fwht_128_kernel<false, true, true>) : launch(clip_rows_fwht_128_kernel<false, false, true>);
        }
    } else {
        if (w) {
            tau ? launch(clip_rows_fwht_128_kernel<true, true, false>) : launch(clip_rows_fwht_128_kernel<true, false, false>);
        } else {
            tau ? launch(clip_rows_fwht_128_kernel<false, true, false>) : launch(clip_rows_fwht_128_kernel<false, false, false>);
        }
    }
}
