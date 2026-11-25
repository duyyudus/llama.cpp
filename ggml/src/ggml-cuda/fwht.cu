#include "common.cuh"
#include "convert.cuh"
#include "fwht.cuh"
#include "ggml-common.h"

// wide FWHT blocks use one row per thread block with this many threads
#define GGML_CUDA_FWHT_BLOCK_NT 256

template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Wide blocks: one row per thread block instead of per warp, so each thread keeps N/NT
// values rather than N/32. Stages below the warp width still shuffle, those up to the
// block width go through shared memory, and the rest stay in registers.
template <int N, int NT, typename T>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T *     src_d = (const T *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            break;
    }

    // wide blocks: one row per thread block
    {
        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;

        dim3 grid_dims_w(rows, 1, 1);
        dim3 block_dims_w(nt, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params_w =
            ggml_cuda_kernel_launch_params(grid_dims_w, block_dims_w, 0, stream);

        switch (n) {
            case 1024:
                ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 2048:
                ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 4096:
                ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
#if !defined(GGML_USE_MUSA)
            // 32 KB of shared memory, above the MUSA limit; falls back there
            case 8192:
                ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
#endif // !defined(GGML_USE_MUSA)
            default:
                return false;
        }
    }
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_cuda_op_fwht_impl<half>(ctx, src, dst);
        default:
            return false;
    }
}

// One warp per 128-element head: RMS_NORM(128) + FWHT(64 or 128) + Q8_0 set_rows scatter.
template <int N_fwht>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void rms_norm_128_fwht_set_rows_q8_0_kernel(
        const float * __restrict__ x, const int64_t * __restrict__ idxs, void * __restrict__ dst_ptr,
        const int64_t nrows, const int n_heads, const int64_t idxs_s10, const int64_t dst_nb1, const float eps) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32 && (N_fwht == 64 || N_fwht == 128), "unexpected warp_size or N_fwht");

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= nrows) {
        return;
    }

    const int lane = threadIdx.x;
    const float * xr = x + r * 128;

    float reg[4];
    float ss = 0.0f;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float v = xr[i * 32 + lane];
        reg[i] = v;
        ss += v * v;
    }

    ss = warp_reduce_sum<32>(ss);
    const float scale = rsqrtf(ss / 128.0f + eps) * (1.0f / sqrtf((float) N_fwht));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        reg[i] *= scale;
    }

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
    if constexpr (N_fwht == 128) {
        const float x0 = reg[0], y0 = reg[2];
        reg[0] = x0 + y0; reg[2] = x0 - y0;
        const float x1 = reg[1], y1 = reg[3];
        reg[1] = x1 + y1; reg[3] = x1 - y1;
    }

    ggml_cuda_pdl_lc();

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
}

void ggml_cuda_op_rms_norm_fwht_set_rows_q8_0(ggml_backend_cuda_context & ctx, const ggml_tensor * x,
        float eps, int n_fwht, const ggml_tensor * idxs, ggml_tensor * dst) {
    const int64_t nrows   = ggml_nrows(x);
    const int     n_heads = (int) x->ne[1];
    constexpr int rows_per_block = 4;
    const int64_t num_blocks = (nrows + rows_per_block - 1) / rows_per_block;
    const ggml_cuda_kernel_launch_params launch_params(dim3(num_blocks, 1, 1), dim3(32, rows_per_block, 1), 0, ctx.stream());
    const int64_t idxs_s = (int64_t) (idxs->nb[0] / sizeof(int64_t));

    if (n_fwht == 64) {
        ggml_cuda_kernel_launch(rms_norm_128_fwht_set_rows_q8_0_kernel<64>, launch_params,
            (const float *) x->data, (const int64_t *) idxs->data, dst->data,
            nrows, n_heads, idxs_s, (int64_t) dst->nb[1], eps);
    } else {
        ggml_cuda_kernel_launch(rms_norm_128_fwht_set_rows_q8_0_kernel<128>, launch_params,
            (const float *) x->data, (const int64_t *) idxs->data, dst->data,
            nrows, n_heads, idxs_s, (int64_t) dst->nb[1], eps);
    }
}

// One warp per N-element row: FWHT(N) followed by sigmoid(gate) * val, with optional Q8_1 side-output.
template <int N, bool write_q8_1>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void fwht_sigmoid_mul_f32(
        const float * __restrict__ src, const float * __restrict__ gate, float * __restrict__ dst,
        block_q8_1 * __restrict__ dst_q8_1, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }

    src  += r * N;
    gate += r * N;
    dst  += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val  = reg[j];
            const float val2 = ggml_cuda_shfl_xor(val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const float g = gate[i * warp_size + lane];
        const float v = reg[i] * (1.0f / (1.0f + expf(-g)));
        dst[i * warp_size + lane] = v;
        if constexpr (write_q8_1) {
            const float  amax = warp_reduce_max<QK8_1>(fabsf(v));
            const float  d    = amax / 127.0f;
            const int8_t q    = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
            const int    sum  = warp_reduce_sum<QK8_1>((int) q);
            const int64_t ib  = r * el_w + i;
            dst_q8_1[ib].qs[lane] = q;
            if (lane == 0) {
                dst_q8_1[ib].ds = make_half2(d, d * sum);
            }
        }
    }
    if constexpr (!write_q8_1) {
        GGML_UNUSED(dst_q8_1);
    }
}

void ggml_cuda_op_fwht_sigmoid_mul(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
        const ggml_tensor * gate, ggml_tensor * dst, int n_fwht, void * dst_q8_1) {
    const int64_t rows = ggml_nelements(dst) / n_fwht;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    constexpr int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    const ggml_cuda_kernel_launch_params launch_params(dim3(num_blocks, 1, 1), dim3(warp_size, rows_per_block, 1), 0, ctx.stream());
    const float scale = 1.0f / sqrtf((float) n_fwht);
    block_q8_1 * q8_d = (block_q8_1 *) dst_q8_1;

    if (n_fwht == 64) {
        if (q8_d) {
            ggml_cuda_kernel_launch(fwht_sigmoid_mul_f32<64, true>, launch_params,
                (const float *) src->data, (const float *) gate->data, (float *) dst->data, q8_d, rows, scale);
        } else {
            ggml_cuda_kernel_launch(fwht_sigmoid_mul_f32<64, false>, launch_params,
                (const float *) src->data, (const float *) gate->data, (float *) dst->data, q8_d, rows, scale);
        }
    } else {
        if (q8_d) {
            ggml_cuda_kernel_launch(fwht_sigmoid_mul_f32<128, true>, launch_params,
                (const float *) src->data, (const float *) gate->data, (float *) dst->data, q8_d, rows, scale);
        } else {
            ggml_cuda_kernel_launch(fwht_sigmoid_mul_f32<128, false>, launch_params,
                (const float *) src->data, (const float *) gate->data, (float *) dst->data, q8_d, rows, scale);
        }
    }
}
