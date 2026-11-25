#include "situ.cuh"
#include "ggml-common.h"

static __device__ __forceinline__ float situ_eval(
        const float x, const float y,
        const float sa0, const float ba0, const float sa1, const float ba1,
        const float sb0, const float bb0, const float sb1, const float bb1) {
    // SCALE is an FMA (scale*x + bias), TANH tanhf, SIGMOID 1/(1 + exp(-x)), as in their own kernels
    const float a = __fmaf_rn(sa1, tanhf(__fmaf_rn(sa0, x, ba0)), ba1);
    const float b = __fmaf_rn(sb1, tanhf(__fmaf_rn(sb0, y, bb0)), bb1);
    const float sig = 1.0f / (1.0f + expf(-x));
    return __fmul_rn(__fmul_rn(sig, a), b);
}

static __global__ void situ_f32_vec4(const float4 * __restrict__ g, const float4 * __restrict__ u, float4 * __restrict__ dst,
        const int64_t n4, const int64_t nrows, const int64_t s4,
        const float sa0, const float ba0, const float sa1, const float ba1,
        const float sb0, const float bb0, const float sb1, const float bb1) {
    ggml_cuda_pdl_lc();
    const int64_t col = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (col >= n4) {
        return;
    }

    ggml_cuda_pdl_sync();
    for (int64_t row = blockIdx.y; row < nrows; row += gridDim.y) {
        const float4 xv = g[row*s4 + col];
        const float4 yv = u[row*s4 + col];
        float4 out;
        out.x = situ_eval(xv.x, yv.x, sa0, ba0, sa1, ba1, sb0, bb0, sb1, bb1);
        out.y = situ_eval(xv.y, yv.y, sa0, ba0, sa1, ba1, sb0, bb0, sb1, bb1);
        out.z = situ_eval(xv.z, yv.z, sa0, ba0, sa1, ba1, sb0, bb0, sb1, bb1);
        out.w = situ_eval(xv.w, yv.w, sa0, ba0, sa1, ba1, sb0, bb0, sb1, bb1);
        dst[row*n4 + col] = out;
    }
}

static __global__ void situ_f32(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ dst,
        const int64_t k, const int64_t n, const int64_t s, const float sa0, const float ba0, const float sa1, const float ba1,
        const float sb0, const float bb0, const float sb1, const float bb1) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int64_t j = (i / n)*s + (i % n);

    ggml_cuda_pdl_sync();
    dst[i] = situ_eval(g[j], u[j], sa0, ba0, sa1, ba1, sb0, bb0, sb1, bb1);
}

// Fused SiTU + Q8_1 side-output for small decode batches, bit-identical to situ_f32 + quantize_q8_1.
static __global__ void situ_f32_q8_1(
        const float * __restrict__ g, const float * __restrict__ u,
        float * __restrict__ dst, block_q8_1 * __restrict__ dst_q8_1,
        const int64_t n, const int64_t n_padded, const int64_t s,
        const float sa0, const float ba0, const float sa1, const float ba1,
        const float sb0, const float bb0, const float sb1, const float bb1) {
    ggml_cuda_pdl_lc();
    const int64_t i0 = (int64_t) blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= n_padded) {
        return;
    }
    const int64_t row = blockIdx.y;

    ggml_cuda_pdl_sync();
    float v = 0.0f;
    if (i0 < n) {
        v = situ_eval(g[row * s + i0], u[row * s + i0], sa0, ba0, sa1, ba1, sb0, bb0, sb1, bb1);
        dst[row * n + i0] = v;
    }

    const float  amax = warp_reduce_max<QK8_1>(fabsf(v));
    const float  d    = amax / 127.0f;
    const int8_t q    = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
    const int    sum  = warp_reduce_sum<QK8_1>((int) q);

    const int64_t ib  = (row * n_padded + i0) / QK8_1;
    const int     iqs = (int) (i0 % QK8_1);
    dst_q8_1[ib].qs[iqs] = q;
    if (iqs == 0) {
        dst_q8_1[ib].ds = make_half2(d, d * sum);
    }
}

void ggml_cuda_op_situ_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u,
        const float * scales, ggml_tensor * dst, void * dst_q8_1, int64_t ncols_padded) {
    const int64_t n = g->ne[0];
    const int64_t k = ggml_nelements(dst);
    const int64_t s = g->nb[1] / sizeof(float);

    const int block_size = 256;
    if (dst_q8_1 && (n % QK8_1) == 0 && (ncols_padded % QK8_1) == 0) {
        const int64_t nrows = k / n;
        const dim3 grid((ncols_padded + block_size - 1) / block_size, (unsigned int) nrows, 1);
        const ggml_cuda_kernel_launch_params launch_params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(situ_f32_q8_1, launch_params,
            (const float *) g->data, (const float *) u->data, (float *) dst->data, (block_q8_1 *) dst_q8_1,
            n, ncols_padded, s, scales[0], scales[1], scales[2], scales[3], scales[4], scales[5], scales[6], scales[7]);
        return;
    }
    if (n % 4 == 0 && s % 4 == 0 && (uintptr_t) g->data % 16 == 0 &&
        (uintptr_t) u->data % 16 == 0 && (uintptr_t) dst->data % 16 == 0) {
        const int64_t n4    = n / 4;
        const int64_t nrows = k / n;
        const dim3 grid((n4 + block_size - 1)/block_size, (unsigned int) std::min<int64_t>(nrows, 65535), 1);
        const ggml_cuda_kernel_launch_params launch_params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(situ_f32_vec4, launch_params,
            (const float4 *) g->data, (const float4 *) u->data, (float4 *) dst->data,
            n4, nrows, s / 4, scales[0], scales[1], scales[2], scales[3], scales[4], scales[5], scales[6], scales[7]);
        return;
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
        dim3((k + block_size - 1)/block_size, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(situ_f32, launch_params, (const float *) g->data, (const float *) u->data, (float *) dst->data,
        k, n, s, scales[0], scales[1], scales[2], scales[3], scales[4], scales[5], scales[6], scales[7]);
}
