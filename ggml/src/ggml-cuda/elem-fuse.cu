#include "elem-fuse.cuh"
#include "fwht.cuh"
#include "reduce_rows.cuh"

#include <cstring>

// ---------------------------------------------------------------------------------------------------------------------
// y = c -/+ tanh(a)*b
// each value is rounded as in the separate TANH (tanhf), MUL and SUB/ADD kernels: the product and the sum are not
// contracted into an FMA

template <bool sub>
static __global__ void tanh_mul_addsub_f32(
        const char * a_ptr, const char * b_ptr, const char * c_ptr, float * dst_ptr,
        const int ne0, const int ne1, const int ne2,
        const int64_t a1, const int64_t a2, const int64_t a3,
        const int64_t b1, const int64_t b2, const int64_t b3,
        const int64_t c1, const int64_t c2, const int64_t c3) {
    const char * GGML_CUDA_RESTRICT a   = a_ptr;
    const char * GGML_CUDA_RESTRICT b   = b_ptr;
    const char * GGML_CUDA_RESTRICT c   = c_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;
    ggml_cuda_pdl_lc();
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z % ne2;
    const int i3 = blockIdx.z / ne2;

    ggml_cuda_pdl_sync();
    if (i0 >= ne0) {
        return;
    }

    const float av = *(const float *) (a + i1*a1 + i2*a2 + i3*a3 + i0*sizeof(float));
    const float bv = *(const float *) (b + i1*b1 + i2*b2 + i3*b3 + i0*sizeof(float));
    const float cv = *(const float *) (c + i1*c1 + i2*c2 + i3*c3 + i0*sizeof(float));

    const float m = __fmul_rn(tanhf(av), bv);

    dst[((int64_t(i3)*ne2 + i2)*ne1 + i1)*ne0 + i0] = sub ? __fsub_rn(cv, m) : __fadd_rn(cv, m);
}

bool ggml_cuda_elem_overlap_ok(const ggml_tensor * dst, const ggml_tensor * src) {
    const char * d0 = (const char *) dst->data;
    const char * d1 = d0 + ggml_nbytes(dst);
    const char * s0 = (const char *) src->data;
    const char * s1 = s0 + ggml_nbytes(src);
    if (d1 <= s0 || s1 <= d0) {
        return true;
    }
    return d0 == s0 && ggml_is_contiguous(dst) && ggml_is_contiguous(src) && ggml_nelements(dst) == ggml_nelements(src) &&
        dst->type == src->type;
}

static bool elem_is_view_op(const ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool elem_f32_rows_ok(const ggml_tensor * t, const ggml_tensor * shape) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && ggml_are_same_shape(t, shape);
}

int ggml_cuda_match_tanh_mul_addsub(const ggml_cgraph * cgraph, int i, ggml_cuda_tanh_mul_addsub_match & match) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    // the views of b (no kernels) may sit between TANH and MUL
    int j = i + 1;
    while (j < i + 3 && j < cgraph->n_nodes && elem_is_view_op(cgraph->nodes[j]->op)) {
        ++j;
    }
    if (j + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * t = cgraph->nodes[i];
    const ggml_tensor * m = cgraph->nodes[j];
    ggml_tensor       * y = cgraph->nodes[j + 1];
    if (t->op != GGML_OP_UNARY || ggml_get_unary_op(t) != GGML_UNARY_OP_TANH || m->op != GGML_OP_MUL ||
        (y->op != GGML_OP_SUB && y->op != GGML_OP_ADD)) {
        return 0;
    }

    const ggml_tensor * b = nullptr;
    if (m->src[0] == t) {
        b = m->src[1];
    } else if (m->src[1] == t) {
        b = m->src[0];
    } else {
        return 0;
    }

    const ggml_tensor * c = nullptr;
    bool sub = false;
    if (y->op == GGML_OP_SUB) {
        if (y->src[1] != m || y->src[0] == m) {
            return 0;
        }
        c   = y->src[0];
        sub = true;
    } else {
        if (y->src[0] == m && y->src[1] != m) {
            c = y->src[1];
        } else if (y->src[1] == m && y->src[0] != m) {
            c = y->src[0];
        } else {
            return 0;
        }
    }

    const ggml_tensor * a = t->src[0];
    if (!elem_f32_rows_ok(a, y) || !elem_f32_rows_ok(b, y) || !elem_f32_rows_ok(c, y) || !elem_f32_rows_ok(t, y) ||
        !elem_f32_rows_ok(m, y) || !ggml_is_contiguous(y) || !ggml_is_contiguous(t) || b == t || c == t) {
        return 0;
    }
    if (y->ne[1] > 65535 || y->ne[2]*y->ne[3] > 65535 || y->ne[0] > INT_MAX) {
        return 0;
    }

    // the skipped views must not be views of an elided node
    for (int k = i + 1; k < j; ++k) {
        if (cgraph->nodes[k]->view_src == t || cgraph->nodes[k]->view_src == m) {
            return 0;
        }
    }

    const ggml_op ops[3]  = { GGML_OP_UNARY, GGML_OP_MUL, y->op };
    const int     idxs[3] = { i, j, j + 1 };
    const int     outs[1] = { j + 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 3, ops, outs, 1)) {
        return 0;
    }

    if (!ggml_cuda_elem_overlap_ok(y, a) || !ggml_cuda_elem_overlap_ok(y, b) || !ggml_cuda_elem_overlap_ok(y, c)) {
        return 0;
    }

    match.a   = a;
    match.b   = b;
    match.c   = c;
    match.dst = y;
    match.sub = sub;
    return j + 2 - i;
}

void ggml_cuda_op_tanh_mul_addsub(ggml_backend_cuda_context & ctx, const ggml_cuda_tanh_mul_addsub_match & match) {
    const ggml_tensor * a = match.a;
    const ggml_tensor * b = match.b;
    const ggml_tensor * c = match.c;
    ggml_tensor       * y = match.dst;

    const int block = 256;
    const dim3 grid((y->ne[0] + block - 1)/block, y->ne[1], y->ne[2]*y->ne[3]);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, dim3(block, 1, 1), 0, ctx.stream());

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            (const char *) a->data, (const char *) b->data, (const char *) c->data, (float *) y->data,
            (int) y->ne[0], (int) y->ne[1], (int) y->ne[2],
            (int64_t) a->nb[1], (int64_t) a->nb[2], (int64_t) a->nb[3],
            (int64_t) b->nb[1], (int64_t) b->nb[2], (int64_t) b->nb[3],
            (int64_t) c->nb[1], (int64_t) c->nb[2], (int64_t) c->nb[3]);
    };
    if (match.sub) {
        launch(tanh_mul_addsub_f32<true>);
    } else {
        launch(tanh_mul_addsub_f32<false>);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// fused control gate: y = (cur - tanh(wf*gate)*cur) + tanh(wg*gate)*msg for T <= 3

template <int T>
static __global__ void ctrl_mmvf_gate_f32_vec4(
        const char * wf_ptr, const char * wg_ptr, const char * gate_ptr,
        const char * cur_ptr, const char * msg_ptr, float * dst_ptr,
        const int K4, const int M,
        const int64_t wf_s1, const int64_t wg_s1, const int64_t gate_s1,
        const int64_t cur_s1, const int64_t msg_s1) {
    ggml_cuda_pdl_lc();
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (row >= M) {
        return;
    }

    const float4 * wf4 = (const float4 *) (wf_ptr + int64_t(row)*wf_s1);
    const float4 * wg4 = (const float4 *) (wg_ptr + int64_t(row)*wg_s1);

    float sf[T] = {0.0f};
    float sg[T] = {0.0f};
    for (int c = lane; c < K4; c += 16) {
        const float4 xf = wf4[c];
        const float4 xg = wg4[c];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 y = ((const float4 *) (gate_ptr + int64_t(t)*gate_s1))[c];
            sf[t] += xf.x*y.x + xf.y*y.y + xf.z*y.z + xf.w*y.w;
            sg[t] += xg.x*y.x + xg.y*y.y + xg.z*y.z + xg.w*y.w;
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int mask = 8; mask >= 1; mask >>= 1) {
            sf[t] += ggml_cuda_shfl_xor(sf[t], mask, 16);
            sg[t] += ggml_cuda_shfl_xor(sg[t], mask, 16);
        }
        if (lane == 0) {
            const float cv  = *(const float *) (cur_ptr + int64_t(t)*cur_s1 + int64_t(row)*sizeof(float));
            const float mv  = *(const float *) (msg_ptr + int64_t(t)*msg_s1 + int64_t(row)*sizeof(float));
            const float sub = __fsub_rn(cv, __fmul_rn(tanhf(sf[t]), cv));
            dst_ptr[int64_t(t)*M + row] = __fadd_rn(sub, __fmul_rn(tanhf(sg[t]), mv));
        }
    }
}

template <int T>
static __global__ void ctrl_mmvf_gate_f32(
        const char * wf_ptr, const char * wg_ptr, const char * gate_ptr,
        const char * cur_ptr, const char * msg_ptr, float * dst_ptr,
        const int K2, const int M,
        const int64_t wf_s1, const int64_t wg_s1, const int64_t gate_s1,
        const int64_t cur_s1, const int64_t msg_s1) {
    ggml_cuda_pdl_lc();
    const int row  = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (row >= M) {
        return;
    }

    const float2 * wf2 = (const float2 *) (wf_ptr + int64_t(row)*wf_s1);
    const float2 * wg2 = (const float2 *) (wg_ptr + int64_t(row)*wg_s1);

    float sf[T] = {0.0f};
    float sg[T] = {0.0f};
    for (int c = lane; c < K2; c += WARP_SIZE) {
        const float2 xf = wf2[c];
        const float2 xg = wg2[c];
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float2 y = ((const float2 *) (gate_ptr + int64_t(t)*gate_s1))[c];
            sf[t] += xf.x*y.x + xf.y*y.y;
            sg[t] += xg.x*y.x + xg.y*y.y;
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
        sf[t] = warp_reduce_sum(sf[t]);
        sg[t] = warp_reduce_sum(sg[t]);
        if (lane == 0) {
            const float cv  = *(const float *) (cur_ptr + int64_t(t)*cur_s1 + int64_t(row)*sizeof(float));
            const float mv  = *(const float *) (msg_ptr + int64_t(t)*msg_s1 + int64_t(row)*sizeof(float));
            const float sub = __fsub_rn(cv, __fmul_rn(tanhf(sf[t]), cv));
            dst_ptr[int64_t(t)*M + row] = __fadd_rn(sub, __fmul_rn(tanhf(sg[t]), mv));
        }
    }
}

static bool elem_disjoint_2d(const ggml_tensor * dst, const ggml_tensor * src) {
    const char * d0 = (const char *) dst->data;
    const char * d1 = d0 + ggml_nbytes(dst);
    const char * s0 = (const char *) src->data;
    const size_t s_span = src->ne[1] > 1
        ? (size_t) (src->ne[1] - 1)*src->nb[1] + (size_t) src->ne[0]*ggml_element_size(src)
        : ggml_nbytes(src);
    const char * s1 = s0 + s_span;
    return d1 <= s0 || s1 <= d0;
}

int ggml_cuda_match_ctrl_mmvf_gate(const ggml_cgraph * cgraph, int i, ggml_cuda_ctrl_mmvf_gate_match & match) {
    if (i + 7 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mm_f = cgraph->nodes[i];
    const ggml_tensor * mm_g = cgraph->nodes[i + 1];
    const ggml_tensor * t_f  = cgraph->nodes[i + 2];
    const ggml_tensor * m_f  = cgraph->nodes[i + 3];
    const ggml_tensor * s_f  = cgraph->nodes[i + 4];
    const ggml_tensor * t_g  = cgraph->nodes[i + 5];
    if (mm_f->op != GGML_OP_MUL_MAT || mm_g->op != GGML_OP_MUL_MAT ||
        t_f->op != GGML_OP_UNARY || ggml_get_unary_op(t_f) != GGML_UNARY_OP_TANH ||
        m_f->op != GGML_OP_MUL || s_f->op != GGML_OP_SUB ||
        t_g->op != GGML_OP_UNARY || ggml_get_unary_op(t_g) != GGML_UNARY_OP_TANH) {
        return 0;
    }

    int k = i + 6;
    while (k < i + 8 && k < cgraph->n_nodes && elem_is_view_op(cgraph->nodes[k]->op)) {
        ++k;
    }
    if (k + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * m_g = cgraph->nodes[k];
    ggml_tensor       * a_g = cgraph->nodes[k + 1];
    if (m_g->op != GGML_OP_MUL || a_g->op != GGML_OP_ADD) {
        return 0;
    }

    if (t_f->src[0] != mm_f || t_g->src[0] != mm_g || mm_f->src[1] != mm_g->src[1]) {
        return 0;
    }
    const ggml_tensor * cur = m_f->src[0] == t_f ? m_f->src[1] : m_f->src[1] == t_f ? m_f->src[0] : nullptr;
    const ggml_tensor * msg = m_g->src[0] == t_g ? m_g->src[1] : m_g->src[1] == t_g ? m_g->src[0] : nullptr;
    if (!cur || !msg || s_f->src[0] != cur || s_f->src[1] != m_f ||
        !((a_g->src[0] == s_f && a_g->src[1] == m_g) || (a_g->src[1] == s_f && a_g->src[0] == m_g))) {
        return 0;
    }

    const ggml_tensor * wf   = mm_f->src[0];
    const ggml_tensor * wg   = mm_g->src[0];
    const ggml_tensor * gate = mm_f->src[1];
    if (wf->type != GGML_TYPE_F32 || wg->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 ||
        wf->ne[2] != 1 || wf->ne[3] != 1 || gate->ne[2] != 1 || gate->ne[3] != 1 ||
        !ggml_are_same_shape(wf, wg) || wf->ne[0] % 2 != 0 || wf->ne[0] > 256 ||
        gate->ne[1] < 1 || gate->ne[1] > 3 || wf->ne[1] > INT_MAX ||
        wf->nb[0] != sizeof(float) || wg->nb[0] != sizeof(float) || gate->nb[0] != sizeof(float) ||
        (uintptr_t) wf->data % 8 != 0 || wf->nb[1] % 8 != 0 ||
        (uintptr_t) wg->data % 8 != 0 || wg->nb[1] % 8 != 0 ||
        (uintptr_t) gate->data % 8 != 0 || gate->nb[1] % 8 != 0 ||
        ggml_cuda_op_mul_mat_use_fwht(mm_f) || ggml_cuda_op_mul_mat_use_fwht(mm_g)) {
        return 0;
    }
    if (!elem_f32_rows_ok(mm_f, a_g) || !elem_f32_rows_ok(mm_g, a_g) ||
        !elem_f32_rows_ok(cur, a_g) || !elem_f32_rows_ok(msg, a_g) ||
        !ggml_is_contiguous(a_g) || a_g->ne[2] != 1 || a_g->ne[3] != 1) {
        return 0;
    }

    for (int v = i + 6; v < k; ++v) {
        const ggml_tensor * vs = cgraph->nodes[v]->view_src;
        if (vs == mm_f || vs == mm_g || vs == t_f || vs == m_f || vs == s_f || vs == t_g) {
            return 0;
        }
    }

    const ggml_op ops[8]  = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
    const int     idxs[8] = { i, i + 1, i + 2, i + 3, i + 4, i + 5, k, k + 1 };
    const int     outs[1] = { k + 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 8, ops, outs, 1)) {
        return 0;
    }

    if (!elem_disjoint_2d(a_g, wf) || !elem_disjoint_2d(a_g, wg) || !elem_disjoint_2d(a_g, gate) ||
        !(elem_disjoint_2d(a_g, cur) || ggml_cuda_elem_overlap_ok(a_g, cur)) ||
        !(elem_disjoint_2d(a_g, msg) || ggml_cuda_elem_overlap_ok(a_g, msg))) {
        return 0;
    }

    match.wf   = wf;
    match.wg   = wg;
    match.gate = gate;
    match.cur  = cur;
    match.msg  = msg;
    match.dst  = a_g;
    return k + 2 - i;
}

void ggml_cuda_op_ctrl_mmvf_gate(ggml_backend_cuda_context & ctx, const ggml_cuda_ctrl_mmvf_gate_match & match) {
    const int K = (int) match.wf->ne[0];
    const int M = (int) match.wf->ne[1];
    const int T = (int) match.gate->ne[1];

    const bool use_vec4 = K % 4 == 0 &&
        (uintptr_t) match.wf->data % 16 == 0 && match.wf->nb[1] % 16 == 0 &&
        (uintptr_t) match.wg->data % 16 == 0 && match.wg->nb[1] % 16 == 0 &&
        (uintptr_t) match.gate->data % 16 == 0 && match.gate->nb[1] % 16 == 0;

    if (use_vec4) {
        constexpr int rows_per_block = 16;
        const dim3 grid((M + rows_per_block - 1) / rows_per_block, 1, 1);
        const dim3 block(16, rows_per_block, 1);
        const ggml_cuda_kernel_launch_params launch_params(grid, block, 0, ctx.stream());
        auto launch4 = [&](auto kernel) {
            ggml_cuda_kernel_launch(kernel, launch_params,
                (const char *) match.wf->data, (const char *) match.wg->data, (const char *) match.gate->data,
                (const char *) match.cur->data, (const char *) match.msg->data, (float *) match.dst->data,
                K / 4, M,
                (int64_t) match.wf->nb[1], (int64_t) match.wg->nb[1], (int64_t) match.gate->nb[1],
                (int64_t) match.cur->nb[1], (int64_t) match.msg->nb[1]);
        };
        switch (T) {
            case 1: launch4(ctrl_mmvf_gate_f32_vec4<1>); break;
            case 2: launch4(ctrl_mmvf_gate_f32_vec4<2>); break;
            case 3: launch4(ctrl_mmvf_gate_f32_vec4<3>); break;
            default: GGML_ABORT("unexpected batch size");
        }
        return;
    }

    constexpr int rows_per_block = 8;
    const dim3 grid((M + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block(WARP_SIZE, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params(grid, block, 0, ctx.stream());

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            (const char *) match.wf->data, (const char *) match.wg->data, (const char *) match.gate->data,
            (const char *) match.cur->data, (const char *) match.msg->data, (float *) match.dst->data,
            K / 2, M,
            (int64_t) match.wf->nb[1], (int64_t) match.wg->nb[1], (int64_t) match.gate->nb[1],
            (int64_t) match.cur->nb[1], (int64_t) match.msg->nb[1]);
    };
    switch (T) {
        case 1: launch(ctrl_mmvf_gate_f32<1>); break;
        case 2: launch(ctrl_mmvf_gate_f32<2>); break;
        case 3: launch(ctrl_mmvf_gate_f32<3>); break;
        default: GGML_ABORT("unexpected batch size");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// fused control gate + rms_norm + weight mul: out = rms_norm((cur - tanh(af)*cur) + tanh(ag)*msg) * w

template <int block_size>
static __global__ void ctrl_gate_rms_norm_mul_f32(
        const float * af_ptr, const float * ag_ptr, const float * cur_ptr,
        const char * msg_ptr, const float * w_ptr, float * dst_ptr,
        const int ncols, const int64_t msg_s1, const float eps) {
    ggml_cuda_pdl_lc();
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    const int ncols4 = ncols / 4;
    const float4 * af4  = (const float4 *) (af_ptr  + row*ncols);
    const float4 * ag4  = (const float4 *) (ag_ptr  + row*ncols);
    const float4 * cur4 = (const float4 *) (cur_ptr + row*ncols);
    const float4 * msg4 = (const float4 *) (msg_ptr + row*msg_s1);
    const float4 * w4   = (const float4 *) w_ptr;
    float4       * dst4 = (float4 *)       (dst_ptr + row*ncols);

    float4 y   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float  tmp = 0.0f;
    if (tid < ncols4) {
        const float4 f = af4[tid];
        const float4 g = ag4[tid];
        const float4 c = cur4[tid];
        const float4 m = msg4[tid];
        y.x = __fadd_rn(__fsub_rn(c.x, __fmul_rn(tanhf(f.x), c.x)), __fmul_rn(tanhf(g.x), m.x));
        y.y = __fadd_rn(__fsub_rn(c.y, __fmul_rn(tanhf(f.y), c.y)), __fmul_rn(tanhf(g.y), m.y));
        y.z = __fadd_rn(__fsub_rn(c.z, __fmul_rn(tanhf(f.z), c.z)), __fmul_rn(tanhf(g.z), m.z));
        y.w = __fadd_rn(__fsub_rn(c.w, __fmul_rn(tanhf(f.w), c.w)), __fmul_rn(tanhf(g.w), m.w));
        tmp = y.x*y.x + y.y*y.y + y.z*y.z + y.w*y.w;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(__fadd_rn(tmp / ncols, eps));

    if (tid < ncols4) {
        const float4 wv = w4[tid];
        dst4[tid] = make_float4(scale*y.x*wv.x, scale*y.y*wv.y, scale*y.z*wv.z, scale*y.w*wv.w);
    }
}

int ggml_cuda_match_ctrl_gate_rms_norm_mul(const ggml_cgraph * cgraph, int i, ggml_cuda_ctrl_gate_rms_norm_mul_match & match) {
    if (i + 7 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * t_f = cgraph->nodes[i];
    const ggml_tensor * m_f = cgraph->nodes[i + 1];
    const ggml_tensor * s_f = cgraph->nodes[i + 2];
    const ggml_tensor * t_g = cgraph->nodes[i + 3];
    if (t_f->op != GGML_OP_UNARY || ggml_get_unary_op(t_f) != GGML_UNARY_OP_TANH ||
        m_f->op != GGML_OP_MUL || s_f->op != GGML_OP_SUB ||
        t_g->op != GGML_OP_UNARY || ggml_get_unary_op(t_g) != GGML_UNARY_OP_TANH) {
        return 0;
    }

    int k = i + 4;
    while (k < i + 6 && k < cgraph->n_nodes && elem_is_view_op(cgraph->nodes[k]->op)) {
        ++k;
    }
    if (k + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * m_g = cgraph->nodes[k];
    const ggml_tensor * add = cgraph->nodes[k + 1];
    const ggml_tensor * rms = cgraph->nodes[k + 2];
    ggml_tensor       * out = cgraph->nodes[k + 3];
    if (m_g->op != GGML_OP_MUL || add->op != GGML_OP_ADD || rms->op != GGML_OP_RMS_NORM || out->op != GGML_OP_MUL) {
        return 0;
    }

    const ggml_tensor * cur = m_f->src[0] == t_f ? m_f->src[1] : m_f->src[1] == t_f ? m_f->src[0] : nullptr;
    const ggml_tensor * msg = m_g->src[0] == t_g ? m_g->src[1] : m_g->src[1] == t_g ? m_g->src[0] : nullptr;
    const ggml_tensor * w   = out->src[0] == rms ? out->src[1] : out->src[1] == rms ? out->src[0] : nullptr;
    if (!cur || !msg || !w || s_f->src[0] != cur || s_f->src[1] != m_f || rms->src[0] != add ||
        !((add->src[0] == s_f && add->src[1] == m_g) || (add->src[1] == s_f && add->src[0] == m_g))) {
        return 0;
    }

    const ggml_tensor * af = t_f->src[0];
    const ggml_tensor * ag = t_g->src[0];
    if (!elem_f32_rows_ok(af, out) || !elem_f32_rows_ok(ag, out) || !elem_f32_rows_ok(cur, out) || !elem_f32_rows_ok(msg, out) ||
        !ggml_is_contiguous(af) || !ggml_is_contiguous(ag) || !ggml_is_contiguous(cur) || !ggml_is_contiguous(out) ||
        w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != out->ne[0] || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1 ||
        out->ne[2] != 1 || out->ne[3] != 1 || out->ne[0] <= 0 || out->ne[0] % 4 != 0 || out->ne[0]/4 > 1024 || out->ne[1] > INT_MAX ||
        (uintptr_t) af->data % 16 != 0 || (uintptr_t) ag->data % 16 != 0 || (uintptr_t) cur->data % 16 != 0 ||
        (uintptr_t) msg->data % 16 != 0 || msg->nb[1] % 16 != 0 || (uintptr_t) w->data % 16 != 0 || (uintptr_t) out->data % 16 != 0) {
        return 0;
    }

    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    if (!(eps >= 0.0f)) {
        return 0;
    }

    for (int v = i + 4; v < k; ++v) {
        const ggml_tensor * vs = cgraph->nodes[v]->view_src;
        if (vs == t_f || vs == m_f || vs == s_f || vs == t_g) {
            return 0;
        }
    }

    const ggml_op ops[8]  = { GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_RMS_NORM, GGML_OP_MUL };
    const int     idxs[8] = { i, i + 1, i + 2, i + 3, k, k + 1, k + 2, k + 3 };
    const int     outs[1] = { k + 3 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 8, ops, outs, 1)) {
        return 0;
    }

    if (!ggml_cuda_elem_overlap_ok(out, af) || !ggml_cuda_elem_overlap_ok(out, ag) ||
        !ggml_cuda_elem_overlap_ok(out, cur) || !elem_disjoint_2d(out, w) ||
        !(elem_disjoint_2d(out, msg) || ggml_cuda_elem_overlap_ok(out, msg))) {
        return 0;
    }

    match.af  = af;
    match.ag  = ag;
    match.cur = cur;
    match.msg = msg;
    match.w   = w;
    match.dst = out;
    match.eps = eps;
    return k + 4 - i;
}

void ggml_cuda_op_ctrl_gate_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_cuda_ctrl_gate_rms_norm_mul_match & match) {
    const int     ncols = (int) match.dst->ne[0];
    const int64_t nrows = match.dst->ne[1];
    const dim3    blocks_num(nrows, 1, 1);

#define CTRL_GATE_RMS_LAUNCH(BLOCK_SIZE) \
    do { \
        const dim3 block_dims(BLOCK_SIZE, 1, 1); \
        const ggml_cuda_kernel_launch_params launch_params{ blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream() }; \
        ggml_cuda_kernel_launch(ctrl_gate_rms_norm_mul_f32<BLOCK_SIZE>, launch_params, \
            (const float *) match.af->data, (const float *) match.ag->data, (const float *) match.cur->data, \
            (const char *) match.msg->data, (const float *) match.w->data, (float *) match.dst->data, \
            ncols, (int64_t) match.msg->nb[1], match.eps); \
    } while (0)

    const int n4 = ncols / 4;
    if (n4 <= 32) {
        CTRL_GATE_RMS_LAUNCH(32);
    } else if (n4 <= 128) {
        CTRL_GATE_RMS_LAUNCH(128);
    } else if (n4 <= 256) {
        CTRL_GATE_RMS_LAUNCH(256);
    } else if (n4 <= 512) {
        CTRL_GATE_RMS_LAUNCH(512);
    } else {
        CTRL_GATE_RMS_LAUNCH(1024);
    }
#undef CTRL_GATE_RMS_LAUNCH
}

// ---------------------------------------------------------------------------------------------------------------------
// fused concat(a, b, dim=0) + mul_mat(W, concat) for small T (<= 3)

template <int T, int block_size>
static __global__ void concat_mmvf_f32(
        const char * w_ptr, const char * a_ptr, const char * b_ptr, float * dst_ptr,
        const int Ka4, const int K4, const int M,
        const int64_t w_s1, const int64_t a_s1, const int64_t b_s1) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_sum[32];

    ggml_cuda_pdl_sync();

    const float4 * w4 = (const float4 *) (w_ptr + int64_t(row)*w_s1);
    float sum[T] = {0.0f};
    for (int c = tid; c < K4; c += block_size) {
        const float4 x  = w4[c];
        const bool   in_a = c < Ka4;
        const int    ca   = in_a ? c : (c - Ka4);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 y = in_a
                ? ((const float4 *) (a_ptr + int64_t(t)*a_s1))[ca]
                : ((const float4 *) (b_ptr + int64_t(t)*b_s1))[ca];
            sum[t] += x.x*y.x + x.y*y.y + x.z*y.z + x.w*y.w;
        }
    }

#pragma unroll
    for (int t = 0; t < T; ++t) {
        sum[t] = block_reduce<block_reduce_method::SUM, block_size>(sum[t], s_sum);
        if (tid == 0) {
            dst_ptr[int64_t(t)*M + row] = sum[t];
        }
        if constexpr (T > 1) {
            __syncthreads();
        }
    }
}

int ggml_cuda_match_concat_mmvf(const ggml_cgraph * cgraph, int i, ggml_cuda_concat_mmvf_match & match) {
    if (i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * cat = cgraph->nodes[i];
    ggml_tensor       * mm  = cgraph->nodes[i + 1];
    if (cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 ||
        mm->op != GGML_OP_MUL_MAT || mm->src[1] != cat) {
        return 0;
    }

    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * a = cat->src[0];
    const ggml_tensor * b = cat->src[1];
    if (w->type != GGML_TYPE_F32 || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
        w->ne[2] != 1 || w->ne[3] != 1 || cat->ne[2] != 1 || cat->ne[3] != 1 ||
        a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
        cat->ne[1] < 1 || cat->ne[1] > 3 || w->ne[1] > INT_MAX || w->ne[0] != a->ne[0] + b->ne[0] ||
        a->ne[0] % 4 != 0 || b->ne[0] % 4 != 0 ||
        w->nb[0] != sizeof(float) || a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || !ggml_is_contiguous(mm) ||
        (uintptr_t) w->data % 16 != 0 || w->nb[1] % 16 != 0 ||
        (uintptr_t) a->data % 16 != 0 || a->nb[1] % 16 != 0 ||
        (uintptr_t) b->data % 16 != 0 || b->nb[1] % 16 != 0 ||
        ggml_cuda_op_mul_mat_use_fwht(mm)) {
        return 0;
    }

    const ggml_op ops[2]  = { GGML_OP_CONCAT, GGML_OP_MUL_MAT };
    const int     outs[1] = { i + 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 2, ops, outs, 1)) {
        return 0;
    }

    if (!elem_disjoint_2d(mm, w) || !elem_disjoint_2d(mm, a) || !elem_disjoint_2d(mm, b)) {
        return 0;
    }

    match.w   = w;
    match.a   = a;
    match.b   = b;
    match.dst = mm;
    return 2;
}

void ggml_cuda_op_concat_mmvf(ggml_backend_cuda_context & ctx, const ggml_cuda_concat_mmvf_match & match) {
    const int Ka4 = (int) (match.a->ne[0] / 4);
    const int K4  = (int) (match.w->ne[0] / 4);
    const int M   = (int) match.w->ne[1];
    const int T   = (int) match.dst->ne[1];

    constexpr int block_size = 512;
    const ggml_cuda_kernel_launch_params launch_params(dim3(M, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());
    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            (const char *) match.w->data, (const char *) match.a->data, (const char *) match.b->data, (float *) match.dst->data,
            Ka4, K4, M,
            (int64_t) match.w->nb[1], (int64_t) match.a->nb[1], (int64_t) match.b->nb[1]);
    };
    switch (T) {
        case 1: launch(concat_mmvf_f32<1, block_size>); break;
        case 2: launch(concat_mmvf_f32<2, block_size>); break;
        case 3: launch(concat_mmvf_f32<3, block_size>); break;
        default: GGML_ABORT("unexpected batch size");
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// x / f(sum_rows(x)) [* scale + bias], f = SCALE (x*s + b) or CLAMP
// one block per row with the block size of sum_rows_f32_cuda, so the row sum is reduce_row_f32's for the same shape;
// SCALE as scale_f32's FMA, the division as op_div's a / b

template <bool clamp, bool has_scale>
static __global__ void row_norm_div_f32(const float * x_ptr, float * dst_ptr, const int ncols,
        const float p0, const float p1, const float out_scale, const float out_bias) {
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int64_t row = blockIdx.x;

    const float * xr = x + row*ncols;
    const float sum = reduce_row_f32(xr, ncols); // waits for the previous kernel
    ggml_cuda_pdl_lc();

    const float d = clamp ? fminf(fmaxf(sum, p0), p1) : __fmaf_rn(p0, sum, p1);

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float v = xr[col] / d;
        dst[row*ncols + col] = has_scale ? __fmaf_rn(out_scale, v, out_bias) : v;
    }
}

int ggml_cuda_match_row_norm_div(const ggml_cgraph * cgraph, int i, ggml_cuda_row_norm_div_match & match) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * sum = cgraph->nodes[i];
    const ggml_tensor * f   = cgraph->nodes[i + 1];
    const ggml_tensor * div = cgraph->nodes[i + 2];
    if (sum->op != GGML_OP_SUM_ROWS || (f->op != GGML_OP_SCALE && f->op != GGML_OP_CLAMP) || div->op != GGML_OP_DIV) {
        return 0;
    }
    const ggml_tensor * x = sum->src[0];
    if (f->src[0] != sum || div->src[0] != x || div->src[1] != f) {
        return 0;
    }
    if (x->type != GGML_TYPE_F32 || sum->type != GGML_TYPE_F32 || f->type != GGML_TYPE_F32 || div->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(div) || !ggml_are_same_shape(x, div) || x->ne[0] > INT_MAX ||
        ggml_nrows(x) > INT_MAX || !ggml_is_contiguous(sum) || !ggml_is_contiguous(f) || ggml_nelements(sum) != ggml_nrows(x)) {
        return 0;
    }

    int n = 3;
    ggml_op ops[5] = { GGML_OP_SUM_ROWS, f->op, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE };
    ggml_tensor * out = cgraph->nodes[i + 2];
    bool has_scale = false;
    if (i + 4 < cgraph->n_nodes) {
        const ggml_tensor * r = cgraph->nodes[i + 3];
        ggml_tensor       * s = cgraph->nodes[i + 4];
        if (r->op == GGML_OP_RESHAPE && r->src[0] == div && s->op == GGML_OP_SCALE && s->src[0] == r &&
            s->type == GGML_TYPE_F32 && ggml_is_contiguous(s) && ggml_nelements(s) == ggml_nelements(div)) {
            n = 5;
            out = s;
            has_scale = true;
        }
    }

    const int outs[1] = { i + n - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs, 1)) {
        if (n == 5) {
            // the plain division alone
            n = 3;
            out = cgraph->nodes[i + 2];
            has_scale = false;
            const int outs3[1] = { i + 2 };
            if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs3, 1)) {
                return 0;
            }
        } else {
            return 0;
        }
    }

    // each thread reads x only at the columns it writes, after the row sum
    if (!ggml_cuda_elem_overlap_ok(out, x)) {
        return 0;
    }

    match.x     = x;
    match.dst   = out;
    match.clamp = f->op == GGML_OP_CLAMP;
    memcpy(&match.p0, (const float *) f->op_params + 0, sizeof(float));
    memcpy(&match.p1, (const float *) f->op_params + 1, sizeof(float));
    match.has_scale = has_scale;
    if (has_scale) {
        memcpy(&match.out_scale, (const float *) out->op_params + 0, sizeof(float));
        memcpy(&match.out_bias,  (const float *) out->op_params + 1, sizeof(float));
    }
    return n;
}

void ggml_cuda_op_row_norm_div(ggml_backend_cuda_context & ctx, const ggml_cuda_row_norm_div_match & match) {
    const ggml_tensor * x = match.x;
    const int     ncols = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    // the block size of sum_rows_f32_cuda for this shape
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int block_size = (nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128);

    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, (const float *) x->data, (float *) match.dst->data, ncols,
            match.p0, match.p1, match.out_scale, match.out_bias);
    };
    if (match.clamp) {
        match.has_scale ? launch(row_norm_div_f32<true, true>) : launch(row_norm_div_f32<true, false>);
    } else {
        match.has_scale ? launch(row_norm_div_f32<false, true>) : launch(row_norm_div_f32<false, false>);
    }
}
