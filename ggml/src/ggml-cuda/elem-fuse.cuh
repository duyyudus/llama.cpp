#pragma once

#include "common.cuh"

// dst may be written over src by a kernel that reads each element before writing the same element:
// no overlap, or the same contiguous layout at the same address
bool ggml_cuda_elem_overlap_ok(const ggml_tensor * dst, const ggml_tensor * src);

// y = c - tanh(a)*b or y = c + tanh(a)*b: UNARY(TANH), MUL, SUB/ADD
struct ggml_cuda_tanh_mul_addsub_match {
    const ggml_tensor * a   = nullptr;
    const ggml_tensor * b   = nullptr;
    const ggml_tensor * c   = nullptr;
    ggml_tensor       * dst = nullptr;
    bool                sub = false;
};

// returns the number of nodes of the chain starting at node i, 0 if there is none
int ggml_cuda_match_tanh_mul_addsub(const ggml_cgraph * cgraph, int i, ggml_cuda_tanh_mul_addsub_match & match);

void ggml_cuda_op_tanh_mul_addsub(ggml_backend_cuda_context & ctx, const ggml_cuda_tanh_mul_addsub_match & match);

// y = (cur - tanh(wf*gate)*cur) + tanh(wg*gate)*msg for small T (<= 3)
struct ggml_cuda_ctrl_mmvf_gate_match {
    const ggml_tensor * wf   = nullptr;
    const ggml_tensor * wg   = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * cur  = nullptr;
    const ggml_tensor * msg  = nullptr;
    ggml_tensor       * dst  = nullptr;
};

int ggml_cuda_match_ctrl_mmvf_gate(const ggml_cgraph * cgraph, int i, ggml_cuda_ctrl_mmvf_gate_match & match);

void ggml_cuda_op_ctrl_mmvf_gate(ggml_backend_cuda_context & ctx, const ggml_cuda_ctrl_mmvf_gate_match & match);

// out = rms_norm((cur - tanh(af)*cur) + tanh(ag)*msg) * weight
struct ggml_cuda_ctrl_gate_rms_norm_mul_match {
    const ggml_tensor * af  = nullptr;
    const ggml_tensor * ag  = nullptr;
    const ggml_tensor * cur = nullptr;
    const ggml_tensor * msg = nullptr;
    const ggml_tensor * w   = nullptr;
    ggml_tensor       * dst = nullptr;
    float               eps = 0.0f;
};

int ggml_cuda_match_ctrl_gate_rms_norm_mul(const ggml_cgraph * cgraph, int i, ggml_cuda_ctrl_gate_rms_norm_mul_match & match);

void ggml_cuda_op_ctrl_gate_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_cuda_ctrl_gate_rms_norm_mul_match & match);

// dst = W * concat(a, b, dim=0) for small T (<= 3)
struct ggml_cuda_concat_mmvf_match {
    const ggml_tensor * w   = nullptr;
    const ggml_tensor * a   = nullptr;
    const ggml_tensor * b   = nullptr;
    ggml_tensor       * dst = nullptr;
};

int ggml_cuda_match_concat_mmvf(const ggml_cgraph * cgraph, int i, ggml_cuda_concat_mmvf_match & match);

void ggml_cuda_op_concat_mmvf(ggml_backend_cuda_context & ctx, const ggml_cuda_concat_mmvf_match & match);

// normalized router weights: SUM_ROWS, SCALE or CLAMP, DIV[, RESHAPE, SCALE]
struct ggml_cuda_row_norm_div_match {
    const ggml_tensor * x   = nullptr;
    ggml_tensor       * dst = nullptr;
    bool  clamp       = false;
    float p0          = 0.0f; // SCALE: scale, CLAMP: min
    float p1          = 0.0f; // SCALE: bias,  CLAMP: max
    bool  has_scale   = false;
    float out_scale   = 1.0f;
    float out_bias    = 0.0f;
};

int ggml_cuda_match_row_norm_div(const ggml_cgraph * cgraph, int i, ggml_cuda_row_norm_div_match & match);

void ggml_cuda_op_row_norm_div(ggml_backend_cuda_context & ctx, const ggml_cuda_row_norm_div_match & match);

// F32 matmul for small K (K <= 64, K % 4 == 0) and small N (4 <= N <= 32), avoids cuBLAS split-K overhead
bool ggml_cuda_try_mul_mat_f32_small_k(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
