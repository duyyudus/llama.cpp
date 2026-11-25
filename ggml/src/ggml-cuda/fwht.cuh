#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// Fused RMS_NORM(128) + FWHT(64 or 128) + Q8_0 set_rows scatter.
void ggml_cuda_op_rms_norm_fwht_set_rows_q8_0(ggml_backend_cuda_context & ctx, const ggml_tensor * x,
        float eps, int n_fwht, const ggml_tensor * idxs, ggml_tensor * dst);

// Fused FWHT(64 or 128) + sigmoid(gate) * cur, with optional Q8_1 side-output when row is unpadded.
void ggml_cuda_op_fwht_sigmoid_mul(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
        const ggml_tensor * gate, ggml_tensor * dst, int n_fwht, void * dst_q8_1 = nullptr);
