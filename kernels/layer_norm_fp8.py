import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_quant_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, Scale_ptr,
    stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride
    Y_ptr += row_idx * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # 1. Standard LayerNorm math
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1 / tl.sqrt(var + eps)
    
    w = tl.load(W_ptr + cols, mask=mask).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask).to(tl.float32)
    y_fp32 = x_centered * rstd * w + b

    # 2. SYMMETRIC QUANTIZATION (The H100 Flex)
    # Find the max absolute value in this row to calculate a per-row scale
    # Professional note: You can also use a global scale provided in Scale_ptr
    max_val = tl.max(tl.abs(y_fp32), axis=0)
    scale = max_val / 448.0  # 448 is the max value for FP8-E4M3
    
    y_fp8 = (y_fp32 / scale).to(tl.float8e4m3fn)

    # 3. Store the FP8 data and the scale factor
    tl.store(Y_ptr + cols, y_fp8, mask=mask)
    if tl.program_id(0) == 0: # Store the scale once for the whole tensor or per row
        tl.store(Scale_ptr + row_idx, scale)

def layernorm_fp8(x, w, b):
    M, N = x.shape
    # Output is now int8/fp8 (half the size of FP16!)
    y = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((M,), device=x.device, dtype=torch.float32)
    
    grid = (M,)
    layernorm_quant_kernel[grid](
        x, y, w, b, scales,
        x.stride(0), N, 1e-5,
        BLOCK_SIZE=triton.next_power_of_2(N)
    )
    return y, scales