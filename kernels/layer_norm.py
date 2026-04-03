import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr, 
    stride, n_cols, eps, 
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row of the matrix
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride
    Y_ptr += row_idx * stride

    # Load a row into SRAM (the fast "inner circle")
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(X_ptr + cols, mask=mask, other=0.0)

    # --- THE COOL PART: Fused Math ---
    # We calculate Mean and Variance without leaving the chip
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    rstd = 1 / tl.sqrt(var + eps) # Reciprocal Standard Deviation

    # Load weights and bias (Gamma and Beta)
    w = tl.load(W_ptr + cols, mask=mask)
    b = tl.load(B_ptr + cols, mask=mask)

    # Final normalization + Scaling + Shifting
    y = x_centered * rstd * w + b

    # Write back to DRAM once
    tl.store(Y_ptr + cols, y, mask=mask)

def layernorm(x, w, b, eps=1e-5):
    M, N = x.shape
    y = torch.empty_like(x)
    # Triton handles the "Warp Specialization" behind the scenes by 
    # choosing the best num_warps based on N.
    grid = (M,)
    layernorm_kernel[grid](
        x, y, w, b, 
        x.stride(0), N, eps, 
        BLOCK_SIZE=triton.next_power_of_2(N)
    )
    return y