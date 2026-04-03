import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # The ReLU math: if x > 0 return x, else return 0
    output = tl.where(x > 0, x, 0.0)

    # Store data
    tl.store(out_ptr + offsets, output, mask=mask)

def relu(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

# Verification
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(4096, device=device)
    z_triton = relu(x)
    z_torch = torch.relu(x)
    
    if torch.allclose(z_triton, z_torch):
        print("✅ ReLU Logic Verified!")