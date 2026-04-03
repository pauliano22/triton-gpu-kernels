import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This 'pid' is which "Block" of the vector this worker is handling
    pid = tl.program_id(axis=0)
    
    # Calculate the range of indices this block handles
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask so we don't read past the end of the array
    mask = offsets < n_elements

    # Load from DRAM to SRAM (the ultra-fast memory near the cores)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # The actual math
    output = x + y

    # Write back to DRAM
    tl.store(out_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # The 'Grid' tells the GPU how many blocks to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch!
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

if __name__ == "__main__":
    # Test on CPU using Interpreter if no GPU is found
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    size = 2**12
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    
    # Verify math
    result = add(x, y)
    expected = x + y
    
    if torch.allclose(result, expected):
        print("✅ Success: Triton math matches PyTorch!")
    else:
        print("❌ Error: Math mismatch.")