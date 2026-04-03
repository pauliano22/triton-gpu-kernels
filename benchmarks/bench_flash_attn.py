import torch
import triton
import triton.testing
from kernels.flash_attn import flash_attn

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['BATCH', 'N_HEADS', 'D_HEAD', 'SEQ_LEN'],
        x_vals=[2**i for i in range(10, 15)], 
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['FlashAttention-Lite (Triton)', 'Naive Attention (PyTorch)'],
        styles=[('blue', '-'), ('red', '-')],
        ylabel='TFLOPS',
        plot_name='flash-attention-performance',
        args={},
    )
)
def benchmark(BATCH, N_HEADS, D_HEAD, SEQ_LEN, provider):
    device = 'cuda'
    dtype = torch.float16
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), device=device, dtype=dtype)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), device=device, dtype=dtype)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), device=device, dtype=dtype)
    
    if provider == 'torch':
        # Naive implementation for comparison
        def naive_attn(q, k, v):
            mask = torch.matmul(q, k.transpose(-2, -1))
            attn = torch.softmax(mask, dim=-1)
            return torch.matmul(attn, v)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_attn(q, k, v))
    
    if provider == 'triton':
        # Your FlashAttention-Lite
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: flash_attn(q, k, v))

    # TFLOPS calculation for Attention: 2 * Batch * Heads * Seq^2 * D_head
    tflops = lambda ms: 2 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * D_HEAD / (ms * 1e-3) / 1e12
    return tflops(ms), tflops(max_ms), tflops(min_ms)

if __name__ == "__main__":
    benchmark.run(save_path='.', show_plots=True)