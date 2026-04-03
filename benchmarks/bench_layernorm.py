import torch
import triton
import triton.testing
from kernels.layer_norm import layernorm

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], 
        x_vals=[1024 * i for i in range(1, 33)], 
        line_arg='provider', 
        line_vals=['triton', 'torch'], 
        line_names=['Triton LayerNorm', 'PyTorch LayerNorm'], 
        styles=[('blue', '-'), ('green', '-')], 
        ylabel='GB/s', 
        plot_name='layernorm-performance',
        args={'M': 4096}, 
    )
)
def benchmark(M, N, provider):
    device = 'cuda'
    x = torch.randn((M, N), device=device, dtype=torch.float16)
    w = torch.ones(N, device=device, dtype=torch.float16)
    b = torch.zeros(N, device=device, dtype=torch.float16)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(x, (N,), w, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: layernorm(x, w, b), quantiles=quantiles)
    
    # GB/s calculation
    gbps = lambda ms: (M * N * 2 * 3) / ms / 1e6 # 2 reads (X, W/B), 1 write (Y)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    benchmark.run(save_path='.', show_plots=True)