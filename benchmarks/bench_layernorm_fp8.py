import torch
import triton
import triton.testing
from kernels.layer_norm_fp8 import layernorm_fp8

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'], 
        x_vals=[512 * i for i in range(2, 32)], 
        line_arg='provider', 
        line_vals=['torch_bf16', 'triton_fp8'], 
        line_names=['PyTorch BF16', 'Triton Fused FP8'], 
        styles=[('green', '-'), ('blue', '-')], 
        ylabel='GB/s', 
        plot_name='layernorm-fp8-vs-bf16',
        args={'M': 2048}, 
    )
)
def benchmark(M, N, provider):
    device = 'cuda'
    x = torch.randn((M, N), device=device, dtype=torch.bfloat16)
    w = torch.ones(N, device=device, dtype=torch.bfloat16)
    b = torch.zeros(N, device=device, dtype=torch.bfloat16)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch_bf16':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(x, (N,), w, b), quantiles=quantiles)
    if provider == 'triton_fp8':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: layernorm_fp8(x, w, b), quantiles=quantiles)
    
    # Calculate Bandwidth: (Reads + Writes) / time
    # BF16: M*N*2 (read X) + M*N*2 (read W/B) + M*N*2 (write Y) 
    # FP8:  M*N*2 (read X) + M*N*2 (read W/B) + M*N*1 (write Y) -> 25% less traffic!
    gbps = lambda ms: (M * N * 6) / ms / 1e6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    benchmark.run(save_path='.', show_plots=True)