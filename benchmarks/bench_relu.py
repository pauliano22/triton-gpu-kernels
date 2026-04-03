import torch
import triton
import triton.testing
import time
from kernels.relu import relu

def simple_bench(fn, iterations=10):
    # Warm up
    for _ in range(5):
        fn()
    
    # Measure
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    end = time.perf_counter()
    
    ms = (end - start) * 1000 / iterations
    return ms, ms, ms # Returning 3 values to match Triton's expected format

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'], 
        x_vals=[2**i for i in range(12, 22, 1)], # Slightly smaller range for CPU speed
        line_arg='provider', 
        line_vals=['triton', 'torch'], 
        line_names=['Triton (Interpreted)', 'PyTorch (CPU)'], 
        styles=[('blue', '-'), ('green', '-')], 
        ylabel='Execution Time (ms)', # Switched to ms because GB/s is misleading on CPU
        plot_name='relu-performance-local',
        args={}, 
    )
)
def benchmark(size, provider):
    device = 'cpu' # Stay on CPU for the local benchmark
    x = torch.randn(size, device=device)
    
    if provider == 'torch':
        return simple_bench(lambda: torch.relu(x))
    if provider == 'triton':
        # Force interpret mode inside the benchmark
        import os
        os.environ["TRITON_INTERPRET"] = "1"
        return simple_bench(lambda: relu(x))

if __name__ == "__main__":
    benchmark.run(save_path='.', show_plots=True)