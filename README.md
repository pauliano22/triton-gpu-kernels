# Triton GPU Kernels: Hopper Architecture Optimization
High-performance LLM operator kernels implemented in OpenAI Triton, targeting NVIDIA H100 (Hopper) systems. This library demonstrates systems-level optimizations for bypassing the memory wall through kernel fusion and reduced-precision arithmetic.

## Technical Specifications
- **Fused LayerNorm (FP8):** Custom kernel performing LayerNorm and Symmetric Quantization (E4M3) in a single pass. Reduces global memory write-traffic by 50% by eliminating intermediate high-precision HBM writes.
- **FlashAttention-Lite:** Tiled attention implementation utilizing Welford-style online softmax to maintain a fixed SRAM footprint regardless of sequence length.
- **TMA & L2-Cache Management:** Optimized tile sizes (128x128) to maximize L1/SRAM residency and minimize L2-cache thrashing on H100 SXM5 interconnects.

## Implementation Roadmap
- **Vector Add:** Baseline Tiling (Verified)
- **Fused ReLU:** Memory Bandwidth Optimization (Verified)
- **LayerNorm:** Cross-thread Reductions (Benchmarking)
- **LN + FP8:** Bandwidth Compression (Benchmarking)
- **FlashAttention:** SRAM Tiling / Online Softmax (Benchmarking)

## Environment Setup
The suite is designed for deployment on AWS P5 (H100) or equivalent Hopper instances.

### Local Verification (Interpret Mode)
Verify mathematical correctness without a local GPU:
```bash
PYTHONPATH=. TRITON_INTERPRET=1 python3 kernels/vector_add.py