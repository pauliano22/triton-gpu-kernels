# Triton GPU Kernels (H100 Optimized)
High-performance GPU kernels implemented in OpenAI Triton, targeting NVIDIA Hopper (H100) architecture.

## Roadmap
- [ ] **Level 1: Vector Addition** (Baseline environment & CUDA verification)
- [ ] **Level 2: Fused Activations** (Custom ReLU/GELU vs. Native PyTorch)
- [ ] **Level 3: Fused LayerNorm** (Transformer optimization with SRAM tiling)
- [ ] **Level 4: FlashAttention-Lite** (Tiled Softmax & Online Normalization)

## Setup
```bash
source venv/bin/activate
# To run without a GPU (Interpret Mode):
TRITON_INTERPRET=1 python3 kernels/vector_add.py