import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    Q, K, V, Out,
    stride_qm, stride_kn, stride_vn, stride_om,
    seq_len,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, d_head: tl.constexpr,
):
    # Each program handles one (batch, head) slice of the attention
    pid = tl.program_id(0)

    # Pointers to the start of this (batch, head) slice's data
    Q_ptr = Q + pid * stride_qm
    K_ptr = K + pid * stride_kn
    V_ptr = V + pid * stride_vn
    Out_ptr = Out + pid * stride_om

    # Initialize online softmax statistics (Running Max and Running Sum)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_head], dtype=tl.float32)

    # Load Query block once (stays in SRAM)
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, d_head)
    q = tl.load(Q_ptr + rm[:, None] * d_head + rk[None, :])

    # Loop over Key/Value blocks (Streaming) across the full key/value sequence
    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + rn[None, :] * d_head + rk[:, None])
        v = tl.load(V_ptr + rn[:, None] * d_head + rk[None, :])

        # 1. Dot Product (Q @ K.T)
        qk = tl.dot(q, k)

        # 2. Online Softmax update (new running max must consider prior blocks too,
        # not just this block's local max)
        m_ij = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_ij)
        p = tl.exp(qk - m_new[:, None])
        l_ij = tl.sum(p, 1)

        # Re-scale previous accumulator to match new maximum
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]

        # Accumulate current block
        acc += tl.dot(p.to(tl.float16), v)

        # Update running statistics
        l_i = l_i * alpha + l_ij
        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]
    tl.store(Out_ptr + rm[:, None] * d_head + rk[None, :], acc.to(tl.float16))


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """FlashAttention-Lite: one program per (batch, head) pair, with the full
    query block resident in SRAM and K/V streamed in BLOCK_N chunks. SEQ_LEN
    must be a power of 2 (no masking is applied to the query or key/value
    blocks).
    """
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)

    # Q, K, V, and Out all share the same (BATCH, N_HEADS, SEQ_LEN, D_HEAD)
    # layout, so the per-(batch, head) slab stride is the same for all of them.
    stride_head = SEQ_LEN * D_HEAD

    grid = (BATCH * N_HEADS,)
    flash_attn_kernel[grid](
        q, k, v, out,
        stride_head, stride_head, stride_head, stride_head,
        SEQ_LEN,
        BLOCK_M=SEQ_LEN, BLOCK_N=min(SEQ_LEN, 64), d_head=D_HEAD,
    )
    return out


if __name__ == "__main__":
    # Test on CPU using Interpreter if no GPU is found
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 2, 4, 128, 64
    q = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), device=device, dtype=torch.float16)
    k = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), device=device, dtype=torch.float16)
    v = torch.randn((BATCH, N_HEADS, SEQ_LEN, D_HEAD), device=device, dtype=torch.float16)

    result = flash_attn(q, k, v)

    # Reference: naive Q.K^T -> softmax -> .V attention (matches benchmarks/bench_flash_attn.py),
    # computed in float32 so the comparison isn't dominated by fp16 softmax rounding noise.
    attn = torch.softmax(torch.matmul(q.float(), k.float().transpose(-2, -1)), dim=-1)
    expected = torch.matmul(attn, v.float()).half()

    if torch.allclose(result, expected, atol=2e-2, rtol=2e-2):
        print("✅ Success: Triton math matches PyTorch!")
    else:
        print("❌ Error: Math mismatch.")
        print((result - expected).abs().max())
