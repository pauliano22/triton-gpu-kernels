import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    Q, K, V, Out,
    stride_qz, stride_kz, stride_vz, stride_oz,
    seq_len, d_head: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program handles one (batch, head) slice of the attention
    pid = tl.program_id(0)

    # Pointers to the start of this slice's data
    Q_ptr = Q + pid * stride_qz
    K_ptr = K + pid * stride_kz
    V_ptr = V + pid * stride_vz
    Out_ptr = Out + pid * stride_oz

    # Initialize online softmax statistics (Running Max and Running Sum)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, d_head], dtype=tl.float32)

    # Load Query block once (stays in SRAM); mask rows past seq_len
    rm = tl.arange(0, BLOCK_M)
    rk = tl.arange(0, d_head)
    m_mask = rm < seq_len
    q = tl.load(Q_ptr + rm[:, None] * d_head + rk[None, :], mask=m_mask[:, None], other=0.0)

    # Loop over Key/Value blocks (Streaming)
    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)
        n_mask = rn < seq_len
        k = tl.load(K_ptr + rn[None, :] * d_head + rk[:, None], mask=n_mask[None, :], other=0.0)
        v = tl.load(V_ptr + rn[:, None] * d_head + rk[None, :], mask=n_mask[:, None], other=0.0)

        # 1. Dot Product (Q @ K.T), masking out-of-range keys before softmax
        qk = tl.dot(q, k)
        qk = tl.where(n_mask[None, :], qk, float('-inf'))

        # 2. Online Softmax update (running max, not just this block's max)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Re-scale previous accumulator to match new running maximum
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        # Accumulate current block
        acc += tl.dot(p.to(tl.float16), v)

        # Update running statistics
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Final normalization
    acc = acc / l_i[:, None]
    tl.store(Out_ptr + rm[:, None] * d_head + rk[None, :], acc.to(tl.float16), mask=m_mask[:, None])


def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Fused attention: softmax(Q @ K^T) @ V (no scaling, no causal mask —
    matches the naive PyTorch baseline this is benchmarked against).

    q, k, v: (BATCH, N_HEADS, SEQ_LEN, D_HEAD), same shape, float16.
    """
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = q.shape
    assert k.shape == q.shape and v.shape == q.shape

    q = q.reshape(BATCH * N_HEADS, SEQ_LEN, D_HEAD).contiguous()
    k = k.reshape(BATCH * N_HEADS, SEQ_LEN, D_HEAD).contiguous()
    v = v.reshape(BATCH * N_HEADS, SEQ_LEN, D_HEAD).contiguous()
    out = torch.empty_like(q)

    BLOCK_M = triton.next_power_of_2(SEQ_LEN)
    BLOCK_N = min(BLOCK_M, 128)

    grid = (BATCH * N_HEADS,)
    flash_attn_kernel[grid](
        q, k, v, out,
        q.stride(0), k.stride(0), v.stride(0), out.stride(0),
        SEQ_LEN, D_HEAD,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out.reshape(BATCH, N_HEADS, SEQ_LEN, D_HEAD)
