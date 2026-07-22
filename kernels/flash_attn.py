import torch
import triton
import triton.language as tl

@triton.jit
def flash_attn_kernel(
    Q, K, V, L, Out,
    stride_qm, stride_kn, stride_vn, stride_om,
    seq_len, d_head: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program handles one 'head' of the attention
    pid = tl.program_id(0)
    
    # Pointers to the start of this head's data
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

    # Loop over Key/Value blocks (Streaming)
    for start_n in range(0, seq_len, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K_ptr + rn[None, :] * d_head + rk[:, None])
        v = tl.load(V_ptr + rn[:, None] * d_head + rk[None, :])

        # 1. Dot Product (Q @ K.T)
        qk = tl.dot(q, k)
        
        # 2. Online Softmax update
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Re-scale previous accumulator to match new maximum
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        
        # Accumulate current block
        acc += tl.dot(p.to(tl.float16), v)
        
        # Update running statistics
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    # Final normalization
    acc = acc / l_i[:, None]
    tl.store(Out_ptr + rm[:, None] * d_head + rk[None, :], acc.to(tl.float16))


def flash_attn(q, k, v):
    """Block-wise attention over (BATCH, N_HEADS, SEQ_LEN, D_HEAD) tensors.

    "Lite" limitation: the kernel loads the full query block for a given
    (batch, head) in one shot with no bounds masking, so SEQ_LEN must be a
    power of two (BLOCK_M == SEQ_LEN exactly).
    """
    assert q.shape == k.shape == v.shape
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = q.shape
    assert SEQ_LEN & (SEQ_LEN - 1) == 0, "flash_attn (lite) requires a power-of-2 SEQ_LEN"

    out = torch.empty_like(q)
    l = torch.empty((BATCH * N_HEADS, SEQ_LEN), device=q.device, dtype=torch.float32)

    BLOCK_M = SEQ_LEN
    BLOCK_N = min(128, SEQ_LEN)
    stride = SEQ_LEN * D_HEAD  # elements between consecutive (batch, head) blocks
    grid = (BATCH * N_HEADS,)

    flash_attn_kernel[grid](
        q, k, v, l, out,
        stride, stride, stride, stride,
        SEQ_LEN, D_HEAD,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out