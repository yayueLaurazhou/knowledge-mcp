# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

*Author: Tri Dao*
*arXiv: 2307.08691 | ICLR 2024*

---

## Abstract

Scaling Transformers to longer sequences requires efficient attention kernels. FlashAttention exploits the GPU memory hierarchy to achieve IO-aware attention with linear memory and 2–4× speedup, but still reaches only 25–40% of theoretical peak FLOPs/s on A100. The bottleneck is suboptimal work partitioning between thread blocks and warps, causing low occupancy and unnecessary shared memory traffic. FlashAttention-2 introduces improved work partitioning and parallelism strategies, yielding ~2× speedup over FlashAttention-1 and reaching 50–73% of theoretical peak FLOPs/s on A100 GPUs.

---

## 1. Introduction

The attention mechanism in Transformers has O(N²) time and memory complexity in sequence length N. FlashAttention (Dao et al., 2022) showed that by tiling the attention computation and fusing it into a single kernel with online softmax, one can reduce HBM reads/writes from O(N²) to O(N), achieving significant speedups in practice.

However, FlashAttention-1 was not nearly as fast as optimized GEMM operations. On an A100 GPU:
- Optimized GEMM achieves ~70–80% of theoretical peak
- FlashAttention-1 achieved only 25–40% of theoretical peak

The inefficiency stems from:
1. Suboptimal work partitioning between thread blocks (low occupancy for small batch sizes or few heads)
2. Suboptimal work distribution among warps within a thread block (shared memory bottleneck)
3. Excess non-matmul operations (rescaling in the backward pass)

---

## 2. Background: GPU Architecture and Execution Model

### 2.1 GPU Execution Hierarchy

- **Thread blocks** are scheduled onto Streaming Multiprocessors (SMs)
- Each SM contains multiple **warps** (32 threads each)
- **Registers** and **shared memory (SRAM)** are on-chip and fast
- **HBM (High Bandwidth Memory)** is off-chip and slow

For an A100:
- HBM bandwidth: ~2 TB/s
- SRAM bandwidth: ~19 TB/s (within SM)
- Tensor core peak throughput: 312 TFLOP/s (FP16)
- Number of SMs: 108
- SRAM per SM: 192 KB

### 2.2 Online Softmax (Flash Attention Recap)

Standard attention: S = QK^T, P = softmax(S), O = PV

FlashAttention computes this in tiles using the online softmax trick:
- Maintain running max m_i and sum l_i
- Update rescaling factors as new tiles are processed
- Write only the final output O to HBM

---

## 3. FlashAttention-2 Algorithm

### 3.1 Reducing Non-Matmul FLOPs

In the backward pass of FlashAttention-1, the algorithm recomputed the attention scores and performed rescaling operations that required multiple non-matmul steps. FlashAttention-2 restructures the backward pass to:
1. Avoid redundant rescaling of dQ across blocks
2. Group the masking operations to minimize divergence

The key insight: instead of rescaling dQ at every step (once per K/V block), rescale once at the end. This reduces the number of non-matmul operations proportionally to the number of blocks.

**Algorithm 1: FlashAttention-2 Forward Pass**

```
Input: Q, K, V ∈ R^{N×d}, softmax scale τ = 1/√d
Output: O ∈ R^{N×d}, L ∈ R^N (logsumexp)

Divide Q into blocks Q_1, ..., Q_{T_r} of size B_r × d
Divide K, V into blocks K_1, ..., K_{T_c} of size B_c × d

for i = 1 to T_r:
    Load Q_i from HBM to SRAM
    Initialize O_i = 0 ∈ R^{B_r×d}, l_i = 0 ∈ R^{B_r}, m_i = -∞ ∈ R^{B_r}
    for j = 1 to T_c:
        Load K_j, V_j from HBM to SRAM
        S_ij = τ · Q_i K_j^T ∈ R^{B_r×B_c}
        m̃_ij = rowmax(S_ij) ∈ R^{B_r}
        P̃_ij = exp(S_ij - m̃_ij) ∈ R^{B_r×B_c}
        l̃_ij = rowsum(P̃_ij) ∈ R^{B_r}

        m_i_new = max(m_i, m̃_ij)
        l_i_new = exp(m_i - m_i_new) · l_i + exp(m̃_ij - m_i_new) · l̃_ij

        O_i = diag(l_i_new)^{-1} · (diag(l_i) · exp(m_i - m_i_new) · O_i
                                    + exp(m̃_ij - m_i_new) · P̃_ij V_j)
        m_i = m_i_new, l_i = l_i_new

    L_i = m_i + log(l_i)
    Write O_i, L_i to HBM
```

### 3.2 Parallelism Improvements

**FlashAttention-1 parallelism:**
- Outer loop over batch dimension B and number of heads H
- Thread blocks assigned to (batch, head) pairs
- Problem: when B × H is small (e.g., large models, single-batch inference), GPU is underutilized

**FlashAttention-2 addition:**
- Also parallelize over sequence length N (specifically over Q blocks)
- For the forward pass: parallelize over both heads AND query blocks
- This ensures full GPU utilization even for small B × H

This parallelization is natural for the forward pass (independent Q_i blocks), and requires care in the backward pass (multiple thread blocks writing to the same dK, dV location → use atomic operations or separate passes).

### 3.3 Work Partitioning Between Warps

**FlashAttention-1 (sliced-K scheme):**
- Within a thread block: split K, V across 4 warps
- Each warp computes a portion of Q K^T and P V
- Problem: all 4 warps must synchronize to compute rowmax/rowsum → shared memory write + sync + reduction

**FlashAttention-2 (split-Q scheme):**
- Instead: split Q across warps, keep K, V shared
- Each warp independently computes its portion of the output
- Each warp does: Q_warp × K^T, softmax (independently), × V
- No synchronization needed between warps within the thread block
- The output slices are directly independent

This eliminates inter-warp synchronization and reduces shared memory traffic.

---

## 4. Implementation Details

### 4.1 Tile Sizes

For A100 GPU (80 GB HBM, 192 KB SRAM per SM):
- Forward pass: B_r = B_c = 64 for head dimension d ≤ 64; B_r = 64, B_c = 128 for d ≤ 128
- The tile sizes balance register pressure vs. memory bandwidth utilization

### 4.2 Online Softmax Computation

The key invariant maintained per row:
- m: running maximum of current tile's attention logits
- l: running sum of exp(S - m) values
- O: running output (will be rescaled at the end)

At each new K/V block, update:
```
m_new = max(m, m_new_block)
l_new = exp(m - m_new) * l + exp(m_new_block - m_new) * l_new_block
O_new = diag(exp(m - m_new)) * O + P_new * V_new
```

### 4.3 Backward Pass

The backward pass computes dQ, dK, dV. FlashAttention-2 uses the stored O and L (logsumexp) from the forward pass to recompute attention weights on-the-fly without storing the N×N attention matrix.

Key: to compute dV, dK for a K/V block j, we need to loop over all Q blocks. To compute dQ for Q block i, we loop over all K/V blocks.

FlashAttention-2 backward:
```
for j = 1 to T_c:  # outer loop for dK, dV
    for i = 1 to T_r:
        Load Q_i, O_i, dO_i, L_i from HBM
        S_ij = τ · Q_i K_j^T
        P_ij = exp(S_ij - L_i)  # recompute using stored L
        dV_j += P_ij^T dO_i
        dP_ij = dO_i V_j^T
        D_i = rowsum(dO_i ∘ O_i)  # dot product trick
        dS_ij = P_ij ∘ (dP_ij - D_i)
        dK_j += dS_ij^T Q_i
        dQ_i += dS_ij K_j    # atomic add across j
```

---

## 5. Multi-Query and Grouped-Query Attention

FlashAttention-2 supports:
- **Multi-Head Attention (MHA)**: H_q = H_kv, standard case
- **Multi-Query Attention (MQA)**: H_q > H_kv = 1, all queries share a single K, V head
- **Grouped-Query Attention (GQA)**: H_q = G × H_kv, groups of queries share K, V heads

For MQA/GQA, the K and V blocks are shared across multiple Q heads, saving memory bandwidth and allowing higher throughput.

---

## 6. Performance Results

### 6.1 Forward Pass Benchmarks (A100 SXM4 80GB)

| Configuration | FlashAttention-1 | FlashAttention-2 | cuDNN |
|--------------|-----------------|-----------------|-------|
| seqlen=128, d=64 | ~100 TFLOPs/s | ~200 TFLOPs/s | ~170 TFLOPs/s |
| seqlen=512, d=64 | ~130 TFLOPs/s | ~230 TFLOPs/s | ~210 TFLOPs/s |
| seqlen=2048, d=64 | ~130 TFLOPs/s | ~230 TFLOPs/s | ~220 TFLOPs/s |
| seqlen=4096, d=64 | ~130 TFLOPs/s | ~225 TFLOPs/s | ~225 TFLOPs/s |

Peak achieved: **~230 TFLOPs/s** (~73% of theoretical 312 TFLOPs/s FP16 on A100)

### 6.2 End-to-End Training Speedup

GPT-3 style models trained with FlashAttention-2:
- **GPT-3 (125M params)**: 2.1× speedup over standard attention
- **GPT-3 (1.3B params)**: 1.8× speedup
- Training throughput: up to **225 TFLOPs/s per A100** (72% MFU)

### 6.3 Speedup vs FlashAttention-1

- Forward: ~2× across most configurations
- Backward: ~2× for most configurations
- End-to-end training: ~1.3–2.0× speedup

### 6.4 Hardware Utilization

- FlashAttention-1: 25–40% utilization on A100
- FlashAttention-2: 50–73% utilization on A100 (close to GEMM baseline)
- GEMM baseline: ~70–80% utilization

---

## 7. Long-Context Performance

FlashAttention-2 enables efficient attention on very long sequences:
- Up to 16K context on A100 (80 GB) for training
- Memory: O(N) instead of O(N²)
- Enables training GPT-4 class models with 8K–32K context

---

## 8. RTX 3090 / A100 Comparison

| GPU | Peak FP16 TFLOPs/s | FA-2 achieved |
|----|-------------------|--------------|
| A100 SXM4 | 312 | 230 (~73%) |
| RTX 3090 | 142 | 65 (~46%) |

---

## 9. Key Contributions Summary

1. **Reduced non-matmul FLOPs**: Restructured backward pass to avoid redundant rescaling operations
2. **Parallelism over sequence length**: Thread blocks assigned to Q blocks, enabling full utilization at small batch sizes
3. **Better warp partitioning**: Split-Q scheme eliminates inter-warp synchronization within thread blocks
4. **MQA/GQA support**: Native support for grouped-query attention variants
5. **~2× speedup** over FlashAttention-1; **50–73% FLOPs utilization** on A100

---

## References

- Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022.
- Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.
- Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need. Multi-Query Attention.
- Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models. EMNLP 2023.
