# AMD ROCm & HIP Introduction

## Overview
ROCm (Radeon Open Compute) is AMD's open-source software stack for GPU computing. HIP (Heterogeneous-Compute Interface for Portability) is the C++ runtime API and kernel language that allows developers to write portable code that can run on both AMD and NVIDIA GPUs.

## HIP vs CUDA
HIP is designed to be very similar to CUDA.
- **Syntax**: `hipMalloc` vs `cudaMalloc`, `hipMemcpy` vs `cudaMemcpy`.
- **Keywords**: `__global__`, `__device__`, `__shared__` work the same.
- **Porting**: The `hipify` tool can automatically translate CUDA source code to HIP.

## Architecture Differences (CDNA)
- **Wavefronts**: AMD GPUs (CDNA) typically use a wavefront size of 64 (vs 32 threads in a CUDA warp). This affects warp-shuffle operations and voting primitives.
- **LDS**: Local Data Share (equivalent to CUDA Shared Memory).
- **Matrix Cores**: AMD's Matrix Core technology (similar to Tensor Cores) is accessed via specific intrinsics or libraries like rocBLAS/ck (Composable Kernel).

## Key Optimization Tips
1.  **Occupancy**: Due to larger register files and different SIMD unit structures, occupancy tuning might differ from NVIDIA.
2.  **Memory**: HBM bandwidth is massive; ensure memory coalescing just like in CUDA.

