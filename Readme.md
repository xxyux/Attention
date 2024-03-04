# Introduction
This is my GPU course final project in MICS600J. The main content is my attempt to implement the attention mechanism efficiently.

# Paraments
In order to simplify the process, I replaced the original matrix dimensions [batch_size, nheads, seq_len, headdim] with [seq_len, headdim].

N: seq_len
d: headdim

# Algorithm process / matrix dimension 
The attention mechanism is well known and I won’t go into details.
1. In = N * d
2. WQ WK WV = d * d
3. Q K V = N * d
4. P = Q * K^T = N * N 
5. S = SoftMax(P) = N * N
6. o = S * V = N * d
6. Out = S * W = N * d

# Technical Highlights
1. Use `Tensor core` to compute GEMM.
2. Use `Asynchronous transfer` to overlap computation and communication(transfer data from global memory to shared memory).
3. `Bank Conflict` free.

# Build
Use `make` command to build program.

# Experiment
1. Range for N,d: N~(32, 1024), d~(32, 2048). Please get more detail from [script](https://github.com/xxyux/Attention/blob/main/test.sh).
2. Test on NVIDIA A100 in HKUST(GZ)-HPC Server.

# Done
Fine-tuning Llama-2-7B, when using Sparse Attention Mechanism, we found that accuracy can be improved and restored with little overhead.  

# Doing...
1. Kernel Fusion, just like [Flash attention](https://github.com/Dao-AILab/flash-attention).
2. Sparse Attention Mechanism, just like [DFSS](https://github.com/apuaaChen/DFSS/tree/main), make full use of `sparse tensor core`.

