# Introduction
This is my GPU course final project in MICS600J. The main content is my attempt to handwrite the attention process.

# Paraments
In order to simplify the process, I replaced the original matrix dimensions [batch_size, nheads, seq_len, headdim] with [seq_len, headdim].

N: seq_len
d: headdim

# Algorithm process
1. In = N * d
2. WQ WK WV = d * d
3. Q K V = N * d
4. P = Q * K^T = N * N 
5. S = SoftMax(P) = N * N
6. o = S * V = N * d
6. Out = S * W = N * d

# Technical Highlights
1. Use `Tensor core` to compute GEMM.
2. Use `Asynchronous transfer` to overlap computation and communication.

# Build
Use `make` command to build program.
