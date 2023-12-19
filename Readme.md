N: seq_len
d: feature_len

1. In = N * d
2. WQ WK WV = d * d
3. Q K V = N * d
4. P = Q * K^T = N * N 
5. S = SoftMax(P) = N * N
6. o = S * V = N * d
6. Out = S * W = N * d

