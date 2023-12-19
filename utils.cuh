#include "cuda_fp16.h"
#include <bits/stdc++.h>
#include "sys/time.h"

void host_malloc(half** ptr, int size) 
{
    *ptr = (half*)malloc(sizeof(half) * size);
}
void device_malloc(half** ptr, int size)
{
  cudaMalloc((void**)ptr, sizeof(half) * size);
}
void copy_to_device(half** d_ptr, half** h_ptr, int size)
{
  cudaMemcpy((*d_ptr), (*h_ptr), sizeof(half) * size, cudaMemcpyHostToDevice);
}

void copy_to_host(half** h_ptr, half** d_ptr, int size)
{
  cudaMemcpy((*h_ptr), (*d_ptr), sizeof(half) * size, cudaMemcpyDeviceToHost);
}

void host_to_host(half** h_ptr, half** d_ptr, int size)
{
  cudaMemcpy((*h_ptr), (*d_ptr), sizeof(half) * size, cudaMemcpyDeviceToDevice);
}

void prepare_data(half* h_A, int M, int K) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 2);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = static_cast<half>(dist(e2));
        }
    }
}

float time_diff(struct timeval *start, struct timeval *end) { // ns
  return (end->tv_sec - start->tv_sec) * 1e9 + 1e3 * (end->tv_usec - start->tv_usec);
}
