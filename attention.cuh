#include "kernel.cuh"

void GEMM_base(half *d_A, half *d_B, half *d_C, int M, int N, int K) {
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 grid((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    gemm_baseline<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }
}

void GEMM(half *d_A, half *d_B, half *d_C, int M, int N, int K) {
    const int BM = 128, BN = 256, BK = 32;
    dim3 blockDim(256);
    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;

    dim3 gridDim(BX, BY);

    cudaFuncSetAttribute(GEMM_sharedmem_wmma,
            cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    unsigned int dsmem = 2 * (BM * (BK + 8) + BK * (BN + 8)) * sizeof(half);
    GEMM_sharedmem_wmma<<<gridDim, blockDim, dsmem>>>(d_A, d_B, d_C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }
}

void Transpose(half *d_A, half *d_B, int M, int N) {
    dim3 blockDim(256);
    dim3 gridDim(128);
    transpose<<<gridDim, blockDim>>>(d_A, d_B, M, N);
}

void SoftMax(half *p, half *s, int N) {
    host_to_host(&s, &p, N * N);
    half *h_mask;
    host_malloc(&h_mask, N * N);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if(i <= j)   
                h_mask[i * N + j] = 1;
            else  
                h_mask[i * N + j] = 0;
        }
    }
    half *d_mask;
    device_malloc(&d_mask, N * N);
    copy_to_device(&d_mask, &h_mask, N * N);

    int size_per_head = 1;
    int batch_size = 1;
    int head_num = 1;
    half scaler = 1 / sqrtf((half)size_per_head * (half)1.0f);
   // launch fusion kernel
    dim3 grid(N);
    dim3 block(N);
    softmax_kernel<<<grid, block>>>(s, d_mask, batch_size, head_num, N, scaler);
}

void SoftMax_base(half *p, half *s, int N) {
    host_to_host(&s, &p, N * N);
    half *h_mask;
    host_malloc(&h_mask, N * N);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if(i <= j)   
                h_mask[i * N + j] = 1;
            else  
                h_mask[i * N + j] = 0;
        }
    }
    half *d_mask;
    device_malloc(&d_mask, N * N);
    copy_to_device(&d_mask, &h_mask, N * N);

    int size_per_head = 1;
    int batch_size = 1;
    int head_num = 1;
    half scaler = 1 / sqrtf((half)size_per_head * (half)1.0f);
   // launch fusion kernel
    dim3 grid(N);
    dim3 block(N);
    addmask_kernel<<<grid, block>>>(s, d_mask, batch_size, head_num, N, scaler);
    cudaDeviceSynchronize();
    softcal_kernel<<<grid, block>>>(s, d_mask, batch_size, head_num, N, scaler);
}

void attention(half *in, 
    half *Wq, half *Wk,half *Wv,
    half *q, half *k, half *v,
    half *kT,
    half *p, half *s,
    half *o, half *Wo,
    half *out,
    int N, int d,
    int kd,
    FILE* file)
{
    struct timeval start, end;
    struct timeval sub1, sub2;
    float t1 = 0; // get q, k, v
    float t2 = 0; // get kT
    float t3 = 0; // q * k^T
    float t4 = 0; // softmax
    float t5 = 0; // s * v
    float t6 = 0; // o * Wo

    gettimeofday(&start, NULL);

    printf("start attention...\n");
    if(kd == 1) {
        for(int i = 0; i < 1000; i++) {
            // get q, k, v
            gettimeofday(&sub1, NULL);
            GEMM(in, Wq, q, N, d, d);
            GEMM(in, Wk, k, N, d, d);
            GEMM(in, Wv, v, N, d, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t1 += time_diff(&sub1, &sub2);

            // get kT
            gettimeofday(&sub1, NULL);
            Transpose(k, kT, N, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t2 += time_diff(&sub1, &sub2);

            // q * k^T
            gettimeofday(&sub1, NULL);
            GEMM(q, kT, p, N, N, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t3 += time_diff(&sub1, &sub2);

            // softmax
            gettimeofday(&sub1, NULL);
            SoftMax(p, s, N);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t4 += time_diff(&sub1, &sub2);

            // s * v
            gettimeofday(&sub1, NULL);
            GEMM(s, v, o, N, d, N);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t5 += time_diff(&sub1, &sub2);

            // o * Wo
            gettimeofday(&sub1, NULL);
            GEMM(o, Wo, out, N, d, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t6 += time_diff(&sub1, &sub2);
        }
    }   
    else if(kd == 0) {
        for(int i = 0; i < 1000; i++) {
            // get q, k, v
            gettimeofday(&sub1, NULL);
            GEMM_base(in, Wq, q, N, d, d);
            GEMM_base(in, Wk, k, N, d, d);
            GEMM_base(in, Wv, v, N, d, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t1 += time_diff(&sub1, &sub2);

            // get kT
            gettimeofday(&sub1, NULL);
            Transpose(k, kT, N, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t2 += time_diff(&sub1, &sub2);

            // q * k^T
            gettimeofday(&sub1, NULL);
            GEMM_base(q, kT, p, N, N, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t3 += time_diff(&sub1, &sub2);

            // softmax
            gettimeofday(&sub1, NULL);
            SoftMax_base(p, s, N);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t4 += time_diff(&sub1, &sub2);

            // s * v
            gettimeofday(&sub1, NULL);
            GEMM_base(s, v, o, N, d, N);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t5 += time_diff(&sub1, &sub2);

            // o * Wo
            gettimeofday(&sub1, NULL);
            GEMM_base(o, Wo, out, N, d, d);
            cudaDeviceSynchronize();
            gettimeofday(&sub2, NULL);
            t6 += time_diff(&sub1, &sub2);
        }
    }
    gettimeofday(&end,NULL);
    fprintf(file, "%d, %d, %d, %f, %f, %f, %f, %f, %f, %f\n", kd, N, d, time_diff(&start, &end)/1000, t1/1000, t2/1000, t3/1000, t4/1000, t5/1000, t6/1000);
    printf("attention end...\n");
}