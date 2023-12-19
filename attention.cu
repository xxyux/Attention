#include "stdio.h"
#include "utils.cuh"
#include "attention.cuh"
#include "check.h"


int main(int argc, char *argv[]) {

    int N = atoi(argv[1]); // 32 - 1024
    int d = atoi(argv[2]); // 32 - 2048
    int kd = atoi(argv[3]); // 0: baseline, 1: our method

    // host
    half *h_in;
    half *h_Wq, *h_Wk, *h_Wv;
    half *h_q, *h_k, *h_v;
    half *h_kT;
    half *h_p, *h_s;
    half *h_o, *h_Wo;
    half *h_out;
    // malloc
    host_malloc(&h_in, N * d);
    host_malloc(&h_Wq, d * d);host_malloc(&h_Wk, d * d);host_malloc(&h_Wv, d * d);
    host_malloc(&h_q, N * d);host_malloc(&h_k, N * d);host_malloc(&h_v, N * d);
    host_malloc(&h_kT, N * N);
    host_malloc(&h_p, N * N);host_malloc(&h_s, N * N);
    host_malloc(&h_o, N * d);host_malloc(&h_Wo, d * d);
    host_malloc(&h_out, N * d);
    // // init
    prepare_data(h_in, N, d);
    prepare_data(h_Wq, d, d);prepare_data(h_Wk, d, d);prepare_data(h_Wv, d, d);
    prepare_data(h_Wo, d, d);

    // device
    half *d_in;
    half *d_Wq, *d_Wk, *d_Wv;
    half *d_q, *d_k, *d_v;
    half *d_kT;
    half *d_p, *d_s;
    half *d_o, *d_Wo;
    half *d_out;
    // // malloc
    device_malloc(&d_in, N * d);
    device_malloc(&d_Wq, d * d);device_malloc(&d_Wk, d * d);device_malloc(&d_Wv, d * d);
    device_malloc(&d_q, N * d);device_malloc(&d_k, N * d);device_malloc(&d_v, N * d);
    device_malloc(&d_kT, N * N);
    device_malloc(&d_p, N * N);device_malloc(&d_s, N * N);
    device_malloc(&d_o, N * d);device_malloc(&d_Wo, d * d);
    device_malloc(&d_out, N * d);
    // copy to device
    copy_to_device(&d_in, &h_in, N * d);
    copy_to_device(&d_Wq, &h_Wq, d * d);copy_to_device(&d_Wk, &h_Wk, d * d);copy_to_device(&d_Wv, &h_Wv, d * d);
    copy_to_device(&d_Wo, &h_Wo, d * d);

    FILE* file = fopen("output.csv", "a");

    attention(d_in,
        d_Wq, d_Wk, d_Wv,
        d_q, d_k, d_v,
        d_kT,
        d_p, d_s,
        d_o, d_Wo,
        d_out,
        N, d,
        kd, // optimized version, our method
        file
    );

    // copy to host
    copy_to_host(&h_q, &d_q, N * d);copy_to_host(&h_k, &d_k, N * d);copy_to_host(&h_v, &d_v, N * d);
    copy_to_host(&h_kT, &d_kT, N * N);
    copy_to_host(&h_p, &d_p, N * N);copy_to_host(&h_s, &d_s, N * N);
    copy_to_host(&h_o, &d_o, N * d);
    copy_to_host(&h_out, &d_out, N * d);

    
    // bool correct = check(h_in,
    //     h_Wq, h_Wk, h_Wv,
    //     h_q, h_k, h_v,
    //     h_p, h_s,
    //     h_o, h_Wo,
    //     h_out,
    //     N, d
    // );
    // if(correct == true) {
    //     printf("PASS\n");
    // }
    // else printf("FAILD\n");

    free(h_in);
    free(h_Wq);free(h_Wk);free(h_Wv);
    free(h_q);free(h_k);free(h_v);
    free(h_kT);
    free(h_p);free(h_s);
    free(h_o);free(h_Wo);
    free(h_out);

    cudaFree(d_in);
    cudaFree(d_Wq);cudaFree(d_Wk);cudaFree(d_Wv);
    cudaFree(d_q);cudaFree(d_k);cudaFree(d_v);
    cudaFree(d_kT);
    cudaFree(d_p);cudaFree(d_s);
    cudaFree(d_o);cudaFree(d_Wo);
    cudaFree(d_out);

    return 0;
}
