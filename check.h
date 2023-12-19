#include "cuda_fp16.h"

bool check(half *in, 
    half *Wq, half *Wk,half *Wv,
    half *q, half *k, half *v,
    half *p, half *s,
    half *o, half *Wo,
    half *out,
    int N, int d) 
{
    printf("checking...\n");
    bool res = true;
    half *check_q;
    host_malloc(&check_q, N * d);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < d; j++) {
            check_q[i * d + j] = 0;
            for(int k = 0; k < d; k++) {
                check_q[i * d + j] += in[i * d + k] * Wq[k * d + j];
            }
        }
    }
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < d; j++) {
            if(fabs((float)check_q[i * d+ j] - (float)q[i * d + j])>1e2) {
                res = false;
                printf("i = %d, j = %d, check_q = %f, q = %f\n", i, j, (float)check_q[i * d+ j], (float)q[i * d + j]);
                goto A;
            }
        }
    }


    A:
    return res;
}