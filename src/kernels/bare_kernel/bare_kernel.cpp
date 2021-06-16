#include "./bare_kernel.h"


void bare_W_mult_H(queue q, C_REAL* WH, C_REAL* W, C_REAL* Htras, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class W_mul_H>(range<2>(N, M), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            WH[i*M + j] = 0.0;

            for(int k = 0; k < K; k++)
                WH[i*M + j] += W[i*K + k] * Htras[j*K + k];
        });
    });
}


void bare_Wt_mult_WH(queue q, C_REAL* Haux, C_REAL* W, C_REAL* WH, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class Wt_mul_WH>(range<2>(K, M), [=](id <2> jk){
            int j = jk[0];
            int k = jk[1];

            Haux[k*K + j] = 0.0;

            for(int i = 0; i < N; i++)
                Haux[k*K + j] += W[i*K + j] * WH[i*M + k];
        });
    });
}


void bare_WH_mult_Ht(queue q, C_REAL* Waux, C_REAL* WH, C_REAL* Htras, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class matrix_mul_sum>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            Waux[i*K + j] = 0.0;

            for(int k = 0; k < M; k++)
                Waux[i*K + j] += WH[i*M + k] * Htras[k*K + j];
        });
    });
}
