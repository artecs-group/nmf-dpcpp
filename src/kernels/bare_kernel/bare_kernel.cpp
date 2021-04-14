#include "./bare_kernel.h"


void bare_W_mult_H(queue &q, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Htras, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto WH = b_WH.get_access<sycl_read_write>(cgh);
        auto W = b_W.get_access<sycl_read>(cgh);
        auto Htras = b_Htras.get_access<sycl_read>(cgh);

        cgh.parallel_for<class W_mul_H>(range<2>(N, M), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            WH[i*M + j] = 0.0;

            for(int k = 0; k < K; k++)
                WH[i*M + j] += W[i*K + k] * Htras[j*K + k];
        });
    });
}


void bare_accum(queue &q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M) {
    q.submit([&](handler& cgh) {
        auto acc = b_acc.get_access<sycl_read_write>(cgh);
        auto X = b_X.get_access<sycl_read>(cgh);

        cgh.parallel_for<class accum_add_matrix>(range<1>(M), [=](id <1> j){

            acc[j] = 0;
            for(int i = 0; i < N; i++)
                acc[j] += X[i*M + j];
        });
    });
}


void bare_Wt_mult_WH(queue &q, buffer<C_REAL, 1> &b_Haux, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_WH, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto Haux = b_Haux.get_access<sycl_read_write>(cgh);
        auto W = b_W.get_access<sycl_read>(cgh);
        auto WH = b_WH.get_access<sycl_read>(cgh);

        cgh.parallel_for<class Wt_mul_WH>(range<2>(K, M), [=](id <2> jk){
            int j = jk[0];
            int k = jk[1];

            Haux[k*K + j] = 0.0;

            for(int i = 0; i < N; i++)
                Haux[k*K + j] += W[i*K + j] * WH[i*M + k];
        });
    });
}


void bare_WH_mult_Ht(queue &q, buffer<C_REAL, 1> &b_Waux, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_Htras, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto Waux = b_Waux.get_access<sycl_read_write>(cgh);
        auto WH = b_WH.get_access<sycl_read>(cgh);
        auto Htras = b_Htras.get_access<sycl_read>(cgh);
        
        cgh.parallel_for<class matrix_mul_sum>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            Waux[i*K + j] = 0.0;

            for(int k = 0; k < M; k++)
                Waux[i*K + j] += WH[i*M + k] * Htras[k*K + j];
        });
    });
}
