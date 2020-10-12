#include "./bare_kernel.h"


void W_mult_H(queue &q, buffer<real, 1> &b_WH, buffer<real, 1> &b_W, buffer<real, 1> &b_Htras, int N, int M, int K) {
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


void accum(queue &q, buffer<real, 1> &b_acc, buffer<real, 1> &b_X, int N, int M) {
    q.submit([&](handler& cgh) {
        auto acc = b_acc.get_access<sycl_write>(cgh);
        auto X = b_X.get_access<sycl_read>(cgh);

        cgh.parallel_for<class accum_init_0>(range<1>(M), [=](id <1> i){
            acc[i] = X[i];
        });
    });

    q.submit([&](handler& cgh) {
        auto acc = b_acc.get_access<sycl_write>(cgh);
        auto X = b_X.get_access<sycl_read>(cgh);

        cgh.parallel_for<class accum_add_matrix>(range<2>(N-1, M), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            acc[j] += X[(i+1)*M + j];
        });
    });
}


void Wt_mult_WH(queue &q, buffer<real, 1> &b_Haux, buffer<real, 1> &b_W, buffer<real, 1> &b_WH, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto Haux = b_Haux.get_access<sycl_write>(cgh);

        cgh.parallel_for<class init_Haux>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            Haux[i*K + j] = 0.0;
        });
    });

    q.submit([&](handler& cgh) {
        auto Haux = b_Haux.get_access<sycl_read_write>(cgh);
        auto W = b_W.get_access<sycl_read>(cgh);
        auto WH = b_WH.get_access<sycl_read>(cgh);

        cgh.parallel_for<class Wt_mul_WH>(range<3>(N, K, M), [=](id <3> ijk){
            int i = ijk[0];
            int j = ijk[1];
            int k = ijk[2];

            Haux[k*K + j] += W[i*K + j] * WH[i*M + k];
        });
    });
}


void WH_mult_Ht(queue &q, buffer<real, 1> &b_Waux, buffer<real, 1> &b_WH, buffer<real, 1> &b_Htras, int N, int M, int K) {
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
