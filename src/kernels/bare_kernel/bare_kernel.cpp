#include "./bare_kernel.h"


void W_mult_H(queue *q, buffer<real, 2> *b_WH, buffer<real, 2> *b_W, buffer<real, 2> *b_Htras, int N, int M, int K) {
        q.submit([&](handler& cgh) {
        auto WH = b_WH.get_access<sycl_read_write>();
        auto W = b_W.get_access<sycl_read>();
        auto Htras = b_Htras.get_access<sycl_read>();

        cgh.parallel_for<class matrix_mul>(range<2>(N, M), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            for(int k = 0; k < K; k++)
                WH[i][j] += W[i][k] * Htras[j][k];
        });
    });
}


void accum(queue *q, buffer<real, 1> *b_acc, buffer<real, 2> *b_X, int N, int M) {
    q.submit([&](handler& cgh) {
        auto acc = b_acc.get_access<sycl_write>();
        auto W = b_W.get_access<sycl_read>();

        cgh.parallel_for<class init_0>(range<1>(M), [=](id <1> i){
            acc[i] = W[0][i];
        });
    });

    q.submit([&](handler& cgh) {
        auto acc = b_acc.get_access<sycl_write>();
        auto W = b_W.get_access<sycl_read>();

        cgh.parallel_for<class add_matrix>(range<2>(N-1, M), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            acc[j] += W[i+1][j];
        });
    });
}


void Wt_mult_WH(queue *q, buffer<real, 2> *b_Haux, buffer<real, 2> *b_W, buffer<real, 2> *b_WH, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto Haux = b_Haux.get_access<sycl_write>();

        cgh.parallel_for<class init_matrix>(range<2>(M, K), [=](id <2> i){
            Haux[i] = 0.0;
        });
    });

    q.submit([&](handler& cgh) {
        auto Haux = b_Haux.get_access<sycl_read_write>();
        auto W = b_W.get_access<sycl_read>();
        auto WH = b_WH.get_access<sycl_read>();

        cgh.parallel_for<class matrix_mul>(range<3>(N, K, M), [=](id <3> kij){
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];

            Haux[j][i] += W[k][i] * WH[k][j];
        });
    });
}


void WH_mult_Ht(queue *q, buffer<real, 2> *b_Waux, buffer<real, 2> *b_WH, buffer<real, 2> *b_Htras, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto Waux = b_Waux.get_access<sycl_read_write>();
        auto WH = b_WH.get_access<sycl_read>();
        auto Htras = b_Htras.get_access<sycl_read>();

        cgh.parallel_for<class matrix_mul>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            Waux[i][j] = 0.0;

            for(int k = 0; k < M; k++)
                Waux[j][i] += WH[k][i] * Htras[k][j];
        });
    });
}
