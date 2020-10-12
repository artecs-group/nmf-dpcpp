#include "./common.h"

void adjust_WH(queue &q, buffer<real, 1> &b_W, buffer<real, 1> &b_Ht, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto W = b_W.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class check_W>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(W[i*K + j] < eps)
                W[i*K + j] = eps;
        });
    });
	
    q.submit([&](handler& cgh) {
        auto Ht = b_Ht.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class check_Ht>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(Ht[i*K + j] < eps)
                Ht[i*K + j] = eps;
        });
    });	 
}


void V_div_WH(queue &q, buffer<real, 1> &b_V, buffer<real, 1> &b_WH, int N, int M) {
    q.submit([&](handler& cgh) {
        auto V = b_V.get_access<sycl_read>(cgh);
        auto WH = b_WH.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class V_div_WH>(range<2>(N, M), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            WH[i*M + j] = V[i*M + j] / WH[i*M + j];
        });
    });
}


void mult_M_div_vect(queue &q, buffer<real, 1> &b_M, buffer<real, 1> &b_Maux, buffer<real, 1> &b_acc, int M, int K) {
    q.submit([&](handler& cgh) {
        auto Mat = b_M.get_access<sycl_read_write>(cgh);
        auto Maux = b_Maux.get_access<sycl_read>(cgh);
        auto acc = b_acc.get_access<sycl_read>(cgh);

        cgh.parallel_for<class mul_M_div_vect>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            Mat[i*K + j] = Mat[i*K + j] * Maux[i*K + j] / acc[j];
        });
    });
}
