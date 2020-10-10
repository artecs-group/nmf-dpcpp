#include "./common.h"

void adjust_WH(queue &q, buffer<real, 2> &b_W, buffer<real, 2> &b_Ht, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto W = b_W.get_access<sycl_read_write>();

        cgh.parallel_for<class check_W>(range<2>(N, K), [=](id <2> i){
            if(W[i] < eps)
                W[i] = eps;
        });
    });
	
    q.submit([&](handler& cgh) {
        auto Ht = b_Ht.get_access<sycl_read_write>();

        cgh.parallel_for<class check_Ht>(range<2>(M, K), [=](id <2> i){
            if(Ht[i] < eps)
                Ht[i] = eps;
        });
    });	 
}


void V_div_WH(queue &q, buffer<real, 2> &b_V, buffer<real, 2> &b_WH, int N, int M) {
    q.submit([&](handler& cgh) {
        auto V = b_V.get_access<sycl_read>();
        auto WH = b_WH.get_access<sycl_read_write>();

        cgh.parallel_for<class div_matrix>(range<2>(N, M), [=](id <2> i){
            WH[i] = V[i] / WH[i];
        });
    });
}


void mult_M_div_vect(queue &q, buffer<real, 2> &b_M, buffer<real, 2> &b_Maux, buffer<real, 1> &b_acc, int M, int K) {
    q.submit([&](handler& cgh) {
        auto M = b_M.get_access<sycl_read_write>();
        auto Maux = b_Maux.get_access<sycl_read>();
        auto acc = b_acc.get_access<sycl_read>();

        cgh.parallel_for<class mul_div_matrix>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            M[i][j] = M[i][j] * Maux[i][j] / acc[j];
        });
    });
}
