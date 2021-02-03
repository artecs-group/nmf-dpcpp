#include "./common.h"

void adjust_WH(queue &q, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Ht, int N, int M, int K, int offsetN, int offsetM) {
    q.submit([&](handler& cgh) {
        auto W = b_W.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class check_W>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(W[(i+offsetN)*K + j] < eps)
                W[(i+offsetN)*K + j] = eps;
        });
    });
	
    q.submit([&](handler& cgh) {
        auto Ht = b_Ht.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class check_Ht>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(Ht[(i+offsetM)*K + j] < eps)
                Ht[(i+offsetM)*K + j] = eps;
        });
    });	 
}


void V_div_WH(queue &q, buffer<C_REAL, 1> &b_V, buffer<C_REAL, 1> &b_WH, int N, int M, int offsetN) {
    q.submit([&](handler& cgh) {
        auto V = b_V.get_access<sycl_read>(cgh);
        auto WH = b_WH.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class V_div_WH>(range<2>(N, M), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            WH[(i+offsetN)*M + j] = V[(i+offsetN)*M + j] / WH[(i+offsetN)*M + j];
        });
    });
}


void mult_M_div_vect(queue &q, buffer<C_REAL, 1> &b_M, buffer<C_REAL, 1> &b_Maux, buffer<C_REAL, 1> &b_acc, int M, int K, int offsetM) {
    q.submit([&](handler& cgh) {
        auto Mat = b_M.get_access<sycl_read_write>(cgh);
        auto Maux = b_Maux.get_access<sycl_read>(cgh);
        auto acc = b_acc.get_access<sycl_read>(cgh);

        cgh.parallel_for<class mul_M_div_vect>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            Mat[(i+offsetM)*K + j] = Mat[(i+offsetM)*K + j] * Maux[(i+offsetM)*K + j] / acc[j];
        });
    });
}
