#include "./common.h"

void adjust_WH(queue q, buffer<C_REAL, 1> b_W, buffer<C_REAL, 1> b_Ht, int N, int M, int K) {
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


void V_div_WH(queue q, buffer<C_REAL, 1> b_V, buffer<C_REAL, 1> b_WH, int N, int M) {
    auto V = b_V.get_access<sycl_read>();
    auto WH = b_WH.get_access<sycl_read_write>();

    #pragma omp parallel for schedule(static)
    {
        for(int i = 0; i < N; i++){

            #pragma omp simd
            #pragma ivdep
            //#pragma vector nodynamic_align
            for(int j = 0; j < M; j++)
                WH[i*M + j] = V[i*M + j] / WH[i*M + j];
        }
    }
}


void mult_M_div_vect(queue q, buffer<C_REAL, 1> b_M, buffer<C_REAL, 1> b_Maux, buffer<C_REAL, 1> b_acc, int M, int K) {
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
