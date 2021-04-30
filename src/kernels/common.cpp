#include "./common.h"

void adjust_WH(queue q, C_REAL *W, C_REAL *Ht, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class check_W>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(W[i*K + j] < eps)
                W[i*K + j] = eps;
        });
    });

    q.submit([&](handler& cgh) {
        cgh.parallel_for<class check_Ht>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(Ht[i*K + j] < eps)
                Ht[i*K + j] = eps;
        });
    });
    q.wait();
}


void V_div_WH(queue q, C_REAL *V, C_REAL *WH, int N, int M) {
    const int R = 1;
    const int THREADS_PER_SUB_SLICE = 56; //Gen 9 and Gen 11 have 56, Gen 12 has 112
    int GROUP_SIZE;

    if(M >= THREADS_PER_SUB_SLICE)
        GROUP_SIZE = THREADS_PER_SUB_SLICE;
    else
        GROUP_SIZE = M;
    //device::get_info<cl::sycl::info::device::max_work_group_size>();

    q.submit([&](handler& cgh) {
        cgh.parallel_for<class V_div_WH>(nd_range(range(N, M), range(R, GROUP_SIZE)), [=](nd_item<2> item){
            int i = item.get_global_id(0);
            int j = item.get_global_id(1);

            WH[i*M + j] = sycl::native::divide(V[i*M + j], WH[i*M + j]);
        });
    });
    // q.submit([&](handler& cgh) {
    //     cgh.parallel_for<class V_div_WH>(range<1>(N), [=](id <1> ij){
    //         int i = ij[0];

    //         for(int j = 0; j < M; j++)
    //             WH[i*M + j] = V[i*M + j] / WH[i*M + j];
    //     });
    // });
    q.wait();
}


void mult_M_div_vect(queue q, C_REAL *Mat, C_REAL *Maux, C_REAL *acc, int M, int K) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class mul_M_div_vect>(range<1>(M), [=](id <1> ij){
            int i = ij[0];

            for(int j = 0; j < K; j++)
                Mat[i*K + j] = Mat[i*K + j] * Maux[i*K + j] / acc[j];
        });
    });
    q.wait();
}
