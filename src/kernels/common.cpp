#include "./common.h"

void adjust_WH(queue q, C_REAL *W, C_REAL *Ht, int N, int M, int K) {
#if defined(INTEL_IGPU_DEVICE)
	
#elif defined(CPU_DEVICE)

#endif

    for (int i = 0; i < N; i++)
        for (int j = 0; j < K; j++)
            if(W[i*K + j] < eps)
                W[i*K + j] = eps;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            if(Ht[i*K + j] < eps)
                Ht[i*K + j] = eps;
}


void V_div_WH(queue q, C_REAL *V, C_REAL *WH, int N, int M) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class V_div_WH>(range<1>(N), [=](id <1> ij){
            int i = ij[0];

            for(int j = 0; j < M; j++)
                WH[i*M + j] = V[i*M + j] / WH[i*M + j];
        });
    });
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
