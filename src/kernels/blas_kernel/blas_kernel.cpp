#include "./blas_kernel.h"


void blas_W_mult_H(queue q, buffer<C_REAL, 1> b_WH, 
buffer<C_REAL, 1> b_W, buffer<C_REAL, 1> b_Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::gemm(q, transA, transB, M, N, K, 1, b_Htras, K, b_W, K, 0, b_WH, M);
    q.wait();
}


void blas_Wt_mult_WH(queue q, buffer<C_REAL, 1> b_Haux, buffer<C_REAL, 1> b_W,
buffer<C_REAL, 1> b_WH, int N, int M, int K) 
{
     oneapi::mkl::blas::gemm(q, transB, transA, K, M, N, 1, b_W, K, b_WH, M, 0, b_Haux, K);
     q.wait();
}


void blas_WH_mult_Ht(queue q, buffer<C_REAL, 1> b_Waux, buffer<C_REAL, 1> b_WH, 
buffer<C_REAL, 1> b_Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::gemm(q, transB, transB, K, N, M, 1, b_Htras, K, b_WH, M, 0, b_Waux, K);
    q.wait();
}
