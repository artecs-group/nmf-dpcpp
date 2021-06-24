#include "./blas_kernel.h"

constexpr oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;


void blas_W_mult_H(queue q, C_REAL* WH, 
C_REAL* W, C_REAL* Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::gemm(q, transA, transB, M, N, K, 1, Htras, K, W, K, 0, WH, M);
}


void blas_Wt_mult_WH(queue q, C_REAL* Haux, C_REAL* W,
C_REAL* WH, int N, int M, int K) 
{
     oneapi::mkl::blas::gemm(q, transB, transA, K, M, N, 1, W, K, WH, M, 0, Haux, K);
}


void blas_WH_mult_Ht(queue q, C_REAL* Waux, C_REAL* WH, 
C_REAL* Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::gemm(q, transB, transB, K, N, M, 1, Htras, K, WH, M, 0, Waux, K);
}
