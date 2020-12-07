#include "./blas_kernel.h"


void blas_W_mult_H(queue &q, buffer<Real, 1> &b_WH, 
buffer<Real, 1> &b_W, buffer<Real, 1> &b_Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::gemm(q, transA, transB, M, N, K, 1, b_Htras, K, b_W, K, 0, b_WH, M);
}


void blas_accum(queue &q, buffer<Real, 1> &b_acc, buffer<Real, 1> &b_X, int N, int M) {
    oneapi::mkl::blas::axpy(q, N, -1.0, b_acc, 1, b_acc, 1);

    q.submit([&](handler& cgh) {
        auto acc = b_acc.get_access<sycl_read_write>(cgh);
        auto X = b_X.get_access<sycl_read>(cgh);

        cgh.parallel_for<class accum_add_matrix>(range<1>(M), [=](id <1> j){
            for(int i = 0; i < N; i++)
                acc[j] += X[i*M + j];
        });
    });
}


void blas_Wt_mult_WH(queue &q, buffer<Real, 1> &b_Haux, buffer<Real, 1> &b_W,
buffer<Real, 1> &b_WH, int N, int M, int K) 
{
     oneapi::mkl::blas::gemm(q, transB, transA, K, M, N, 1, b_W, K, b_WH, M, 0, b_Haux, K);
}


void blas_WH_mult_Ht(queue &q, buffer<Real, 1> &b_Waux, buffer<Real, 1> &b_WH, 
buffer<Real, 1> &b_Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::gemm(q, transB, transB, K, N, M, 1, b_Htras, K, b_WH, M, 0, b_Waux, K);
}
