#ifndef _BLAS_KERNEL_
#define _BLAS_KERNEL_

#include "../common.h"
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

constexpr oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

void blas_W_mult_H(queue &q, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Htras, int N, int M, int K);
void blas_init_accum(queue &q, buffer<C_REAL, 1> &b_acc, int N);
void blas_accum(queue &q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M, int offset_M);
void blas_Wt_mult_WH(queue &q, buffer<C_REAL, 1> &b_Haux, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_WH, int N, int M, int K);
void blas_WH_mult_Ht(queue &q, buffer<C_REAL, 1> &b_Waux, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_Htras, int N, int M, int K);

#endif
