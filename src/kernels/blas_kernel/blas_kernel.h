#ifndef _BLAS_KERNEL_
#define _BLAS_KERNEL_

#include "../common.h"
#include "mkl.h"
#include "mkl_blas_sycl.hpp"

constexpr oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

void W_mult_H(queue &q, buffer<Real, 1> &b_WH, buffer<Real, 1> &b_W, buffer<Real, 1> &b_Htras, int N, int M, int K);
void accum(queue &q, buffer<Real, 1> &b_acc, buffer<Real, 1> &b_X, int N, int M);
void Wt_mult_WH(queue &q, buffer<Real, 1> &b_Haux, buffer<Real, 1> &b_W, buffer<Real, 1> &b_WH, int N, int M, int K);
void WH_mult_Ht(queue &q, buffer<Real, 1> &b_Waux, buffer<Real, 1> &b_WH, buffer<Real, 1> &b_Htras, int N, int M, int K);

#endif