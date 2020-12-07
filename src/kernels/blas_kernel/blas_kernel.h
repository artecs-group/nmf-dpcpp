#ifndef _BLAS_KERNEL_
#define _BLAS_KERNEL_

#include "../common.h"
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
//#include "mkl_blas_sycl.hpp" /* To be included in version <beta0.9 */

constexpr oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

void blas_W_mult_H(queue &q, buffer<Real, 1> &b_WH, buffer<Real, 1> &b_W, buffer<Real, 1> &b_Htras, int N, int M, int K);
void blas_accum(queue &q, buffer<Real, 1> &b_acc, buffer<Real, 1> &b_X, int N, int M);
void blas_Wt_mult_WH(queue &q, buffer<Real, 1> &b_Haux, buffer<Real, 1> &b_W, buffer<Real, 1> &b_WH, int N, int M, int K);
void blas_WH_mult_Ht(queue &q, buffer<Real, 1> &b_Waux, buffer<Real, 1> &b_WH, buffer<Real, 1> &b_Htras, int N, int M, int K);

#endif
