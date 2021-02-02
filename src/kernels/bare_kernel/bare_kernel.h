#ifndef _BARE_KERNEL_
#define _BARE_KERNEL_

#include "../common.h"

void bare_W_mult_H(queue &q, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Htras, int N, int M, int K);
void bare_accum(queue &q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M);
void bare_Wt_mult_WH(queue &q, buffer<C_REAL, 1> &b_Haux, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_WH, int N, int M, int K);
void bare_WH_mult_Ht(queue &q, buffer<C_REAL, 1> &b_Waux, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_Htras, int N, int M, int K);

#endif