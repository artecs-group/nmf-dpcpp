#ifndef _KERNELS_
#define _KERNELS_

#include "../common.hpp"

#if defined(CPU_DEVICE)	
#define accum accum_cpu
#else
#define accum accum_gpu
#endif

void W_mult_H(queue q, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Htras, int N, int M, int K);
void Wt_mult_WH(queue q, buffer<C_REAL, 1> &b_Haux, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_WH, int N, int M, int K);
void WH_mult_Ht(queue q, buffer<C_REAL, 1> &b_Waux, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_Htras, int N, int M, int K);
void adjust_WH(queue q, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Ht, int N, int M, int K);
void V_div_WH(queue q, buffer<C_REAL, 1> &b_V, buffer<C_REAL, 1> &b_WH, int N, int M);
void mult_M_div_vect(queue q, buffer<C_REAL, 1> &b_Mat, buffer<C_REAL, 1> &b_Maux, buffer<C_REAL, 1> &b_acc, int M, int K);
void accum_gpu(queue q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M);
void accum_cpu(queue q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M);

#endif