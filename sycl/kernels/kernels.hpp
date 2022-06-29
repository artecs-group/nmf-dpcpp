#ifndef _KERNELS_
#define _KERNELS_

#include "../common.hpp"

#if defined(CPU_DEVICE)	
#define accum accum_cpu
#else
#define accum accum_gpu
#endif

void W_mult_H(queue q, C_REAL* WH, C_REAL* W, C_REAL* Htras, int N, int M, int K);
void Wt_mult_WH(queue q, C_REAL* Haux, C_REAL* W, C_REAL* WH, int N, int M, int K);
void WH_mult_Ht(queue q, C_REAL* Waux, C_REAL* WH, C_REAL* Htras, int N, int M, int K);
void adjust_WH(queue q, C_REAL* W, C_REAL* Ht, int N, int M, int K);
void V_div_WH(queue q, C_REAL* V, C_REAL* WH, int N, int M);
void mult_M_div_vect(queue q, C_REAL* Mat, C_REAL* Maux, C_REAL* acc, int M, int K);
void accum_gpu(queue q, C_REAL* acc, C_REAL* X, int N, int M);
void accum_cpu(queue q, C_REAL* acc, C_REAL* X, int N, int M);

#endif