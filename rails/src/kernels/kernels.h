#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "../common.hpp"

#ifdef REAL
	#define cublasRgemm cublasSgemm
	#define cublasRaxpy cublasSaxpy
#else
	#define cublasRgemm cublasDgemm
	#define cublasRaxpy cublasDaxpy
#endif

#define BLOCK_SIZE 32
//#define BLOCK_SIZE 16

void init_timers();
void delete_timers();
void W_mult_H(real *WH, real *W, real *Htras, int N, int M, int K);
void V_div_WH( real* V, real* WH, int ny, int nx);
void W_div_acc( real* W, real* accW, int ny, int nx);
void Wt_mult_WH( real *Haux, real *W, real *WH, int N, int M, int K);
void WH_mult_Ht( real *Waux, real *WH, real *Htras, int N, int M, int K);
void accum( real *acc, real* X, int n, int nx);
void mult_M_div_vect(real *M, real *Maux, real *acc, int ny, int nx);
void adjust_WH_GPU(real *W, real *Htras, int N, int M, int K);

#endif
