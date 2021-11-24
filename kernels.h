#ifndef _KERNELS_H_
#define _KERNELS_H_

#if REAL <=4
	#define real float
	#define cublasRgemm cublasSgemm
	#define cublasRaxpy cublasSaxpy
#else
	#define real double
	#define cublasRgemm cublasDgemm
	#define cublasRaxpy cublasDaxpy
#endif

//#define BLOCK_SIZE 32
#define BLOCK_SIZE 16

/* Spacing of realing point numbers. */
#define EPS 2.2204e-16

#ifdef __cplusplus
extern "C"
#endif
void W_mult_H(real *WH, real *W, real *Htras, int N, int M, int K);

#ifdef __cplusplus
extern "C"
#endif
void V_div_WH( real* V, real* WH, int ny, int nx);

#ifdef __cplusplus
extern "C"
#endif
void W_div_acc( real* W, real* accW, int ny, int nx);

#ifdef __cplusplus
extern "C"
#endif
void Wt_mult_WH( real *Haux, real *W, real *WH, int N, int M, int K);

#ifdef __cplusplus
extern "C"
#endif
void WH_mult_Ht( real *Waux, real *WH, real *Htras, int N, int M, int K);

#ifdef __cplusplus
extern "C"
#endif
void accum( real *acc, real* X, int n, int nx);

#ifdef __cplusplus
extern "C"
#endif
void mult_M_div_vect(real *M, real *Maux, real *acc, int ny, int nx);

#ifdef __cplusplus
extern "C"
#endif
void adjust_WH_GPU(real *W, real *Htras, int N, int M, int K);


#ifdef __cplusplus
extern "C"
#endif
void W_mult_S( real *WS, real *W, real *S, int N, int K);

#ifdef __cplusplus
extern "C"
#endif
void S_mult_H( real *SHtras, real *S, real *Htras, int M, int K);

#endif
