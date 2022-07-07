#ifndef _COMMON_
#define _COMMON_

#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifdef NVIDIA_GPU_DEVICE
#include "cublas.h"
#else
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

#define RANDOM
//#define DEBUG

#ifdef REAL_S
#define C_REAL float
#else
#define C_REAL double
#endif

#ifdef NVIDIA_GPU_DEVICE
    #ifdef REAL_S
    #define cblas_rgemm cublasSgemm
    #else
    #define cblas_rgemm cublasDgemm
    #endif
#else
    #ifdef REAL_S
    #define cblas_rgemm cblas_sgemm
    #else
    #define cblas_rgemm cblas_dgemm
    #endif
#endif

#ifdef GPU_DEVICE
#define nmf gpu_nmf
#else
#define nmf cpu_nmf
#endif

void gpu_nmf(int deviceId, int niter, C_REAL *V, C_REAL *WH, 
	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
	C_REAL *acumm_W, C_REAL *acumm_H, int N, int M, int K);

void cpu_nmf(int deviceId, int niter, C_REAL *V, C_REAL *WH, 
	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
	C_REAL *acumm_W, C_REAL *acumm_H, int N, int M, int K);

#endif