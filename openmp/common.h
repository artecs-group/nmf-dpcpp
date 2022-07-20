#ifndef _COMMON_
#define _COMMON_

#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include "mkl.h"
#include <omp.h>
#include "mkl_omp_offload.h"

#define RANDOM
//#define DEBUG

#ifdef REAL_S
#define C_REAL float
#define cblas_rgemm cblas_sgemm
#else
#define C_REAL double
#define cblas_rgemm cblas_dgemm
#endif

#if defined(NVIDIA_GPU_DEVICE)
	#define nmf nvidia_nmf
#elif defined(INTEL_GPU_DEVICE)
    #define nmf intel_gpu_nmf
#else
    #define nmf cpu_nmf
#endif

void cpu_nmf(int niter, C_REAL *V, C_REAL *WH, 
	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
	C_REAL *acumm_W, C_REAL *acumm_H, int N, int M, int K);

void nvidia_nmf(int deviceId, int niter, C_REAL *V, C_REAL *WH, 
	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
	C_REAL *acumm_W, C_REAL *acumm_H, int N, int M, int K);

void intel_gpu_nmf(int deviceId, int niter, C_REAL *V, C_REAL *WH, 
	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
	C_REAL *acumm_W, C_REAL *acumm_H, int N, int M, int K);

#endif