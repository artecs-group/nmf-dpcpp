#ifndef _COMMON_
#define _COMMON_

#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mkl.h"
#include <omp.h>
#include "mkl_omp_offload.h"

#define RANDOM
//#define DEBUG
const bool verbose = false;
const char PAD = 32;
//static int HW_SPECIFIC_ADVICE_RO = 0;

#ifdef REAL_S
#define C_REAL float
#else
#define C_REAL double
#endif

/* Number of iterations before testing convergence (can be adjusted) */
const int NITER_TEST_CONV = 10;

/* Spacing of floating point numbers. */
const C_REAL eps = 2.2204e-16;

#endif