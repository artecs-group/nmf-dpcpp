#ifndef _COMMON_
#define _COMMON_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "cublas.h"
#include <sys/times.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

#define RANDOM
#define verbose 0

#ifdef REAL
	#define real float
#else
	#define real double
#endif

void printMATRIX(real *m, int I, int J);

#endif