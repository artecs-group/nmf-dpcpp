#ifndef _COMMON_
#define _COMMON_

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

#define RANDOM
//#define DEBUG
#define PAD 32

#ifdef REAL_S
#define C_REAL float
#else
#define C_REAL double
#endif

#ifdef BLAS_KERNEL
#define W_mult_H blas_W_mult_H
#define Wt_mult_WH blas_Wt_mult_WH
#define WH_mult_Ht blas_WH_mult_Ht
#else
#define W_mult_H bare_W_mult_H
#define Wt_mult_WH bare_Wt_mult_WH
#define WH_mult_Ht bare_WH_mult_Ht
#endif

/* Number of iterations before testing convergence (can be adjusted) */
#define NITER_TEST_CONV 10

// CUDA device selector
class CUDASelector : public device_selector {
    public:
        int operator()(const device &Device) const override {
            const std::string DriverVersion = Device.get_info<info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                return 1;

            return 0;
        }
};

// Intel iGPU
class IntelGPUSelector : public device_selector {
    public:
        int operator()(const device &Device) const override {
            const std::string vendor = Device.get_info<info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
                return 1;

            return 0;
        }
};


class queue_data {
    public:
        queue q;
        int N, N_split, M, M_split, K;
        C_REAL *V_row, *V_col, *W, *Htras, *WH_row, *WH_col, *Haux, *Waux, *accH, *accW;

        queue_data(int _N, int _N_split, int _M, int _M_split, int _K, queue _q);
        ~queue_data();
};

void adjust_WH(queue q, C_REAL* W, C_REAL* Ht, int N, int M, int K);
void V_div_WH(queue q, C_REAL* V, C_REAL* WH, int N, int M);
void mult_M_div_vect(queue q, C_REAL* Mat, C_REAL* Maux, C_REAL* acc, int M, int K);
void accum(queue q, C_REAL* acc, C_REAL* X, int N, int M);
void copy_matrix_to(queue q, C_REAL* Mat, C_REAL* dMat, int N, int M);
void copy_matrix_from(queue q, C_REAL* Mat, C_REAL* dMat, int N, int M);
void sync_queues(int queues, queue_data* qd);

#endif