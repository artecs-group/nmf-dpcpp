#ifndef _COMMON_
#define _COMMON_

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <CL/sycl.hpp>

using namespace cl::sycl;

constexpr access::mode sycl_read               = access::mode::read;
constexpr access::mode sycl_write              = access::mode::write;
constexpr access::mode sycl_read_write         = access::mode::read_write;
constexpr access::mode sycl_discard_read_write = access::mode::discard_read_write;
constexpr access::mode sycl_discard_write      = access::mode::discard_write;

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

#define RANDOM
//#define DEBUG
const bool verbose = false;
const char PAD = 32;

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
const int NITER_TEST_CONV = 10;

/* Spacing of floating point numbers. */
const C_REAL eps = 2.2204e-16;

constexpr int split_factor = 2;
constexpr int N = VAR_N;
constexpr int M = VAR_M;
constexpr int K = VAR_K;
constexpr int N1 = (N / split_factor);
constexpr int N2 = N - N1;
constexpr int M1 = (M / split_factor);
constexpr int M2 = M - M1;

void adjust_WH(queue q, buffer<C_REAL, 1> b_W, buffer<C_REAL, 1> b_Ht, int N, int M, int K);
void V_div_WH(queue q, buffer<C_REAL, 1> b_V, buffer<C_REAL, 1> b_WH, int N, int M);
void mult_M_div_vect(queue q, buffer<C_REAL, 1> b_M, buffer<C_REAL, 1> b_Maux, buffer<C_REAL, 1> b_acc, int M, int K);
void accum(queue q, buffer<C_REAL, 1> b_acc, buffer<C_REAL, 1> b_X, int N, int M);
#endif