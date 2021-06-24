#include "kernels/common.h"

queue_data::queue_data(int _N, int _N_split, int _M, int _M_split, int _K, std::string device_name) {
    if(!device_name.compare(std::string("IntelGPU"))){
        IntelGPUSelector selector{};
        q = queue(selector, property::queue::in_order());
    }
    else {
        cpu_selector selector{};
        q = queue(selector, property::queue::in_order());
    }
    
    N = _N;
    N_split = _N_split;
    M = _M;
    M_split = _M_split;
    K = _K;

    W                 = malloc_shared<C_REAL>(N * K, q);
    Htras             = malloc_shared<C_REAL>(M * K, q);
    WH_row            = malloc_device<C_REAL>(N_split * M, q);
    WH_col            = malloc_device<C_REAL>(N * M_split, q);
    V_row             = malloc_shared<C_REAL>(N_split * M, q);
    V_col             = malloc_shared<C_REAL>(N * M_split, q);
    Haux              = malloc_device<C_REAL>(M_split * K, q);
    Waux              = malloc_device<C_REAL>(N_split * K, q);
    accW              = malloc_device<C_REAL>(K, q);
    accH              = malloc_device<C_REAL>(K, q);
}

queue_data::~queue_data() {
    sycl::free(W, q);
    sycl::free(Htras, q);
    sycl::free(WH_row, q);
    sycl::free(WH_col, q);
    sycl::free(V_row, q);
    sycl::free(V_col, q);
    sycl::free(Haux, q);
    sycl::free(Waux, q);
    sycl::free(accW, q);
    sycl::free(accH, q);
}