#include"queue_data.hpp"

queue_data::queue_data(int _N, int _N_split, int _M, int _M_split, int _K, device_selector selector) {
    q = queue{selector};
    
    N = _N;
    N_split = _N_split;
    M = _M;
    M_split = _M_split;
    K = _K;

    W                 = malloc_device<C_REAL>(N * K, q);
    Htras             = malloc_device<C_REAL>(M * K, q);
    WH_row            = malloc_device<C_REAL>(N_split * M, q);
    WH_col            = malloc_device<C_REAL>(N * M_split, q);
    V_row             = malloc_device<C_REAL>(N_split * M, q);
    V_col             = malloc_device<C_REAL>(N * M_split, q);
    Haux              = malloc_device<C_REAL>(M_split * K, q);
    Waux              = malloc_device<C_REAL>(N_split * K, q);
    accW              = malloc_device<C_REAL>(K, q);
    accH              = malloc_device<C_REAL>(K, q);
}

queue_data::~queue_data() {
    free(W, q);
    free(Htras, q);
    free(WH_row, q);
    free(WH_col, q);
    free(V_row, q);
    free(V_col, q);
    free(Haux, q);
    free(Waux, q);
    free(accW, q);
    free(accH, q);
}