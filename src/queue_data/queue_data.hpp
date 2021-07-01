#ifndef _QUEUE_DATA_
#define _QUEUE_DATA_

#include "../common.hpp"

class queue_data {
    public:
        queue q;
        int N, N_split, M, M_split, K;
        C_REAL *V_row, *V_col, *W, *Htras, *WH_row, *WH_col, *Haux, *Waux, *accH, *accW;

        queue_data(void) {};
        queue_data(int _N, int _N_split, int _M, int _M_split, int _K, std::string device_name);
        ~queue_data();
};

#endif