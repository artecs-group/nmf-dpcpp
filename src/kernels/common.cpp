#include "./common.h"

void adjust_WH(queue *q, buffer<real, 2> *b_W, buffer<real, 2> *b_Ht, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto W = b_W.get_access<sycl_read_write>();

        cgh.parallel_for<class check_W>(range<2>(N, K), [=](id <2> i){
            if(W[i] < eps)
                W[i] = eps;
        });
    });
	
    q.submit([&](handler& cgh) {
        auto Ht = b_Ht.get_access<sycl_read_write>();

        cgh.parallel_for<class check_Ht>(range<2>(M, K), [=](id <2> i){
            if(Ht[i] < eps)
                Ht[i] = eps;
        });
    });	 
}
