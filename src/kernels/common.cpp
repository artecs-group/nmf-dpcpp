#include "./common.h"

void adjust_WH(queue q, C_REAL *W, C_REAL *Ht, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class check_W>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(W[i*K + j] < eps)
                W[i*K + j] = eps;
        });
    });

    q.submit([&](handler& cgh) {
        cgh.parallel_for<class check_Ht>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(Ht[i*K + j] < eps)
                Ht[i*K + j] = eps;
        });
    });
    q.wait();
}


void V_div_WH(queue q, C_REAL *V, C_REAL *WH, int N, int M) {
    int max_work_group_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int GROUP_SIZE = max_work_group_size < N ? max_work_group_size : N;
    // adjust work-groups number 
    int remainder = (N == GROUP_SIZE) ? 0 : GROUP_SIZE - (N % GROUP_SIZE);

    q.submit([&](handler& cgh) {
        cgh.parallel_for<class V_div_WH>(nd_range(range((N+remainder) * M), range(GROUP_SIZE)), [=](nd_item<1> item){
            int i = item.get_global_id(0);
            // TODO: add offset to WH and V to remove if-then-else
            if(i < N*M)
                WH[i] = sycl::native::divide(V[i], WH[i]);
        });
    });
    // q.submit([&](handler& cgh) {
    //     cgh.parallel_for<class V_div_WH>(range<1>(N), [=](id <1> ij){
    //         int i = ij[0];

    //         for(int j = 0; j < M; j++)
    //             WH[i*M + j] = V[i*M + j] / WH[i*M + j];
    //     });
    // });
    q.wait();
}


void mult_M_div_vect(queue q, C_REAL *Mat, C_REAL *Maux, C_REAL *acc, int M, int K) {
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class mul_M_div_vect>(range<1>(M), [=](id <1> ij){
            int i = ij[0];

            for(int j = 0; j < K; j++)
                Mat[i*K + j] = Mat[i*K + j] * Maux[i*K + j] / acc[j];
        });
    });
    q.wait();
}


void accum(queue q, C_REAL *acc, C_REAL *X, int N, int M) {
    // q.submit([&](handler& cgh) {
    //     cgh.parallel_for<class accum_add_matrix>(range<1>(M), [=](id <1> j){

    //         acc[j] = 0;
    //         for(int i = 0; i < N; i++)
    //             acc[j] += X[i*M + j];
    //     });
    // });
    // q.wait();

    // init acc
    q.submit([&](auto &h) {
        h.parallel_for(sycl::range<1>(M), [=](id <1> i) {
            acc[i] = 0;
        });
    });
    q.wait();
    
    int max_work_group_size = 2;//q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int fixed_N = max_work_group_size < N ? max_work_group_size : N;
    int data_size = fixed_N * M;
    int j = 0;

    for(int i = 0; i < N; i+=fixed_N){
        q.submit([&](auto &h) {
            int offset = data_size*j;
            sycl::accessor<C_REAL, 1, sycl::access::mode::read_write, sycl::access::target::local> scratch(fixed_N, h);

            h.parallel_for(sycl::nd_range(range(data_size), range(fixed_N)), [=](sycl::nd_item<1> item) {
                int local_id = item.get_local_id(0);
                int group_id = item.get_group(0);
                size_t global_id = local_id * (M-1) + local_id + group_id + offset;

                // TODO: add padding to X in order to remove if-then-else
                if (global_id < N*M)
                    scratch[local_id] = X[global_id];
                else
                    scratch[local_id] = 0;

                // Do a tree reduction on items in work-group
                for (int i = fixed_N / 2; i > 0; i >>= 1) {
                    item.barrier(sycl::access::fence_space::local_space);

                    if (local_id < i)
                        scratch[local_id] += scratch[local_id + i];
                }

                if (local_id == 0)
                    // take into account if N was odd
                    // TODO: add padding to scratch in order to remove if-then-else
                    acc[group_id] += fixed_N % 2 == 0 ? scratch[0] : scratch[0] + scratch[fixed_N-1];
            });
        });
        q.wait();
        j++;
    }
}