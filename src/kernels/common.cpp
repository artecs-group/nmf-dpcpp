#include "./common.h"

range<3> get_range_in_3d(int data_size, int max_work_group_size) {
    int dim[3] = { 1, 1, 1 };

    for(int i = 0; i < 3; i++) {
        if(data_size <= max_work_group_size){
            dim[i] = data_size;
            break;
        }
        else
            dim[i] = max_work_group_size;

        data_size = (data_size + max_work_group_size - 1) / max_work_group_size;
    }

    return range<3>(dim[0], dim[1], dim[2]);
}


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
    const int R = 2;
    const int THREADS_PER_SUB_SLICE = 56; //Gen 9 and Gen 11 have 56, Gen 12 has 112
    int GROUP_SIZE;

    if(M >= THREADS_PER_SUB_SLICE)
        GROUP_SIZE = THREADS_PER_SUB_SLICE;
    else
        GROUP_SIZE = M;
    //device::get_info<cl::sycl::info::device::max_work_group_size>();

    q.submit([&](handler& cgh) {
        cgh.parallel_for<class V_div_WH>(nd_range(range(N, M), range(R, GROUP_SIZE)), [=](nd_item<2> item){
            int i = item.get_global_id(0);
            int j = item.get_global_id(1);

            WH[i*M + j] = sycl::native::divide(V[i*M + j], WH[i*M + j]);
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
    tree_reduction(q, acc, X, N, M);
    q.wait();

    // q.submit([&](handler& cgh) {
    //     cgh.parallel_for<class accum_add_matrix>(range<1>(M), [=](id <1> j){

    //         acc[j] = 0;
    //         for(int i = 0; i < N; i++)
    //             acc[j] += X[i*M + j];
    //     });
    // });
    // q.wait();
}


void tree_reduction(queue q, C_REAL *acc, C_REAL *X, int N, int M) {
    q.submit([&](auto &h) {
        sycl::range<3> work_group_range = get_range_in_3d(N, q.get_device().get_info<cl::sycl::info::device::max_work_group_size>());
        int fixed_N = work_group_range.get(0) * work_group_range.get(1) * work_group_range.get(2);
        int data_size = fixed_N * M;
        sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> scratch(fixed_N, h);

        h.parallel_for(sycl::nd_range(range(data_size, 1, 1), work_group_range), [=](sycl::nd_item<3> item) {
            int local_i = item.get_local_id(0);
            int local_j = item.get_local_id(1);
            int local_k = item.get_local_id(2);
            int local_id = local_i*work_group_range.get(1) + local_j*work_group_range.get(2) + local_k;

            int group_id = item.get_group(0);
            size_t global_id = (local_id + group_id) + (local_id * (M-1)); // offset

            if (global_id < N*M)
                scratch[local_id] = X[global_id];
            else
                scratch[local_id] = 0;

            // Do a tree reduction on items in work-group
            for (int i = N / 2; i > 0; i >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);

                if (local_id < i)
                    scratch[local_id] += scratch[local_id + i];
            }

            if (local_id == 0 && group_id < M)
                                // take into account if N was odd
                acc[group_id] = N % 2 == 0 ? scratch[0] : scratch[0] + scratch[N-1];
        });
    });
}