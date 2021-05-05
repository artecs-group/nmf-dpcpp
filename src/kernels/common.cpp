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
    // int data_size = N*M;
    // int work_groups = M;
    // int inner_work_groups = 10;
    // int work_group_size = work_groups * inner_work_groups;
    // int inner_work_group_size = (int) ceil(data_size / work_group_size);
    // C_REAL *aux_acc = malloc_device<C_REAL>(inner_work_group_size, q);

    // q.submit([&](auto &h) {
    //     // auto sumr = sycl::ONEAPI::reduction(aux_acc, sycl::ONEAPI::plus<>());
    //     // h.parallel_for(sycl::nd_range<1>{data_size, inner_work_group_size}, sumr, [=](sycl::nd_item<1> item, auto &sumr_arg) {
    //     //     int global_id = item.get_global_id(0);
    //     //     sumr_arg += X[global_id];
    //     // });
    //     sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> scratch(inner_work_group_size, h);

    //     h.parallel_for(sycl::nd_range<1>(data_size, inner_work_group_size), [=](sycl::nd_item<1> item) {
    //         size_t global_id = item.get_global_id(0);
    //         int local_id = item.get_local_id(0);
    //         int group_id = item.get_group(0);

    //         if (global_id < data_size)
    //             scratch[local_id] = X[global_id];
    //         else
    //             scratch[local_id] = 0;

    //         // Do a tree reduction on items in work-group
    //         for (int i = inner_work_group_size / 2; i > 0; i >>= 1) {
    //             item.barrier(sycl::access::fence_space::local_space);

    //             if (local_id < i)
    //                 scratch[local_id] += scratch[local_id + i];
    //         }

    //         if (local_id == 0)
    //             aux_acc[group_id] = scratch[0];
    //     });
    // });
    // q.wait();

    // q.submit([&](auto &h) {
    //     // auto sumr = sycl::ONEAPI::reduction(acc, sycl::ONEAPI::plus<>());
    //     // h.parallel_for(sycl::nd_range<1>{inner_work_group_size, work_group_size}, sumr, [=](sycl::nd_item<1> item, auto &sumr_arg) {
    //     //     int global_id = item.get_global_id(0);
    //     //     sumr_arg += aux_acc[global_id];
    //     // });
    //     sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> scratch(work_group_size, h);

    //     h.parallel_for(sycl::nd_range<1>(inner_work_group_size, work_group_size), [=](sycl::nd_item<1> item) {
    //         size_t global_id = item.get_global_id(0);
    //         int local_id = item.get_local_id(0);
    //         int group_id = item.get_group(0);

    //         if (global_id < inner_work_group_size)
    //             scratch[local_id] = aux_acc[global_id];
    //         else
    //             scratch[local_id] = 0;

    //         // Do a tree reduction on items in work-group
    //         for (int i = work_group_size / 2; i > 0; i >>= 1) {
    //             item.barrier(sycl::access::fence_space::local_space);

    //             if (local_id < i)
    //                 scratch[local_id] += scratch[local_id + i];
    //         }

    //         if (local_id == 0)
    //             acc[group_id] = scratch[0];
    //     });
    // });

    //     h.parallel_for(num_work_items, [=](auto index) {
    //         size_t glob_id = index[0];
    //         C_REAL sum = 0;

    //         for (size_t i = glob_id; i < data_size; i += num_work_items)
    //             sum += X[i];

    //         acc[glob_id] = sum;
    //     });
    // });
    q.submit([&](handler& cgh) {
        cgh.parallel_for<class accum_add_matrix>(range<1>(M), [=](id <1> j){

            acc[j] = 0;
            for(int i = 0; i < N; i++)
                acc[j] += X[i*M + j];
        });
    });
    q.wait();
}
