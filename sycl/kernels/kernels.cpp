#include "./kernels.hpp"

constexpr oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

/* Spacing of floating point numbers. */
constexpr C_REAL eps{2.2204e-16};

void W_mult_H(queue q, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::column_major::gemm(q, transA, transB, M, N, K, 1, b_Htras, K, b_W, K, 0, b_WH, M);
    q.wait();
}


void Wt_mult_WH(queue q, buffer<C_REAL, 1> &b_Haux, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_WH, int N, int M, int K) 
{
     oneapi::mkl::blas::column_major::gemm(q, transB, transA, K, M, N, 1, b_W, K, b_WH, M, 0, b_Haux, K);
     q.wait();
}


void WH_mult_Ht(queue q, buffer<C_REAL, 1> &b_Waux, buffer<C_REAL, 1> &b_WH, buffer<C_REAL, 1> &b_Htras, int N, int M, int K) 
{
    oneapi::mkl::blas::column_major::gemm(q, transB, transB, K, N, M, 1, b_Htras, K, b_WH, M, 0, b_Waux, K);
    q.wait();
}


void adjust_WH(queue q, buffer<C_REAL, 1> &b_W, buffer<C_REAL, 1> &b_Ht, int N, int M, int K) {
    q.submit([&](handler& cgh) {
        auto W = b_W.get_access<sycl_read_write>(cgh);

        cgh.parallel_for<class check_W>(range<2>(N, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(W[i*K + j] < eps)
                W[i*K + j] = eps;
        });
    });

    q.submit([&](handler& cgh) {
        auto Ht = b_Ht.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class check_Ht>(range<2>(M, K), [=](id <2> ij){
            int i = ij[0];
            int j = ij[1];

            if(Ht[i*K + j] < eps)
                Ht[i*K + j] = eps;
        });
    });
    q.wait();
}


// void V_div_WH3(queue q, buffer<C_REAL, 1> &b_V, buffer<C_REAL, 1> &b_WH, int N, int M) {
//     oneapi::mkl::vm::div(q, N*M, V, WH, WH);
//     q.wait();
// }


void V_div_WH(queue q, buffer<C_REAL, 1> &b_V, buffer<C_REAL, 1> &b_WH, int N, int M) {
    int max_work_group_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int group_size = max_work_group_size < M ? max_work_group_size : M;
    // adjust work-groups number 
    int remainder = (M == group_size) ? 0 : group_size - (M % group_size);
    int work_items = N * (M + remainder);

    q.submit([&](handler& cgh) {
        auto V = b_V.get_access<sycl_read>(cgh);
        auto WH = b_WH.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class V_div_WH>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            int i = item.get_global_id(0);
            WH[i] = sycl::native::divide(V[i], WH[i]);
        });
    });
    q.wait();
}


void V_div_WH2(queue q, buffer<C_REAL, 1> &b_V, buffer<C_REAL, 1> &b_WH, int N, int M) {
    q.submit([&](handler& cgh) {
        auto V = b_V.get_access<sycl_read>(cgh);
        auto WH = b_WH.get_access<sycl_read_write>(cgh);
        cgh.parallel_for<class V_div_WH2>(range<1>(N), [=](id <1> ij){
            int i = ij[0];

            for(int j = 0; j < M; j++)
                WH[i*M + j] = V[i*M + j] / WH[i*M + j];
        });
    });
    q.wait();
}


void mult_M_div_vect(queue q, buffer<C_REAL, 1> &b_Mat, buffer<C_REAL, 1> &b_Maux, buffer<C_REAL, 1> &b_acc, int M, int K) {
    q.submit([&](handler& cgh) {
        auto Mat = b_Mat.get_access<sycl_read_write>(cgh);
        auto Maux = b_Maux.get_access<sycl_read>(cgh);
        auto acc = b_acc.get_access<sycl_read>(cgh);
        cgh.parallel_for<class mul_M_div_vect>(range<1>(M), [=](id <1> ij){
            int i = ij[0];

            for(int j = 0; j < K; j++)
                Mat[i*K + j] = Mat[i*K + j] * Maux[i*K + j] / acc[j];
        });
    });
    q.wait();
}


void accum_gpu(queue q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M) {
    // init acc
    q.submit([&](auto &h) {
        auto acc = b_acc.get_access<sycl_write>(h);
        h.parallel_for(sycl::range<1>(M), [=](id <1> i) {
            acc[i] = 0;
        });
    });
    q.wait();
    
    int max_work_group_size = q.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    int fixed_N = max_work_group_size < N ? max_work_group_size : N;
    int data_size = fixed_N * M;

    q.submit([&](auto &h) {
        auto acc = b_acc.get_access<sycl_read_write>(h);
        auto X = b_X.get_access<sycl_read>(h);
        sycl::accessor<C_REAL, 1, sycl::access::mode::read_write, sycl::access::target::local> scratch(fixed_N, h);

        h.parallel_for(sycl::nd_range(range(data_size), range(fixed_N)), [=](sycl::nd_item<1> item) {
            int local_id = item.get_local_id(0);
            int group_id = item.get_group(0);
            int blocks = 0;
            int offset;
            int global_id = local_id * (M-1) + local_id + group_id;
            int global_id_offset;

            for(int i = 0; i < N; i+=fixed_N){
                offset = data_size * blocks;
                global_id_offset = global_id + offset;

                scratch[local_id] = X[global_id_offset];

                // Do a tree reduction on items in work-group
                for (int j = fixed_N / 2; j > 0; j >>= 1) {
                    item.barrier(sycl::access::fence_space::local_space);

                    if (local_id < j)
                        scratch[local_id] += scratch[local_id + j];
                }

                if (local_id == 0)
                    acc[group_id] += scratch[0];
                
                blocks++;
                item.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
    q.wait();
}


void accum_cpu2(queue q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M) { 
    q.submit([&](handler& cgh) {
        auto acc = b_acc.get_access<sycl_read_write>(cgh);
        auto X = b_X.get_access<sycl_read>(cgh);
        cgh.parallel_for<class accum_add_matrix>(range<1>(M), [=](id <1> j){

            acc[j] = 0;
            for(int i = 0; i < N; i++)
                acc[j] += X[i*M + j];
        });
    });
    q.wait();
}


void accum_cpu(queue q, buffer<C_REAL, 1> &b_acc, buffer<C_REAL, 1> &b_X, int N, int M) {
    auto acc = b_acc.get_access<sycl_read_write>();
    auto X = b_X.get_access<sycl_read>();
    for (int j = 0; j < M; j++){
        acc[j] = 0;
        for (int i = 0; i < N; i++) {
            acc[j] += X[i*M + j];
        }
    }
}
