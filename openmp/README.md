# nmf-openmp
Non-negative Matrix Factorization algorithm implemented in OpenMP

## Run
Use the 'OMP_TARGET_OFFLOAD' env variable to run over GPU, and the 'LIBOMPTARGET_PLUGIN' for selecting the backend.

`export OMP_TARGET_OFFLOAD=MANDATORY`

`export LIBOMPTARGET_PLUGIN=<OPENCL/LEVEL0>`

You can also choose the number of threads to launch in CPU with the variable 'KMP_HW_SUBSET', selecting (in the below example) 24 cores and 1 thread per core. For more information go [here]('https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/optimization-and-programming-guide/openmp-support/controlling-thread-allocation.html').

`export KMP_HW_SUBSET=24c,1t`


## Debug
In order to get GPU debug profiling set:

`export LIBOMPTARGET_DEBUG=1`

`export LIBOMPTARGET_INFO=2`

`export LIBOMPTARGET_PROFILE=T`