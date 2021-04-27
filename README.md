# nmf-openmp
Non-negative Matrix Factorization algorithm implemented in OpenMP

## Run
Use these env variable to run over GPU.

`export LIBOMPTARGET_PLUGIN=OPENCL`

`export OMP_TARGET_OFFLOAD="MANDATORY"`


## Debug
In order to get GPU debug profiling set:

`export LIBOMPTARGET_DEBUG=1`

`export LIBOMPTARGET_INFO=2`

`export LIBOMPTARGET_PROFILE=T`