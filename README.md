# nmf-openmp
Non-negative Matrix Factorization algorithm implemented in OpenMP

## Run
Use this env variable to run over GPU.

`export OMP_TARGET_OFFLOAD="MANDATORY"`


## Debug
In order to get GPU debug profiling set:

`export LIBOMPTARGET_DEBUG=1`
`export LIBOMPTARGET_PROFILE=T`