# Non-negative Matrix Factorization for SYCL
Non-negative Matrix Factorization algorithm implemented in SYCL.

## Requirements
To compile and launch the code you need to install [oneAPI base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html). To run on Nvidia GPUs you have to install the [intel open compiler](https://github.com/intel/llvm) as well as build the [oneMKL toolchain](https://github.com/oneapi-src/oneMKL).

## How to compile it?
To compile the code you can simply do it with `make` command. However, the Makefile offers the following compilation options:

* OPTIMIZE=[yes, no] --> Enables compiler -O3 option. Activated by default.
* DEBUG=[yes, no] --> Sets the -g compiler option. Disabled by default.
* DEVICE=[cpu, nvidia, igpu] --> States where to run the SYCL kernels. It is set by default to cpu.
* REAL=[simple, double] --> Determines whether to use simple or double float point precision. It is set by default to simple.

## How to run it?
Once you copile it, you will get the nmf executable. This binary takes the following parameters:

* matrix binary sample used to initialize the V matrix.
* N --> An unsigned integer which defines the N dimension of the matrix.
* M --> An unsigned integer which defines the M dimension of the matrix.
* K --> It is an unsigned integer, and states the factorization parameter of the matrix.
* Tests --> The number of tests to perform.
* Stop threshold --> The NMF stop threshold.

An example to run it could be:

`./nmf ../data/V.bin 1000 500 4 1 10`

Running on a hyper-threding CPU could reduce performance, to disable it you can use the following variable set to the number of phisical cores on the CPU.

`export DPCPP_CPU_NUM_CUS=12`

To sufficient splitting to balance load, use:

`export DPCPP_CPU_SCHEDULE=dynamic`

For changing the backend where the device executes, you can use "SYCL_DEVICE_FILTER" variables, for example:

`export SYCL_DEVICE_FILTER=gpu,level_zero`

## Acknowledgements

This work has been supported by the EU (FEDER) and the Spanish MINECO and CM under grants S2018/TCS-4423 and RTI2018-093684-B-I00.
