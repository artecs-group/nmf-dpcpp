# nmf-DPCpp
Non-negative Matrix Factorization algorithm implemented in DPC++

## Requirements
To compile and launch the code you need to install [oneAPI base Toolkit](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html).

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

`./nmf V.bin 1000 500 4 1 10`

Running on a hyper-threding CPU could reduce performance, to disable it you can use the following variable set to the number of phisical cores on the CPU.

`export DPCPP_CPU_NUM_CUS=12`

## Publications

* Igual, Francisco D., et al. "Non-negative matrix factorization on low-power architectures and accelerators: A comparative study." Computers & Electrical Engineering 46 (2015): 139-156.

    * [Available here](https://www.sciencedirect.com/science/article/abs/pii/S0045790615001287)

## Acknowledgements

This work has been supported by the EU (FEDER) and the Spanish MINECO and CM under grants S2018/TCS-4423 and RTI2018-093684-B-I00.