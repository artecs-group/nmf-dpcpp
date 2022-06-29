# Non-negative Matrix Factorization for CUDA
The NMF algorithm implemented in CUDA

## Requirements
To compile and launch the code you need to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).

## How to compile it?
To compile the code you can simply do it with `make` command. However, the Makefile offers the following compilation options:

* OPTIMIZE=[yes, no] --> Enables compiler -O3 option. Activated by default.
* DEBUG=[yes, no] --> Sets the -g compiler option. Disabled by default.
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