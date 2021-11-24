#!/bin/sh
rm -f NMF.cuda *.o
nvcc -c kernels.cu -DGPU -DDEBUG -O3 -g -I/usr/local/cuda/include  -I. -lcuda -lcublas
g++ -c  NMF_LeeSeung.c -O3 -DREAL -DGPU  -I/usr/local/cuda/include -lcuda -lcublas -lm
g++ -o NMF.cuda  NMF_LeeSeung.o kernels.o -O3 -DREAL -DGPU  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas -lm