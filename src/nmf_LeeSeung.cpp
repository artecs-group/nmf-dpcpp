#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include "queue_data.hpp"

#ifdef BLAS_KERNEL
#include "./kernels/blas_kernel/blas_kernel.h"
#else
#include "./kernels/bare_kernel/bare_kernel.h" //default kernels
#endif


inline int pow2roundup(int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}


double gettime() {
	double final_time;
	struct timeval tv1;
	
	gettimeofday(&tv1, (struct timezone*)0);
	final_time = (tv1.tv_usec + (tv1.tv_sec)*1000000ULL);

	return final_time;
}


void matrix_copy1D_uchar(unsigned char* in, unsigned char* out, int nx) {
	for (int i = 0; i < nx; i++)
		out[i] = in[i];
}


void matrix_copy2D(C_REAL* in, C_REAL* out, int nx, int ny) {
	for (int i = 0; i < nx; i++)
		for(int j = 0; j < ny; j++)
			out[i*ny + j] = in[i*ny + j];
}


void initWH(int N, int M, int K, C_REAL* W, C_REAL* Htras) {	
	// int seedi;
	// FILE *fd;

	// /* Generated random values between 0.00 - 1.00 */
	// fd = fopen("/dev/urandom", "r");
	// fread(&seedi, sizeof(int), 1, fd);
	// fclose(fd);
	// srand(seedi);

#ifdef DEBUG
	/* Added to debug */
	FILE *fIn;
	int size_W{N*K};

	fIn = fopen("w_bin.bin", "r");
	fread(W, sizeof(C_REAL), size_W, fIn);
	fclose(fIn);

	int size_H{M*K};
	fIn = fopen("h_bin.bin", "r");
	fread(Htras, sizeof(C_REAL), size_H, fIn);
	fclose(fIn);
#else
	srand(0);

	for (int i = 0; i < N*K; i++)
		W[i] = ((C_REAL)(rand()))/RAND_MAX;

	for (int i = 0; i < M*K; i++)
		Htras[i] = ((C_REAL)(rand()))/RAND_MAX;
#endif

	//std::fill(W + (N*K), W + (N_pad*K), 0);
	//std::fill(Htras + (M*K), Htras + (M_pad*K), 0);
}


void printMATRIX(C_REAL *m, int I, int J) {	
	printf("--------------------- matrix --------------------\n");
	printf("             ");
	for (int j = 0; j < J; j++) {
		if (j < 10)
			printf("%i      ", j);
		else if (j < 100)
			printf("%i     ", j);
		else 
			printf("%i    ", j);
	}
	printf("\n");

	for (int i = 0; i < I; i++) {
		if (i<10)
			printf("Line   %i: ", i);
		else if (i<100)
			printf("Line  %i: ", i);
		else
			printf("Line %i: ", i);

		for (int j = 0; j < J; j++)
			printf("%5.4f ", m[i*J + j]);
		printf("\n");
	}
}


void init_V(C_REAL *V, char* file_name, queue_data *qd1, queue_data *qd2) {
	int N = qd1->N;
	int M = qd1->M;
	int M1 = qd1->M_split;
	int M2 = qd2->M_split;

#ifndef RANDOM
	FILE *fIn = fopen(file_name, "r");
	fread(V, sizeof(C_REAL), N*M, fIn);
	fclose(fIn);
#else
	/* Generated random values between 0.00 - 1.00 */
	// FILE *fd;
	// int seedi;
    // fd = fopen("/dev/urandom", "r");
    // fread( &seedi, sizeof(int), 1, fd);
    // fclose (fd);
    srand( 0 );

    for (int i = 0; i < N*M; i++)
        V[i] = ((C_REAL)(rand()))/RAND_MAX;
#endif

	for (int i = 0; i < N; i++)
        for (int j = 0; j < M1; j++)
			qd1->V_col[i*M1 + j] = V[i*M + j];

	for (int i = 0; i < N; i++)
        for (int j = 0; j < M2; j++)
            qd2->V_col[i*M2 + j] = V[i*M + j + M1];

	std::copy(V, V + (qd1->N_split * M), qd1->V_row);
	std::copy(V + (qd1->N_split * M), V + (N*M), qd2->V_row);
}


/* Gets the difference between matrix_max_index_h and conn_last matrices. */
int get_difference(unsigned char* classification, unsigned char* last_classification, int nx) {
	int diff{0};
	int conn, conn_last;
	
	for(int i = 0; i < nx; i++)
		for(int j = i+1; j < nx; j++) {
			conn = (int)( classification[j] == classification[i] );
			conn_last = (int) ( last_classification[j] == last_classification[i] );
			diff += ( conn != conn_last );
		}

	return diff;
}


/* Get consensus from the classificacion vector */
void get_consensus(unsigned char* classification, unsigned char* consensus, int nx) {
	unsigned char conn;
	int ii{0};
	
	for(int i = 0; i < nx; i++)
		for(int j = i+1; j < nx; j++) {
			conn = ( classification[j] == classification[i] );
			consensus[ii] += conn;
			ii++;
		}
}


/* Obtain the classification vector from the Ht matrix */
void get_classification(C_REAL* Htras, unsigned char* classification, int M, int K)
{
	C_REAL max;
	
	for (int i = 0; i < M; i++) {
		max = 0.0;
		for (int j = 0; j < K; j++)
			if (Htras[i*K + j] > max) {
				classification[i] = (unsigned char)(j);
				max = Htras[i*K + j];
			}
	}
}


C_REAL get_Error(C_REAL* V, C_REAL* W, C_REAL* Htras, int N, int M, int K) {
	/*
	* norm( V-WH, 'Frobenius' ) == sqrt( sum( diag( (V-WH)'* (V-WH) ) )
	* norm( V-WH, 'Frobenius' )**2 == sum( diag( (V-WH)'* (V-WH) ) )
	*/

	/*
	* d[1..m] = diag( (V-WH)t * (V-WH) )
	*
	* is equivalent to:
	*     for i=1..m
	*         d[i] = sum( V-WH[:,i] .* V-WH[:,i] )
	*
	*
	* is equivalent to; error = sum( ( V-Vnew[:,i] .* V-Vnew[:,i] )
	*
	*/
    
	C_REAL error{0.0};
	C_REAL Vnew;

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++){
			Vnew = 0.0;
			for(int k = 0; k < K; k++)
				Vnew += W[i*K + k] * Htras[j*K + k];

			error += (V[i*M + j] - Vnew) * (V[i*M + j] - Vnew);
		}
	}
	
	return error;
}


void writeSolution(C_REAL* W, C_REAL* Ht, unsigned char* consensus, int N, int M,
    int K, int nTests)
{
	FILE *fOut;
	char file[100];
	C_REAL *H = new C_REAL[K*M];
	
	for (int i = 0; i < K; i++)
		for (int j = 0; j < M; j++)
			H[i*M + j] = Ht[j*K + i];
	
	sprintf(file,"solution-NMFLeeSeung_%i", K);
	fOut = fopen(file, "w");
	fwrite( &N, sizeof(int), 1, fOut);
	fwrite( &M, sizeof(int), 1, fOut);
	fwrite( &K, sizeof(int), 1, fOut);
	fwrite( W, sizeof(C_REAL), N*K, fOut);
	fwrite( H, sizeof(C_REAL), K*M, fOut);
	fwrite( &nTests, sizeof(int), 1, fOut);
	fwrite( consensus, sizeof(unsigned char), (M*(M-1))/2, fOut);
	fclose( fOut );
	delete [] H;
}


void nmf(int niter, queue_data qd1, queue_data qd2) {
	/*************************************/
	/*                                   */
	/*      Main Iterative Process       */
	/*                                   */
	/*************************************/

	for (int iter = 0; iter < niter; iter++) {
		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/

        /* WH = W*H */
		W_mult_H(q1, b_WH_col1 , b_W, b_Htras1, N, M1, K);
		W_mult_H(q2, b_WH_col2, b_W, b_Htras2, N, M2, K);

		/* WH = (V./(W*H) */
		V_div_WH(q1, b_V_col1, b_WH_col1, N, M1);
		q1.wait();
        V_div_WH(q2, b_V_col2, b_WH_col2, N, M2);
		q2.wait();

		/* Shrink into one column */
        accum(q2, b_accW, b_W, N_pad, K);
		q2.wait();

		/* Haux = (W'* {V./(WH)} */
        Wt_mult_WH(q1, b_Haux1, b_W, b_WH_col1, N, M1, K);
		q1.wait();
		Wt_mult_WH(q2, b_Haux2, b_W, b_WH_col2, N, M2, K);
		q2.wait();

		/* H = H .* (Haux) ./ accum_W */
        mult_M_div_vect(q1, b_Htras1, b_Haux1, b_accW, M1, K);
		q1.wait();
		mult_M_div_vect(q2, b_Htras2, b_Haux2, b_accW, M2, K);
		q2.wait();

		/*******************************************/
		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
		/*******************************************/

		/* WH = W*H */
		W_mult_H(q1, b_WH_row1, b_W1, b_Htras, N1, M, K);
		q1.wait();
		W_mult_H(q2, b_WH_row2, b_W2, b_Htras, N2, M, K);
		q2.wait();

		/* WH = (V./(W*H) */
		V_div_WH(q1, b_V_row1, b_WH_row1, N1, M);
		q1.wait();
        V_div_WH(q2, b_V_row2, b_WH_row2, N2, M);
		q2.wait();

		/* Waux =  {V./(W*H)} *H' */
        WH_mult_Ht(q1, b_Waux1, b_WH_row1, b_Htras, N1, M, K);
		q1.wait();
		WH_mult_Ht(q2, b_Waux2, b_WH_row2, b_Htras, N2, M, K);
		q2.wait();

		/* Shrink into one column */
        accum(q2, b_accH, b_Htras, M_pad, K);
		q2.wait();

		/* W = W .* Waux ./ accum_H */
		mult_M_div_vect(q1, b_W1, b_Waux1, b_accH, N1, K);
		q1.wait();
		mult_M_div_vect(q2, b_W2, b_Waux2, b_accH, N2, K);
		q2.wait();

    }
	/* Adjust small values to avoid undeflow: h=max(h,eps);w=max(w,eps); */
	adjust_WH(q1, b_W, b_Htras, N, M, K);
	q1.wait();
	// adjust_WH(q2, b_W2, b_Htras2, N2, M2, K);
	// q2.wait();
}


int main(int argc, char *argv[]) {
	int niters;

	C_REAL *V, *Htras, *W, *W_best, *Htras_best;
	unsigned char *classification, *last_classification;
	unsigned char *consensus;

	int stop;
	char file_name[255];
	int iter;
	int diff, inc;
	int N_pad = pow2roundup(N);
	int M_pad = pow2roundup(M);
	
	double time0, time1;
	
	C_REAL error;
	C_REAL error_old = 9.99e+50;

    setbuf( stdout, NULL );
	
	if (argc != 7) {
		printf("./exec dataInput.bin N M K nTests stop_threshold (argc=%i %i)\n", argc, atoi(argv[2]));
		return 1;
	}

	strcpy(file_name, argv[1]);
	int N              = atoi(argv[2]);
	int M              = atoi(argv[3]);
	int K              = atoi(argv[4]);
	int nTests         = atoi(argv[5]);
	int stop_threshold = atoi(argv[6]);

    printf("file=%s\nN=%i M=%i K=%i nTests=%i stop_threshold=%i\n", file_name, N, M, K, nTests, stop_threshold);

	constexpr int split_factor = 2;
	int N1 = (N / split_factor);
	int N2 = N - N1;
	int M1 = (M / split_factor);
	int M2 = M - M1;

	queue_data qd1{N, N1, M, M1, K, cpu_selector{}};
	queue_data qd2{N, N2, M, M2, K, IntelGPUSelector{}};

	std::cout << "Running on "
				<< qd1.q.get_device().get_info<sycl::info::device::name>()
				<< std::endl
				<< qd2.q.get_device().get_info<sycl::info::device::name>()
				<< std::endl;

	V                   = new C_REAL[N*M];
	Htras               = new C_REAL[M*K];
	W                   = new C_REAL[N*K];
    W_best              = new C_REAL[N*K];
    Htras_best          = new C_REAL[M*K];
    classification      = new unsigned char[M];
	last_classification = new unsigned char[M];
	consensus           = new unsigned char[M*(M-1)/2];

	init_V(V, file_name, &qd1, &qd2);
	initWH(N, M, K, W, Htras);


	/**********************************/
	/******     MAIN PROGRAM     ******/
	/**********************************/
	time0 = gettime();

	for(int test = 0; test < nTests; test++) {
		/* Copy W and H to devices*/
		copy_WH_to(qd1.q, W, qd1.W, Htras, qd1.Htras, qd1.N, qd1.M, qd1.K);
		copy_WH_to(qd2.q, W, qd2.W, Htras, qd2.Htras, qd2.N, qd2.M, qd2.K);

		niters = 2000 / NITER_TEST_CONV;

		stop   = 0;
		iter   = 0;
		inc    = 0;
		while(iter < niters && !stop) {
			iter++;

			/* Main Proccess of NMF Brunet */
			nmf(NITER_TEST_CONV, cpu_q, gpu_q, b_V,
				b_V_col1, b_V_col2, b_WH, b_WH_col1, b_WH_col2, b_W, 
				b_Htras, b_Waux, b_Haux, b_acumm_W, b_acumm_H, N_pad, M_pad);

			/* Copy back W and H from devices*/
			copy_WH_from(qd1.q, W, qd1.W, Htras, qd1.Htras, qd1.N, qd1.M, qd1.K);
			copy_WH_from(qd2.q, W, qd2.W, Htras, qd2.Htras, qd2.N, qd2.M, qd2.K);

			/* Test of convergence: construct connectivity matrix */
			get_classification(Htras, classification, M, K);

			diff = get_difference(classification, last_classification, M);
			matrix_copy1D_uchar(classification, last_classification, M);

			if(diff > 0) 	/* If connectivity matrix has changed, then: */
				inc = 0;  /* restarts count */
			else		/* else, accumulates count */
				inc++;

			if (verbose)
				printf("iter=%i inc=%i number_changes=%i\n", iter*NITER_TEST_CONV, inc, 2*diff);

			/* Breaks if connectivity stops changing: NMF converged */
			if (inc > stop_threshold)
				stop = 1;
		}

		/* Get Matrix consensus */
		get_consensus(classification, consensus, M);

		/* Get variance of the method error = |V-W*H| */
		error = get_Error(V, W, Htras, N, M, K);
		if (error < error_old) {
			printf("Better W and H, Error %e Test=%i, Iter=%i\n", error, test, iter);
			matrix_copy2D(W, W_best, N, K);
			matrix_copy2D(Htras, Htras_best, M, K);
			error_old = error;
		}
	}
	time1 = gettime();
	/**********************************/
	/**********************************/

	printf("\n\n\n EXEC TIME %f (us).       N=%i M=%i K=%i Tests=%i (%lu)\n", time1-time0, N, M, K, nTests, sizeof(C_REAL));
	printf("Final error %e \n", error);
	
	/* Write the solution of the problem */
	//writeSolution(W_best, Htras_best, consensus, N, M, K, nTests);

	//printMATRIX(W_best, N, K);

    /* Free memory used */
	delete[] V;
	delete[] W;
	delete[] Htras;
	delete[] W_best;
	delete[] Htras_best;
	delete[] classification;
	delete[] last_classification;
	delete[] consensus;

	return 0;
}
