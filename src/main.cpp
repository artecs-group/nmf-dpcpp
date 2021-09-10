#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "common.hpp"
#include "./kernels/kernels.hpp"

double nmf_t{0}, nmf_total{0}, gemm_t{0}, gemm_total{0}, div_t{0}, div_total{0}, red_t{0}, 
	red_total{0}, mulM_t{0}, mulM_total{0};


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


void matrix_copy1D_uchar(unsigned char *in, unsigned char *out, int nx) {
	for (int i = 0; i < nx; i++)
		out[i] = in[i];
}


void matrix_copy2D(C_REAL *in, C_REAL *out, int nx, int ny) {
	for (int i = 0; i < nx; i++)
		for(int j = 0; j < ny; j++)
			out[i*ny + j] = in[i*ny + j];
}


void initWH(C_REAL *W, C_REAL *Htras, int N, int M, int K, int N_pad, int M_pad) {	
	// int seedi;
	// FILE *fd;

	// /* Generated random values between 0.00 - 1.00 */
	// fd = fopen("/dev/urandom", "r");
	// fread(&seedi, sizeof(int), 1, fd);
	// fclose(fd);
	// srand(seedi);
	srand(0);

	for (int i = 0; i < N*K; i++)
		W[i] = ((C_REAL)(rand())) / ((C_REAL) RAND_MAX);

	for (int i = 0; i < M*K; i++)
		Htras[i] = ((C_REAL)(rand())) / ((C_REAL) RAND_MAX);

#ifdef DEBUG
	/* Added to debug */
	FILE *fIn;
	int size_W = N*K;

	fIn = fopen("w_bin.bin", "r");
	fread(W, sizeof(C_REAL), size_W, fIn);
	fclose(fIn);

	int size_H = M*K;
	fIn = fopen("h_bin.bin", "r");
	fread(Htras, sizeof(C_REAL), size_H, fIn);
	fclose(fIn);
#endif

	std::fill(W + (N*K), W + (N_pad*K), 0);
	std::fill(Htras + (M*K), Htras + (M_pad*K), 0);
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


void print_WH(C_REAL *W, C_REAL *Htras, int N, int M, int K) {
	for (int i = 0; i < N; i++){
		printf("W[%i]: ", i);

		for (int j = 0; j < K; j++)
			printf("%f ", W[i*K + j]);

		printf("\n");
	}

	for (int i = 0; i < K; i++){
		printf("H[%i]: ", i);

        	for (int j = 0; j < M; j++)
				printf("%f ", Htras[j*K + i]);

		printf("\n");
	}
}


C_REAL *get_V(int N, int M, char* file_name, queue q) {
	C_REAL *V = malloc_shared<C_REAL>(N * M, q);

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
        V[i] = ((C_REAL)(rand())) / ((C_REAL) RAND_MAX);
#endif

	return V;
}


/* Gets the difference between matrix_max_index_h and conn_last matrices. */
int get_difference(unsigned char *classification, unsigned char *last_classification, int nx) {
	int diff = 0;
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
void get_consensus(unsigned char *classification, unsigned char *consensus, int nx) {
	unsigned char conn;
	int ii = 0;
	
	for(int i = 0; i < nx; i++)
		for(int j = i+1; j < nx; j++) {
			conn = ( classification[j] == classification[i] );
			consensus[ii] += conn;
			ii++;
		}
}


/* Obtain the classification vector from the Ht matrix */
void get_classification(C_REAL *Htras, unsigned char *classification, int M, int K) {
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


C_REAL get_Error(C_REAL *V, C_REAL *W, C_REAL *Htras, int N, int M, int K) {
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
	C_REAL error = 0.0;
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


void writeSolution(C_REAL *W, C_REAL*Ht, unsigned char *consensus, int N, int M,
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


void nmf(int niter, queue q, C_REAL *V, C_REAL *WH, 
	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
	C_REAL *accW, C_REAL *accH, int N, int M, int K, int N_pad, int M_pad)
{
	/*************************************/
	/*                                   */
	/*      Main Iterative Process       */
	/*                                   */
	/*************************************/

	nmf_t = gettime();
	for (int iter = 0; iter < niter; iter++) {
		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/

		gemm_t = gettime();
        W_mult_H(q, WH, W, Htras, N, M, K);	/* WH = W*H */
		gemm_total += (gettime() - gemm_t);

		div_t = gettime();
        V_div_WH(q, V, WH, N, M);			/* WH = (V./(W*H) */
		div_total += (gettime() - div_t);

		red_t = gettime();
        accum(q, accW, W, N_pad, K); 		/* Shrink into one column */
		red_total += (gettime() - red_t);

		gemm_t = gettime();
        Wt_mult_WH(q, Haux, W, WH, N, M, K);	/* Haux = (W'* {V./(WH)} */
		gemm_total += (gettime() - gemm_t);

		mulM_t = gettime();
        mult_M_div_vect(q, Htras, Haux, accW, M, K);/* H = H .* (Haux) ./ accum_W */
		mulM_total += (gettime() - mulM_t);

		/*******************************************/
		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
		/*******************************************/

		gemm_t = gettime();
        W_mult_H(q, WH, W, Htras, N, M, K);	/* WH = W*H */
		gemm_total += (gettime() - gemm_t);

		div_t = gettime();
        V_div_WH(q, V, WH, N, M);			/* WH = (V./(W*H) */
		div_total += (gettime() - div_t);

		gemm_t = gettime();
        WH_mult_Ht(q, Waux, WH, Htras, N, M, K);/* Waux =  {V./(W*H)} *H' */
		gemm_total += (gettime() - gemm_t);

		red_t = gettime();
        accum(q, accH, Htras, M_pad, K);		/* Shrink into one column */
		red_total += (gettime() - red_t);

		mulM_t = gettime();
        mult_M_div_vect(q, W, Waux, accH, N, K);/* W = W .* Waux ./ accum_H */
		mulM_total += (gettime() - mulM_t);
    }

	nmf_total += (gettime() - nmf_t);
}


int main(int argc, char *argv[]) {
	int niters;

	C_REAL *V, *WH, *W, *Htras, *Haux, *Waux, *acumm_W, *acumm_H;
	C_REAL *W_best, *Htras_best;
	unsigned char *classification, *last_classification;
	unsigned char *consensus;

	int stop;
	char file_name[255];
	int iter;
	int diff, inc;
	
	double time0, time1;

	constexpr bool verbose{false};
	
	C_REAL error;
	C_REAL error_old = 9.99e+50;

    setbuf( stdout, NULL );
	
	if (argc != 7) {
		printf("./exec dataInput.bin N M K nTests stop_threshold (argc=%i %i)\n", argc, atoi(argv[2]));
		return 1;
	}

	strcpy(file_name, argv[1]);
	int N              = atoi(argv[2]);
	int N_pad          = pow2roundup(N);
	int M              = atoi(argv[3]);
	int M_pad          = pow2roundup(M);
	int K              = atoi(argv[4]);
	int nTests         = atoi(argv[5]);
	int stop_threshold = atoi(argv[6]);

    printf("file=%s\nN=%i M=%i K=%i nTests=%i stop_threshold=%i\n", file_name, N, M, K, nTests, stop_threshold);

#if defined(INTEL_IGPU_DEVICE)
	IntelGPUSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CUDASelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue q{selector};
	std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

	V            	  = get_V(N, M, file_name, q);
	W                 = malloc_shared<C_REAL>(N_pad * K, q);
	Htras             = malloc_shared<C_REAL>(M_pad * K, q);
	WH                = malloc_device<C_REAL>(N * M, q);

	Haux              = malloc_device<C_REAL>(M * K, q);
	Waux              = malloc_device<C_REAL>(N * K, q);
	acumm_W           = malloc_device<C_REAL>(K, q);
	acumm_H           = malloc_device<C_REAL>(K, q);

    W_best              = malloc_host<C_REAL>(N * K, q);
    Htras_best          = malloc_host<C_REAL>(M * K, q);
    classification      = malloc_host<unsigned char>(M, q);
	last_classification = malloc_host<unsigned char>(M, q);
	consensus           = malloc_host<unsigned char>(M*(M-1)/2, q);

	/**********************************/
	/******     MAIN PROGRAM     ******/
	/**********************************/
	time0 = gettime();

	for(int test = 0; test < nTests; test++) {
		/* Init W and H */
		initWH(W, Htras, N, M, K, N_pad, M_pad);

		niters = 2000 / NITER_TEST_CONV;

		stop   = 0;
		iter   = 0;
		inc    = 0;
		while(iter < niters && !stop) {
			iter++;

			/* Main Proccess of NMF Brunet */
			nmf(NITER_TEST_CONV, q, V, WH, W, 
				Htras, Waux, Haux, acumm_W, acumm_H,
				N, M, K, N_pad, M_pad);

			/* Adjust small values to avoid undeflow: h=max(h,eps);w=max(w,eps); */
			adjust_WH(q, W, Htras, N, M, K);

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

	std::cout << std::endl 
			<< "Total NMF time = " << nmf_total << " (us) --> 100%" << std::endl
			<< "    Gemm time = " << gemm_total << " (us) --> " << gemm_total/nmf_total*100 << "%" << std::endl
			<< "    Division time = " << div_total << " (us) --> " << div_total/nmf_total*100 << "%" << std::endl
			<< "    Reduction time = " << red_total << " (us) --> " << red_total/nmf_total*100 << "%" << std::endl
			<< "    Mult elem by elem time = " << mulM_total << " (us) --> " << mulM_total/nmf_total*100 << "%" << std::endl;

	printf("\n\n\n EXEC TIME %f (us).       N=%i M=%i K=%i Tests=%i (%lu)\n", time1-time0, N, M, K, nTests, sizeof(C_REAL));
	printf("Final error %e \n", error);

	/* Write the solution of the problem */
	writeSolution(W_best, Htras_best, consensus, N, M, K, nTests);

	//printMATRIX(W_best, N, K);

    /* Free memory used */
	free(V, q);
	free(W, q);
	free(Htras, q);
	free(WH, q);
	free(Haux, q);
	free(Waux, q);
	free(acumm_W, q);
	free(acumm_H, q);
	free(W_best, q);
	free(Htras_best, q);
	free(classification, q);
	free(last_classification, q);
	free(consensus, q);

	return 0;
}
