#include "common.hpp"
#include "kernels/kernels.h"

/* Number of iterations before testing convergence (can be adjusted) */
constexpr int NITER_TEST_CONV = 10;

/* Spacing of floating point numbers. */
constexpr real eps = 2.2204e-16;

double nmf_t{0};
double nmf_total{0};
extern float gemm_total;
extern float div_total;
extern float red_total;
extern float mulM_total;


inline int pow2roundup(int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}


double gettime()
{
	double final_time;
	struct timeval tv1;
	
	gettimeofday(&tv1, (struct timezone*)0);
	final_time = (tv1.tv_usec + (tv1.tv_sec)*1000000ULL);

	return(final_time);
}


void matrix_copy1D_uchar(unsigned char *in, unsigned char *out, int nx) {
	for (int i = 0; i < nx; i++)
		out[i] = in[i];
}


void matrix_copy2D(real *in, real *out, int nx, int ny) {
	for (int i = 0; i < nx; i++)
		for(int j = 0; j < ny; j++)
			out[i*ny + j] = in[i*ny + j];
}


void initWH(real *W, real *Htras, int N, int M, int K, int N_pad, int M_pad)
{
	int seedi = 0;
	int size_W = N*K;
	int size_H = M*K;
	
	/* Generated random values between 0.00 - 1.00 */
	// FILE *fd;
	// fd = fopen("/dev/urandom", "r");
	// fread( &seedi, sizeof(int), 1, fd); 
	// fclose (fd);
	srand( seedi ); 
	
	for (int i = 0; i < N*K; i++)
		W[i] = ((real)(rand()))/((real) RAND_MAX);

	for (int i = 0; i < M*K; i++)
		Htras[i] = ((real)(rand()))/((real) RAND_MAX);

#ifndef RANDOM
	/* Added to debug */
	FILE *fIn = fopen("w_bin.bin", "r");;
	fread(W, sizeof(real), size_W, fIn);
	fclose(fIn);
	
	fIn = fopen("h_bin.bin", "r");
	fread(H, sizeof(real), size_H, fIn);
	fclose(fIn);
#endif

	std::fill(W + (N*K), W + (N_pad*K), 0);
	std::fill(Htras + (M*K), Htras + (M_pad*K), 0);
}


void printMATRIX(real *m, int I, int J) {	
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


void print_WH(real *W, real *Htras, int N, int M, int K) {
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


real *getV(char* file_name, int N, int M)
{	
	int size_V = N*M;
	real *V = new real[size_V];

#ifndef RANDOM
	FILE *fIn = fopen(file_name, "r");
	fread(V, sizeof(real), size_V, fIn);
	fclose(fIn);
#else
	/* Generated random values between 0.00 - 1.00 */
	int seedi = 0;
	// FILE *fd;
	// fd = fopen("/dev/urandom", "r");
	// fread( &seedi, sizeof(int), 1, fd);
	// fclose (fd);
	srand( seedi );

	for (int i = 0; i < size_V; i++)
		V[i] = ((real)(rand()))/((real) RAND_MAX);
#endif
	return(V);
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
void get_classification(real *Htras, unsigned char *classification, int M, int K) {
	real max;
	
	for (int i = 0; i < M; i++) {
		max = 0.0;
		for (int j = 0; j < K; j++)
			if (Htras[i*K + j] > max) {
				classification[i] = (unsigned char)(j);
				max = Htras[i*K + j];
			}
	}
}


real get_Error(real *V, real *W, real *Htras, int N, int M, int K) {
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
	real error = 0.0;
	real Vnew;

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


void writeSolution(real *W, real* Ht, unsigned char *consensus, int N, int M,
    int K, int nTests)
{
	FILE *fOut;
	char file[100];
	real *H = new real[K*M];
	
	for (int i = 0; i < K; i++)
		for (int j = 0; j < M; j++)
			H[i*M + j] = Ht[j*K + i];
	
	sprintf(file,"solution-NMFLeeSeung_%i", K);
	fOut = fopen(file, "w");
	fwrite( &N, sizeof(int), 1, fOut);
	fwrite( &M, sizeof(int), 1, fOut);
	fwrite( &K, sizeof(int), 1, fOut);
	fwrite( W, sizeof(real), N*K, fOut);
	fwrite( H, sizeof(real), K*M, fOut);
	fwrite( &nTests, sizeof(int), 1, fOut);
	fwrite( consensus, sizeof(unsigned char), (M*(M-1))/2, fOut);
	fclose( fOut );
	delete [] H;
}


void nmf(int niter, real *d_V, real *d_WH, real *d_W, real *d_Htras, real *d_Waux, real *d_Haux,
	real *d_accW, real *d_accH,
	int N, int M, int K, int N_pad, int M_pad)
{
	/*************************************/
	/*                                   */
	/*      Main Iterative Process       */
	/*                                   */
	/*************************************/
	nmf_t = gettime();
	for (int iter = 0; iter < niter; iter++)
	{
		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/

		W_mult_H(d_WH, d_W, d_Htras, N, M, K);	/* WH = W*H */
		V_div_WH(d_V, d_WH, N, M);			/* WH = (V./(W*H) */
		accum(d_accW, d_W, N_pad, K); 		/* Reducir a una columna */
		Wt_mult_WH(d_Haux, d_W, d_WH, N, M, K);	/* Haux = (W'* {V./(WH)} */
		mult_M_div_vect(d_Htras, d_Haux, d_accW, M, K);/* H = H .* (Haux) ./ accum_W */

		/*******************************************/
		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
		/*******************************************/

		W_mult_H(d_WH, d_W, d_Htras, N, M, K);	/* WH = W*H */
		V_div_WH(d_V, d_WH, N, M);			/* WH = (V./(W*H) */
		WH_mult_Ht(d_Waux, d_WH, d_Htras, N, M, K);/* Waux =  {V./(W*H)} *H' */
		accum(d_accH, d_Htras, M_pad, K);		/* Reducir a una columna */
		mult_M_div_vect(d_W, d_Waux, d_accH, N, K);/* W = W .* Waux ./ accum_H */
	}
	nmf_total += (gettime() - nmf_t);
}


int main(int argc, char *argv[])
{
	int nTests, niters;

	int i,j;
	real obj;

	real *V;
	real *W, *W_best;
	real *Htras, *Htras_best;
	unsigned char *classification, *last_classification;
	unsigned char *consensus; /* upper half-matrix size M*(M-1)/2 */

	/* Auxiliares para el computo */
	real *WH;
	real *Haux, *Waux;
	real *acumm_W;
	real *acumm_H;
	
	//For GPU
	real *d_V;
	real *d_W;
	real *d_Htras;
	real *d_WH;
	real *d_Haux;
	real *d_Waux;
	real *d_acumm_W;
	real *d_acumm_H;

	int N, N_pad;
	int M, M_pad;
	int K;
	int stop_threshold, stop;
	char file_name[255];
	int iter;
	int diff, inc;
	
	double time0, time1;
	double timeGPU2CPU, timeGPU1, timeGPU0;
	
	real error;
	real error_old = 3.4e+38;
    setbuf( stdout, NULL );
	
	if(argc != 7) {
	 	printf("./exec dataInput.bin N M K nTests stop_threshold (argc=%i %i)\n", argc, atoi(argv[2]));
		return(0);
	}

	strcpy(file_name, argv[1]);
	N              = atoi(argv[2]);
	N_pad      	   = pow2roundup(N);
	M              = atoi(argv[3]);
	M_pad          = pow2roundup(M);
	K              = atoi(argv[4]);
	nTests         = atoi(argv[5]);
	stop_threshold = atoi(argv[6]);

	printf("file=%s\nN=%i M=%i K=%i nTests=%i stop_threshold=%i\n", file_name, N, M, K, nTests, stop_threshold);

	V          = getV(file_name, N, M_pad);
	W          = new real[N_pad*K];
	Htras      = new real[M_pad*K];
	W_best     = new real[N*K];
	Htras_best = new real[M*K];
	classification      = new unsigned char[M];
	last_classification = new unsigned char[M];
	consensus           = new unsigned char[M*(M-1)/2];

	init_timers();
	cublasInit();
	cudaMalloc((void **)&d_V,      N*M_pad*sizeof(real));
	cudaMemcpy(d_V, V,             N*M_pad*sizeof(real), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_W,      N_pad*K*sizeof(real));
	cudaMalloc((void **)&d_Htras,  M_pad*K*sizeof(real));
	cudaMalloc((void **)&d_WH,     N*M_pad*sizeof(real));
	cudaMalloc((void **)&d_Waux,   N*K*sizeof(real));
	cudaMalloc((void **)&d_Haux,   M*K*sizeof(real));
	cudaMalloc((void **)&d_acumm_W, K*sizeof(real));
	cudaMalloc((void **)&d_acumm_H, K*sizeof(real));

	/**********************************/
	/******     MAIN PROGRAM     ******/
	/**********************************/
	time0 = gettime();

	for (int test = 0; test < nTests; test++)
	{
		/* Init W and H */
		initWH(W, Htras, N, M, K, N_pad, M_pad);

		timeGPU0 = gettime();
		cudaMemcpy(d_W, W, N_pad * K * sizeof(real), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Htras, Htras, K * M_pad * sizeof(real), cudaMemcpyHostToDevice);
		timeGPU1 = gettime();
		timeGPU2CPU += timeGPU1-timeGPU0;

		niters = 2000 / NITER_TEST_CONV;
		stop   = 0;
		iter   = 0;
		inc    = 0;

		while (iter < niters && !stop)
		{
			iter++;

			/* Main Proccess of NMF Brunet */
			nmf(NITER_TEST_CONV, 
				d_V, d_WH, d_W, d_Htras, d_Waux, d_Haux, d_acumm_W, d_acumm_H,
				N, M, K, N_pad, M_pad);

			/* Adjust small values to avoid undeflow: h=max(h,eps);w=max(w,eps); */
			adjust_WH_GPU(d_W, d_Htras, N, M, K);

			timeGPU0 = gettime();
			cudaMemcpy(W, d_W, N_pad * K * sizeof(real), cudaMemcpyDeviceToHost);
			cudaMemcpy(Htras, d_Htras, K * M_pad * sizeof(real), cudaMemcpyDeviceToHost);
			timeGPU1 = gettime();
			timeGPU2CPU += timeGPU1-timeGPU0;

			/* Test of convergence: construct connectivity matrix */
			get_classification( Htras, classification, M, K);

			diff = get_difference( classification, last_classification, M);
			matrix_copy1D_uchar( classification, last_classification, M );

			if( diff > 0 ) 	/* If connectivity matrix has changed, then: */
				inc=0;  /* restarts count */
			else		/* else, accumulates count */
				inc++;

			if (verbose)
				printf("iter=%i inc=%i number_changes=%i\n", iter*NITER_TEST_CONV, inc, 2*diff);

			/* Breaks if connectivity stops changing: NMF converged */
			if ( inc > stop_threshold ) {
				stop = 1;
			}	
		}

		/* Get Matrix consensus */
		get_consensus( classification, consensus, M );

		/* Get variance of the method error = |V-W*H| */
		error = get_Error(V, W, Htras, N, M, K);
		if (error<error_old){
			printf("Better W and H, Error %e Test=%i, Iter=%i\n", error, test, iter*NITER_TEST_CONV);
			matrix_copy2D(W, W_best, N, K);
			matrix_copy2D(Htras, Htras_best, M, K);
			error_old = error;
		}		
	}
	time1 = (gettime() - time0)/1000;
	/**********************************/
	/**********************************/

	nmf_total /= 1000;
	std::cout << std::endl 
			<< "Total NMF time = " << nmf_total << " (ms) --> 100%" << std::endl
			<< "    Gemm time = " << gemm_total << " (ms) --> " << gemm_total/nmf_total*100 << "%" << std::endl
			<< "    Division time = " << div_total << " (ms) --> " << div_total/nmf_total*100 << "%" << std::endl
			<< "    Reduction time = " << red_total << " (ms) --> " << red_total/nmf_total*100 << "%" << std::endl
			<< "    Dot product time = " << mulM_total << " (ms) --> " << mulM_total/nmf_total*100 << "%" << std::endl;

	printf("\n\n EXEC TIME %f (ms).       N=%i M=%i K=%i Tests=%i (%lu)\n", time1, N, M, K, nTests, sizeof(real));
	printf("Final error %e \n", error);


	/* Write the solution of the problem */
	writeSolution(W_best, Htras_best, consensus, N, M, K, nTests);

	/* Free memory used */
	delete_timers();
	delete [] W;
	delete [] Htras;
	delete [] W_best;
	delete [] Htras_best;
	delete [] classification;
	delete [] last_classification;
	delete [] consensus;
}
