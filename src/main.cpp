#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "cublas.h"
#include "kernels/kernels.h"
#include <sys/times.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>

#define RANDOM
#define pinned_memory
#define verbose 0
#define PAD 32

#if REAL <=4
	#define real float
	#define cblas_rgemm cblas_sgemm
	#define cblas_rdot cblas_sdot
	#define cblas_rcopy cblas_scopy
	#define rmax(a,b) ( ( (a) > (b) )? (a) : (b) )
	#define rsqrt sqrtf
#else
	#define real double
	#define cblas_rgemm cblas_dgemm
	#define cblas_rdot cblas_ddot
	#define cblas_rcopy cblas_dcopy
	#define rmax fmax
	#define rsqrt sqrt
#endif

/* Number of iterations before testing convergence (can be adjusted) */
#define NITER_TEST_CONV 10

/* Spacing of floating point numbers. */
real eps=2.2204e-16;

double nmf_t{0}, nmf_total{0}, gemm_t{0}, gemm_total{0}, division_t{0}, div_total{0}, red_t{0}, 
	red_total{0}, mulM_t{0}, mulM_total{0};

void printMATRIX(real **m, int I, int J);


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

real *get_memory1D( int nx )
{ 
	int i;
	real *buffer;	

	if( (buffer=(real *)malloc(nx*sizeof(real)))== NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

	for( i=0; i<nx; i++ )
	{
		buffer[i] = (real)(i*10);
	}

	return( buffer );
}


void delete_memory1D( real *buffer )
{ 
	free(buffer);
}

unsigned char *get_memory1D_uchar( int nx )
{ 
	int i;
	unsigned char *buffer;	

	if( (buffer=(unsigned char *)malloc(nx*sizeof(int)))== NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

	for( i=0; i<nx; i++ )
	{
		buffer[i] = (int)(0);
	}

	return( buffer );
}


void delete_memory1D_uchar( unsigned char *buffer )
{ 
	free(buffer);
}

real **get_memory2D( int nx, int ny )
{ 
	int i,j;
	real **buffer;
	real *tmp_Pinned;

	if( (buffer=(real **)malloc(nx*sizeof(real *)))== NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

#ifdef pinned_memory
	cudaError_t status = cudaMallocHost((void**)&(tmp_Pinned), nx*ny*sizeof(real));
	if (status != cudaSuccess)
	{
		fprintf( stderr, "ERROR in pinned memory allocation\n" );
		free( buffer );
		return( NULL );
	}
	buffer[0] = tmp_Pinned;
#else
	if( (buffer[0]=(real *)malloc(nx*ny*sizeof(real)))==NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		free( buffer );
		return( NULL );
	}
#endif
	for( i=1; i<nx; i++ )
	{
		buffer[i] = buffer[i-1] + ny;
	}

	for( i=0; i<nx; i++ )
		for( j=0; j<ny; j++ )
		{
			buffer[i][j] = (real)(i*100+j);
		}

	return( buffer );
}

void delete_memory2D( real **buffer )
{ 
#ifdef pinned_memory
	cudaFreeHost(buffer[0]);
#else
	free(buffer[0]);
#endif
	free(buffer);
}

real **get_memory2D_renew( int nx, int ny, real *mem )
{ 
	int i,j;
	real **buffer;	

	if( (buffer=(real **)malloc(nx*sizeof(real *)))== NULL )
	{
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}


	for( i=0; i<nx; i++ )
	{
		buffer[i] = &(mem[i*ny]);
	}


	return( buffer );
}

void matrix_copy1D_uchar( unsigned char *in, unsigned char *out, int nx )
{ 
	int i;

	for (i=0; i<nx; i++)
		out[i] = in[i];
}

void matrix_copy2D(real **in, real **out, int nx, int ny)
{
	int i,j;
	
	for (i=0; i<nx; i++)
		for(j=0;j<ny;j++)
			out[i][j] = in[i][j];
}


void initWH(real **W, real **Htras, int N, int M, int K, int N_pad, int M_pad)
{
	int i,j;
	int ii, jj;
	
	int seedi = 0;
	FILE *fd;

	/* Generated random values between 0.00 - 1.00 */	
	// fd = fopen("/dev/urandom", "r");
	// fread( &seedi, sizeof(int), 1, fd); 
	// fclose (fd);
	srand( seedi ); 
	
	for (i=0; i<N; i++){
		for (j=0; j<K; j++)
			W[i][j] = ((real)(rand()))/RAND_MAX;
	}

	for (i=0; i<M; i++){
        for (j=0; j<K; j++)
			Htras[i][j] = ((real)(rand()))/RAND_MAX;
	}

#ifndef RANDOM
	/* Added to debug */
	FILE *fIn;
	real **Wtmp = get_memory2D(N, K);
	int size_W = N*K;
	fIn = fopen("w_bin.bin", "r");
	fread(&Wtmp[0][0], sizeof(real), size_W, fIn);
	fclose(fIn);
	for (i=0; i<N; i++)
                for (j=0; j<K; j++)
			W[i][j] = Wtmp[i][j];
	delete_memory2D(Wtmp);
	

	int size_H = M*K;
	real **Htmp = get_memory2D(M, K);
	fIn = fopen("h_bin.bin", "r");
	fread(&Htmp[0][0], sizeof(real), size_H, fIn);
	fclose(fIn);
	for (i=0; i<M; i++)
                for (j=0; j<K; j++)
			Htras[i][j] = Htmp[i][j];
	delete_memory2D(Htmp);
	
#endif

	std::fill(W + (N*K), W + (N_pad*K), 0);
	std::fill(Htras + (M*K), Htras + (M_pad*K), 0);
}

void printMATRIX(real **m, int I, int J)
{
	int i, j;
	
	printf("--------------------- matrix --------------------\n");
	printf("             ");
	for (j=0; j<J; j++){
		if (j<10)
			printf("%i      ", j);
		else if (j<100)
			printf("%i     ", j);
		else 
			printf("%i    ", j);
	}
	printf("\n");

	for (i=0; i<I; i++){
		if (i<10)
			printf("Line   %i: ", i);
		else if (i<100)
			printf("Line  %i: ", i);
		else
			printf("Line %i: ", i);
		for (j=0; j<J; j++)
			printf("%5.4f ", m[i][j]);
		printf("\n");
	}
}

void print_WH(real **W, real **Htras, int N, int M, int K)
{
	int i,j;	
	
	for (i=0; i<N; i++){
		printf("W[%i]: ", i);
		for (j=0; j<K; j++)
			printf("%f ",W[i][j]);
		printf("\n");
	}

	for (i=0; i<K; i++){
		printf("H[%i]: ", i);
        	for (j=0; j<M; j++)
			printf("%f ",Htras[j][i]);
		printf("\n");
	}
}

real **getV(char* file_name, int N, int M)
{
	char *data;
	FILE *fIn;
	
	/* Local variables */
	int i, j;
	int size_V;
	real **V;

	fIn = fopen(file_name, "r");

	V = get_memory2D(N, M);
	size_V = N*M;

#ifndef RANDOM 
	if (sizeof(real)==sizeof(float))
	{
		fread(&V[0][0], sizeof(float), size_V, fIn);
		fclose(fIn);
	} else {
		int i, j;
		float *Vaux = (float*)malloc(size_V*sizeof(float));
		fread(&Vaux[0], sizeof(float), size_V, fIn);
		fclose(fIn);
		for (i=0; i<N; i++)
			for (j=0; j<M; j++)
				V[i][j] = Vaux[i*M+j];
	}
#else
	/* Generated random values between 0.00 - 1.00 */
	FILE *fd;
	int seedi = 0;
        // fd = fopen("/dev/urandom", "r");
        // fread( &seedi, sizeof(int), 1, fd);
        // fclose (fd);
        srand( seedi );

        for (i=0; i<N; i++)
			for (j=0; j<M; j++)
				V[i][j] = ((real)(rand()))/RAND_MAX;

#endif	
	return(V);
}


/* Gets the difference between matrix_max_index_h and conn_last matrices. */
int get_difference( unsigned char *classification, unsigned char *last_classification, int nx)
{
	int diff;
	int conn, conn_last;
	int i, j;
	
	diff = 0;
	for(i=0; i<nx; i++)
		for(j=i+1; j<nx; j++) {
			conn = (int)( classification[j] == classification[i] );
			conn_last = (int) ( last_classification[j] == last_classification[i] );
			diff += ( conn != conn_last );
		}

	return (diff);
}

/* Get consensus from the classificacion vector */
void get_consensus( unsigned char *classification, unsigned char *consensus, int nx )
{
	unsigned char conn;
	int i, j,ii;
	
	ii = 0;
	for(i=0; i<nx; i++)
		for(j=i+1; j<nx; j++) {
			conn = ( classification[j] == classification[i] );
			consensus[ii] += conn;
			ii++;
		}
}

/* Obtain the classification vector from the Ht matrix */
void get_classification(real **Htras, unsigned char *classification, int M, int K)
{
	int i, j;
	
	real max;
	
	for (i=0; i<M; i++){
		max = 0.0;
		for (j=0; j<K; j++)
			if (Htras[i][j] > max)
			{
				classification[i] = (unsigned char)(j);
				max = Htras[i][j];
			}
	}
}


void adjust_WH(real **W, real **Ht, int N, int M, int K)
{
	int i, j;
	
	for (i=0; i<N; i++)
		for (j=0; j<K; j++)
			if (W[i][j]<eps)
				W[i][j]=eps;
				
	for (i=0; i<M; i++)
		for (j=0; j<K; j++)
			if (Ht[i][j]<eps)
				Ht[i][j]=eps;				 
}


real get_Error(real **V, real **W, real **Htras, int N, int M, int K)
{
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


	real error;
	int i, j, k;
	real Vnew;
	
	error=0.0;
	for(i=0; i<N; i++)
	{
		for(j=0; j<M; j++){
			Vnew = 0.0;
			for(k=0; k<K; k++)
				Vnew+=W[i][k]*Htras[j][k];
			error+=(V[i][j]-Vnew)*(V[i][j]-Vnew);
		}
	}
	
	return(error);


}


void writeSolution(real **W, real **Ht, unsigned char *consensus, int N, int M, int K, int nTests)
{
	int i, j;
	FILE *fOut;
	char file[100];
	real **H;
	
	H = get_memory2D(K, M);
	for (i=0; i<K; i++)
		for (j=0; j<M; j++)
			H[i][j] = Ht[j][i];
	
	sprintf(file,"solution-NMFLeeSeung_%i", K);
	fOut = fopen(file, "w");
	fwrite( &N, sizeof(int), 1, fOut);
	fwrite( &M, sizeof(int), 1, fOut);
	fwrite( &K, sizeof(int), 1, fOut);
	fwrite( W[0], sizeof(real), N*K, fOut);
	fwrite( H[0], sizeof(real), K*M, fOut);
	fwrite( &nTests, sizeof(int), 1, fOut);
	fwrite( consensus, sizeof(unsigned char), (M*(M-1))/2, fOut);
	fclose( fOut );
	
	delete_memory2D(H);
}


void nmf(int niter, real *d_V, real *d_WH, real *d_W, real *d_Htras, real *d_Waux, real *d_Haux,
	real *d_accW, real *d_accH,
	int N, int M, int K, int N_pad, int M_pad)
{
	int iter;
	
	int i, j, k;

	real diff, tot_diff;
	real Vn;

	double t0, t1;

	/*************************************/
	/*                                   */
	/*      Main Iterative Process       */
	/*                                   */
	/*************************************/
	nmf_t = gettime();
	for (iter=0; iter<niter; iter++)
	{
		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/

		gemm_t = gettime();
		W_mult_H(d_WH, d_W, d_Htras, N, M, K);	/* WH = W*H */
		gemm_total += (gettime() - gemm_t);

		division_t = gettime();
		V_div_WH(d_V, d_WH, N, M);			/* WH = (V./(W*H) */
		div_total += (gettime() - division_t);

		red_t = gettime();
		accum(d_accW, d_W, N_pad, K); 		/* Reducir a una columna */
		red_total += (gettime() - red_t);

		gemm_t = gettime();
		Wt_mult_WH(d_Haux, d_W, d_WH, N, M, K);	/* Haux = (W'* {V./(WH)} */
		gemm_total += (gettime() - gemm_t);

		mulM_t = gettime();
		mult_M_div_vect(d_Htras, d_Haux, d_accW, M, K);/* H = H .* (Haux) ./ accum_W */
		mulM_total += (gettime() - mulM_t);

		/*******************************************/
		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
		/*******************************************/

		gemm_t = gettime();
		W_mult_H(d_WH, d_W, d_Htras, N, M, K);	/* WH = W*H */
		gemm_total += (gettime() - gemm_t);

		division_t = gettime();
		V_div_WH(d_V, d_WH, N, M);			/* WH = (V./(W*H) */
		div_total += (gettime() - division_t);

		gemm_t = gettime();
		WH_mult_Ht(d_Waux, d_WH, d_Htras, N, M, K);/* Waux =  {V./(W*H)} *H' */
		gemm_total += (gettime() - gemm_t);

		red_t = gettime();
		accum(d_accH, d_Htras, M_pad, K);		/* Reducir a una columna */
		red_total += (gettime() - red_t);

		mulM_t = gettime();
		mult_M_div_vect(d_W, d_Waux, d_accH, N, K);/* W = W .* Waux ./ accum_H */
		mulM_total += (gettime() - mulM_t);
	}
	nmf_total += (gettime() - nmf_t);
}


int main(int argc, char *argv[])
{
	int nTests, niters;

	int i,j;
	real obj;

	real **V;
	real **W, **W_best;
	real **Htras, **Htras_best;
	unsigned char *classification, *last_classification;
	unsigned char *consensus; /* upper half-matrix size M*(M-1)/2 */

	/* Auxiliares para el computo */
	real **WH;
	real **Haux,  **Waux;
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
	int test, iter;
	int diff, inc;
	
	double time0, time1;
	double timeGPU2CPU, timeGPU1, timeGPU0;
	
	real error;
	real error_old=9.99e+50;

        setbuf( stdout, NULL );
	
	if (argc==7){
		strcpy(file_name, argv[1]);
		N              = atoi(argv[2]);
		N_pad      	   = pow2roundup(N);
		M              = atoi(argv[3]);
		M_pad          = pow2roundup(M);
		K              = atoi(argv[4]);
		nTests         = atoi(argv[5]);
		stop_threshold = atoi(argv[6]);
	} else {
	 	printf("./exec dataInput.bin N M K nTests stop_threshold (argc=%i %i)\n", argc, atoi(argv[2]));

		return(0);
	}

	V = getV(file_name, N, M_pad);
	printf("file=%s\nN=%i M=%i K=%i nTests=%i stop_threshold=%i\n", file_name, N, M, K, nTests, stop_threshold);	

	W          = get_memory2D(N_pad, K);
	Htras      = get_memory2D(M_pad, K);
	W_best     = get_memory2D(N, K);
	Htras_best = get_memory2D(M, K);
	
	WH         = get_memory2D(N, M_pad);
	
	Haux       = get_memory2D(M, K);
	Waux       = get_memory2D(N, K);
	
	acumm_W    = get_memory1D(K);
	acumm_H    = get_memory1D(K);
	
	classification      = get_memory1D_uchar(M);
	last_classification = get_memory1D_uchar(M);
	consensus           = get_memory1D_uchar(M*(M-1)/2);

	cublasInit();
	cudaMalloc((void **)&d_V,      N*M_pad*sizeof(real));
	cudaMemcpy(d_V, V[0], N*M_pad*sizeof(real), cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_W,      N_pad*K*sizeof(real));
	cudaMalloc((void **)&d_Htras,  M_pad*K*sizeof(real));
	cudaMalloc((void **)&d_WH,     N*M_pad*sizeof(real));

	cudaMalloc((void **)&d_Waux,   N*K*sizeof(real));
	cudaMalloc((void **)&d_Haux,   M*K*sizeof(real));

	cudaMalloc((void **)&d_acumm_W,K*sizeof(real));
	cudaMalloc((void **)&d_acumm_H,K*sizeof(real));

	/**********************************/
	/******     MAIN PROGRAM     ******/
	/**********************************/
	time0 = gettime();

	for (test=0; test<nTests; test++)
	{
		/* Init W and H */
		initWH(W, Htras, N, M, K, N_pad, M_pad);

		timeGPU0 = gettime();
		cudaMemcpy(d_W,     W[0],     N_pad*K*sizeof(real), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Htras, Htras[0], K*M_pad*sizeof(real), cudaMemcpyHostToDevice);
		timeGPU1 = gettime();
		timeGPU2CPU += timeGPU1-timeGPU0;

		niters = 2000/NITER_TEST_CONV;

		stop   = 0;
		iter   = 0;
		inc    = 0;
		while (iter<niters && !stop)
		{
			iter++;

			/* Main Proccess of NMF Brunet */
			nmf(NITER_TEST_CONV, 
				d_V, d_WH, d_W, d_Htras, d_Waux, d_Haux, d_acumm_W, d_acumm_H,
				N, M, K, N_pad, M_pad);

			/* Adjust small values to avoid undeflow: h=max(h,eps);w=max(w,eps); */
			adjust_WH_GPU(d_W, d_Htras, N, M, K);

			timeGPU0 = gettime();
			cudaMemcpy(W[0],     d_W,     N_pad*K*sizeof(real), cudaMemcpyDeviceToHost);
			cudaMemcpy(Htras[0], d_Htras, K*M_pad*sizeof(real), cudaMemcpyDeviceToHost);
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
	time1 = gettime();
	/**********************************/
	/**********************************/

	std::cout << std::endl 
			<< "Total NMF time = " << nmf_total << " (us) --> 100%" << std::endl
			<< "    Gemm time = " << gemm_total << " (us) --> " << gemm_total/nmf_total*100 << "%" << std::endl
			<< "    Division time = " << div_total << " (us) --> " << div_total/nmf_total*100 << "%" << std::endl
			<< "    Reduction time = " << red_total << " (us) --> " << red_total/nmf_total*100 << "%" << std::endl
			<< "    Dot product time = " << mulM_total << " (us) --> " << mulM_total/nmf_total*100 << "%" << std::endl;

	printf("\n\n EXEC TIME %f (us).       N=%i M=%i K=%i Tests=%i (%lu)\n", time1-time0, N, M, K, nTests, sizeof(real));
	printf("Final error %e \n", error);


	/* Write the solution of the problem */
	writeSolution(W_best, Htras_best, consensus, N, M, K, nTests);

	/* Free memory used */
	delete_memory2D(W);
	delete_memory2D(Htras);
	delete_memory2D(W_best);
	delete_memory2D(Htras_best);	
	delete_memory2D(WH);
	delete_memory2D(Waux);
	delete_memory2D(Haux);
	delete_memory1D(acumm_W);
	delete_memory1D(acumm_H);
	delete_memory1D_uchar(classification);
	delete_memory1D_uchar(last_classification);
	delete_memory1D_uchar(consensus);

}
