#include "common.h"

/* Number of iterations before testing convergence (can be adjusted) */
const int NITER_TEST_CONV = 10;

/* Spacing of floating point numbers. */
const C_REAL eps = 2.2204e-16;

const bool verbose = false;
const char PAD = 32;

double nmf_t{0}, nmf_total{0}, WH_t{0}, WH_total{0}, V_t{0}, V_total{0}, acc_t{0}, 
	acc_total{0}, Wt_t{0}, Wt_total{0}, mulM_t{0}, mulM_total{0};

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


void initWH(C_REAL *W, C_REAL *Htras, int N, int M, int K) {	
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
#else
	srand(0);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < K; j++)
			W[i*K + j] = ((C_REAL)(rand()))/((C_REAL) RAND_MAX);

	for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
			Htras[i*K + j] = ((C_REAL)(rand()))/((C_REAL) RAND_MAX);
#endif
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


C_REAL *get_V(int N, int M, char* file_name) {
	C_REAL *V = (C_REAL *) mkl_malloc(sizeof(C_REAL) * N*M, 64);

#ifndef RANDOM
	FILE *fIn = fopen(file_name, "r");
	fread(V, sizeof(C_REAL), N*M, fIn);
	fclose(fIn);
#else
    srand( 0 );

    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            V[i*M + j] = ((C_REAL)(rand()))/((C_REAL) RAND_MAX);
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


void adjust_WH(C_REAL *W, C_REAL *Ht, int N, int M, int K) {
	for (int i = 0; i < N; i++)
		for (int j = 0; j < K; j++)
			if (W[i*K + j] < eps)
				W[i*K + j] = eps;
				
	for (int i = 0; i < M; i++)
		for (int j = 0; j < K; j++)
			if (Ht[i*K + j] < eps)
				Ht[i*K + j] = eps;				 
}


void gpu_nmf(int niter, C_REAL *V, C_REAL *WH, 
	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
	C_REAL *acumm_W, C_REAL *acumm_H, int N, int M, int K)
{
	/*************************************/
	/*                                   */
	/*      Main Iterative Process       */
	/*                                   */
	/*************************************/

	int num_blocks = 20;
	nmf_t = gettime();

	#pragma omp target enter data map(alloc:WH[0:N*M], Waux[0:N*K], Haux[0:M*K], acumm_W[0:K], acumm_H[0:K]) \
	map(to:W[0:N*K], Htras[0:M*K], V[0:N*M])
	{
		for (int iter = 0; iter < niter; iter++) {
		
			/*******************************************/
			/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
			/*******************************************/

			/* WH = W*H */
			WH_t = gettime();
			#pragma omp target variant dispatch use_device_ptr(W, Htras, WH)
			{
				cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
					N,				/* [m] */ 
					M,				/* [n] */
					K,				/* [k] */
					1, 				/* alfa */ 
					W, K, 			/* A[m][k], num columnas (lda) */
					Htras, K,		/* B[k][n], num columnas (ldb) */
					0,				/* beta */ 
					WH, M			/* C[m][n], num columnas (ldc) */
				);
			}
			WH_total += (gettime() - WH_t);

			V_t = gettime();
			#pragma omp target teams distribute parallel for simd
			//num_teams(num_blocks)
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < M; j++) {
					WH[i*M + j] = V[i*M + j] / WH[i*M + j]; /* V./(W*H) */
				}
			}
			V_total += (gettime() - V_t);

			acc_t = gettime();
			/* Reducir a una columna */
			#pragma omp target teams distribute parallel for simd
			for(int i = 0; i < K; i++) {
				acumm_W[i] = 0;
			}

			for (int j = 0; j < K; j++){
				#pragma omp target teams distribute parallel for simd reduction(+:acumm_W[j]) map(to:j)
				for (int i = 0; i < N; i++) {
					acumm_W[j] += W[i*K + j];
				}
			}
			acc_total += (gettime() - acc_t);

			Wt_t = gettime();
			#pragma omp target variant dispatch use_device_ptr(W, WH, Haux)
			{
				cblas_rgemm(CblasColMajor, CblasNoTrans, CblasTrans,
					K,				/* [m] */
					M,				/* [n] */
					N,				/* [k] */
					1,				/* alfa */
					W, K,			/* A[m][k], num columnas (lda) */
					WH, M,			/* B[k][n], num columnas (ldb) */
					0,                      	/* beta */
					Haux, K			/* C[m][n], num columnas (ldc) */
				);
			}
			Wt_total += (gettime() - Wt_t);

			mulM_t = gettime();
			#pragma omp target teams distribute parallel for simd
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < K; i++) {
					Htras[j*K + i] = Htras[j*K + i] * Haux[j*K + i] / acumm_W[i]; /* H = H .* (Haux) ./ accum_W */
				}
			}
			mulM_total += (gettime() - mulM_t);

			/*******************************************/
			/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
			/*******************************************/

			/* WH = W*H */
			WH_t = gettime();
			#pragma omp target variant dispatch use_device_ptr(W, Htras, WH)
			{
				cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasTrans, 
					N,				/* [m] */ 
					M, 				/* [n] */
					K,				/* [k] */
					1, 				/* alfa */ 
					W, K,		 	/* A[m][k], num columnas (lda) */
					Htras, K,		/* B[k][n], num columnas (ldb) */
					0,				/* beta */ 
					WH, M			/* C[m][n], num columnas (ldc) */
				);
			}
			WH_total += (gettime() - WH_t);

			V_t = gettime();
			#pragma omp target teams distribute parallel for simd
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < M; j++) {
					WH[i*M + j] = V[i*M + j] / WH[i*M + j]; /* V./(W*H) */
				}
			}
			V_total += (gettime() - V_t);

			/* Waux =  {V./(W*H)} *H' */
			/* W = W .* Waux ./ accum_H */
			Wt_t = gettime();
			#pragma omp target variant dispatch use_device_ptr(WH, Htras, Waux)
			{
				cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
					N,				/* [m] */ 
					K, 				/* [n] */
					M,				/* [k] */
					1, 				/* alfa */ 
					WH, M,		 	/* A[m][k], num columnas (lda) */
					Htras, K,		/* B[k][n], num columnas (ldb) */
					0,				/* beta */ 
					Waux, K			/* C[m][n], num columnas (ldc) */
				);
			}
			Wt_total += (gettime() - Wt_t);

			/* Reducir a una columna */
			acc_t = gettime();
			#pragma omp target teams distribute parallel for simd
			for(int i = 0; i < K; i++) {
				acumm_H[i] = 0;
			}

			for (int j = 0; j < K; j++){
				#pragma omp target teams distribute parallel for simd reduction(+:acumm_H[j]) map(to:j)
				for (int i = 0; i < M; i++) {
					acumm_H[j] += Htras[i*K + j];
				}
			}
			acc_total += (gettime() - acc_t);

			mulM_t = gettime();
			#pragma omp target teams distribute parallel for simd
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < K; j++) {
					W[i*K + j] = W[i*K + j] * Waux[i*K + j] / acumm_H[j]; /* W = W .* Waux ./ accum_H */
				}
			}
			mulM_total += (gettime() - mulM_t);
		}
	}
	#pragma omp target exit data map(from:W[0:N*K], Htras[0:M*K])
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

	V                 = get_V(N, M, file_name);
	W                 = (C_REAL *) mkl_malloc(sizeof(C_REAL) * N*K, 64);
	Htras             = (C_REAL *) mkl_malloc(sizeof(C_REAL) * M*K, 64);
	WH                = (C_REAL *) mkl_malloc(sizeof(C_REAL) * N*M, 64);
	Haux              = (C_REAL *) mkl_malloc(sizeof(C_REAL) * M*K, 64);
	Waux              = (C_REAL *) mkl_malloc(sizeof(C_REAL) * N*K, 64);
	acumm_W           = (C_REAL *) mkl_malloc(sizeof(C_REAL) * K, 64); 
	acumm_H           = (C_REAL *) mkl_malloc(sizeof(C_REAL) * K, 64); 

    W_best              = (C_REAL *) mkl_malloc(sizeof(C_REAL) * N*K, 64);
    Htras_best          = (C_REAL *) mkl_malloc(sizeof(C_REAL) * M*K, 64);
    classification      = (unsigned char *) mkl_malloc(sizeof(unsigned char) * M, 64);
	last_classification = (unsigned char *) mkl_malloc(sizeof(unsigned char) * M, 64);
	consensus           = (unsigned char *) mkl_malloc(sizeof(unsigned char) * (M*(M-1)/2), 64);

	/**********************************/
	/******     MAIN PROGRAM     ******/
	/**********************************/
	time0 = gettime();

	for(int test = 0; test < nTests; test++) {

		/* Init W and H */
		initWH(W, Htras, N, M, K);

		niters = 2000 / NITER_TEST_CONV;

		stop   = 0;
		iter   = 0;
		inc    = 0;

		while(iter < niters && !stop) {
			iter++;

			/* Main Proccess of NMF Brunet */
			gpu_nmf(NITER_TEST_CONV, V, WH, W, 
				Htras, Waux, Haux, acumm_W, acumm_H,
				N, M, K);

			/* Adjust small values to avoid undeflow: h=max(h,eps);w=max(w,eps); */
			adjust_WH(W, Htras, N, M, K);

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
			<< "    W_mult_H time = " << WH_total << " (us) --> " << WH_total/nmf_total*100 << "%" << std::endl
			<< "    V_div_WH time = " << V_total << " (us) --> " << V_total/nmf_total*100 << "%" << std::endl
			<< "    accum time = " << acc_total << " (us) --> " << acc_total/nmf_total*100 << "%" << std::endl
			<< "    Wt_mult_WH = " << Wt_total << " (us) --> " << Wt_total/nmf_total*100 << "%" << std::endl
			<< "    mult_M_div_vect = " << mulM_total << " (us) --> " << mulM_total/nmf_total*100 << "%" << std::endl;

	printf("\n\n\n EXEC TIME %f (us).       N=%i M=%i K=%i Tests=%i (%lu)\n", time1-time0, N, M, K, nTests, sizeof(C_REAL));
	printf("Final error %e \n", error);

	/* Write the solution of the problem */
	writeSolution(W_best, Htras_best, consensus, N, M, K, nTests);

	//printMATRIX(W_best, N, K);

    /* Free memory used */
	mkl_free (V);
	mkl_free (W);
	mkl_free (Htras);
	mkl_free (WH);
	mkl_free (Haux);
	mkl_free (Waux);
	mkl_free (acumm_W);
	mkl_free (acumm_H);
	mkl_free (W_best);
	mkl_free (Htras_best);
	mkl_free (classification);
	mkl_free (last_classification);
	mkl_free (consensus);

	return 0;
}
