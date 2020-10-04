#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/times.h>
#include <malloc.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <CL/sycl.hpp>


using namespace cl::sycl;

//#define RANDOM
//const bool pinned_memory = true;
#define DEBUG
const bool verbose = false;
const char PAD = 32;

#if REAL <=4
	#define real float
	#define cblas_rgemm cblas_sgemm
	#define cblas_rdot cblas_sdot
	#define cblas_rcopy cblas_scopy
	#define rmax(a,b) ( ( (a) > (b) )? (a) : (b) )
	#define rsqrt sqrtf

/* Number of iterations before testing convergence (can be adjusted) */
const int NITER_TEST_CONV = 10;

/* Spacing of floating point numbers. */
const real eps = 2.2204e-16;

constexpr access::mode sycl_read       = access::mode::read;
constexpr access::mode sycl_write      = access::mode::write;
constexpr access::mode sycl_read_write = access::mode::read_write;
constexpr access::mode sycl_discard_read_write = access::mode::discard_read_write;
constexpr access::mode sycl_discard_write = access::mode::discard_write;

// CUDA device selector
class CUDASelector : public device_selector {
    public:
        int operator()(const device &Device) const override {
            const std::string DriverVersion = Device.get_info<info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                //std::cout << " CUDA device found " << std::endl;
                return 1;

            return -1;
        }
};


double gettime() {
	double final_time;
	struct timeval tv1;
	
	gettimeofday(&tv1, (struct timezone*)0);
	final_time = (tv1.tv_usec + (tv1.tv_sec)*1000000ULL);

	return(final_time);
}


void init_memory1D(int nx, buffer<real, 1> *buff) {
    auto memory = buff.get_access<sycl_write>();

	for(int i = 0; i < nx; i++)
		memory[i] = (real)(i*10);
}


unsigned char *get_memory1D_uchar(int nx) { 
	int i;
	unsigned char *buffer;	

	if( (buffer=(unsigned char *)malloc(nx*sizeof(int)))== NULL ) {
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

	// for( i=0; i<nx; i++ )
	// 	buffer[i] = (int)(0);

    std::fill(std::begin(buffer), std::end(buffer), 0);

	return( buffer );
}


void delete_memory1D_uchar( unsigned char *buffer ) { 
	free(buffer);
}


real **get_memory2D(int nx, int ny) {
	int i,j;
	real **buffer;

	if( (buffer=(real **)malloc(nx*sizeof(real *)))== NULL ) {
		fprintf( stderr, "ERROR in memory allocation\n" );
		return( NULL );
	}

	if( (buffer[0]=(real *)malloc(nx*ny*sizeof(real)))==NULL ) {
		fprintf( stderr, "ERROR in memory allocation\n" );
		free( buffer );
		return( NULL );
	}

	for (i = 1; i < nx; i++)
		buffer[i] = buffer[i-1] + ny;

	for(i = 0; i < nx; i++)
		for(j = 0; j < ny; j++)
			buffer[i][j] = (real)(i*100 + j);

	return( buffer );
}


void init_memory2D(int nx, int ny, buffer<real, 2> *buff) { 
	int i,j;
    auto memory = buff.get_access<sycl_read_write>();

	for(i = 1; i < nx; i++)
		memory[i] = memory[i-1] + ny;

	for(i = 0; i < nx; i++)
		for(j = 0; j < ny; j++)
			memory[i][j] = (real)(i*100 + j);
}


void delete_memory2D(real **buffer) { 
	free(buffer);
}


void matrix_copy1D_uchar(unsigned char *in, unsigned char *out, int nx) {
	for (int i = 0; i < nx; i++)
		out[i] = in[i];
}


void matrix_copy2D(buffer<real, 2> *b_in, real **out, int nx, int ny) {
	auto in = b_in.get_access<sycl_read>();
	
	for (int i = 0; i < nx; i++)
		for(int j = 0; j < ny; j++)
			out[i][j] = in[i][j];
}


void initWH(buffer<real, 2> *b_W, buffer<real, 2> *b_Htras, int N, int M, 
    int K, int Kpad)
{
	int i,j;
	int ii, jj;
	
    auto W = b_W.get_access<sycl_read>();
    auto Htras = b_Htras.get_access<sycl_read>();

	int seedi;
	FILE *fd;

	/* Generated random values between 0.00 - 1.00 */	
	fd = fopen("/dev/urandom", "r");
	fread(&seedi, sizeof(int), 1, fd); 
	fclose(fd);
	srand(seedi); 
	
	for (i = 0; i < N; i++) {
		for (j = 0; j < K; j++)
			W[i][j] = ((real)(rand()))/RAND_MAX;
			//W[i][j] = (real)(i);
		for (j = K; j < Kpad; j++)
			W[i][j] = 0.0;
	}

	for (i = 0; i < M; i++) {
        for (j = 0; j < K; j++)
			Htras[i][j] = ((real)(rand()))/RAND_MAX;
			//Htras[i][j] = (real)(i);
		for (j = K; j < Kpad; j++)
			Htras[i][j] = 0.0;
	}

#ifdef DEBUG
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
}


void printMATRIX(real **m, int I, int J) {
	int i, j;
	
	printf("--------------------- matrix --------------------\n");
	printf("             ");
	for (j=0; j<J; j++) {
		if (j<10)
			printf("%i      ", j);
		else if (j<100)
			printf("%i     ", j);
		else 
			printf("%i    ", j);
	}
	printf("\n");

	for (i=0; i<I; i++) {
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


void initV(int N, int M, char* file_name, buffer<real, 2> *b_V) {
	char *data;
	FILE *fIn;
	
	/* Local variables */
	int i, j;
	const int size_V = N*M;
    auto V = b_V.get_access<sycl_write>();

	fIn = fopen(file_name, "r");

#ifndef RANDOM 
	if (sizeof(real) == sizeof(float)) {
		fread(&V[0][0], sizeof(float), size_V, fIn);
		fclose(fIn);
	} 
    else {
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
	int seedi;
    fd = fopen("/dev/urandom", "r");
    fread( &seedi, sizeof(int), 1, fd);
    fclose (fd);
    srand( seedi );

    for (i=0; i<N; i++)
        for (j=0; j<M; j++)
            V[i][j] = ((real)(rand()))/RAND_MAX;

#endif
}


/* Gets the difference between matrix_max_index_h and conn_last matrices. */
int get_difference(unsigned char *classification, 
    unsigned char *last_classification, int nx)
{
	int diff;
	int conn, conn_last;
	
	diff = 0;
	for(int i = 0; i < nx; i++)
		for(int j = i+1; j < nx; j++) {
			conn = (int)( classification[j] == classification[i] );
			conn_last = (int) ( last_classification[j] == last_classification[i] );
			diff += ( conn != conn_last );
		}

	return (diff);
}


/* Get consensus from the classificacion vector */
void get_consensus(unsigned char *classification, unsigned char *consensus,
    int nx)
{
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
void get_classification(buffer<real, 2> *b_Htras, unsigned char *classification,
    int M, int K)
{
    auto Htras = b_Htras.get_access<sycl_read>();
	real max;
	
	for (int i = 0; i < M; i++) {
		max = 0.0;
		for (int j = 0; j < K; j++)
			if (Htras[i][j] > max) {
				classification[i] = (unsigned char)(j);
				max = Htras[i][j];
			}
	}
}


real get_Error(buffer<real, 2> *b_V, buffer<real, 2> *b_W, 
    buffer<real, 2> *b_Htras, int N, int M, int K) 
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

    auto V = b_V.get_access<sycl_read>();
    auto W = b_W.get_access<sycl_read>();
    auto Htras = b_Htras.get_access<sycl_read>();
    

	real error, tot_error;
	real Vnew;
	
	error=0.0;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++){
			Vnew = 0.0;
			for(int k = 0; k < K; k++)
				Vnew += W[i][k] * Htras[j][k];

			error += (V[i][j] - Vnew) * (V[i][j] - Vnew);
		}
	}
	
	return(error);
}


void writeSolution(real **W, real**Ht, unsigned char *consensus, int N, int M,
    int K, int nTests)
{
	FILE *fOut;
	char file[100];
	real **H;
	
	H = get_memory2D(K, M);
	for (int i = 0; i < K; i++)
		for (int j = 0; j < M; j++)
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


double tW0=0.0, tW1=0.0, tW2=0.0, tW3=0.0, tW4=0.0;
double tH0=0.0, tH1=0.0, tH2=0.0, tH3=0.0, tH4=0.0;


// TODO: updated to run in the device (add it in the kernel file)
void adjust_WH(buffer<real, 2> *b_W, buffer<real, 2> *b_Ht, int N, int M, int K) {
	auto W = b_W.get_access<sycl_read_write>();
    auto b_Ht = b_Ht.get_access<sycl_read_write>();
    
    int i, j;
	
	for (i = 0; i < N; i++)
		for (j = 0; j < K; j++)
			if (W[i][j] < eps)
				W[i][j] = eps;
				
	for (i = 0; i < M; i++)
		for (j = 0; j < K; j++)
			if (Ht[i][j] < eps)
				Ht[i][j] = eps;				 
}


// TODO: updated to run in the device
void nmf(int niter, real *d_V, real *d_WH, real *d_W, real *d_Htras, 
    real *d_Waux, real *d_Haux,
	real *d_accW, real *d_accH,
	int N, int M, int K, int Kpad)
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
	for (iter=0; iter<niter; iter++) {
		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/
        t0 = gettime();
                W_mult_H(d_WH, d_W, d_Htras, N, M, K, Kpad);	/* WH = W*H */
        t1 = gettime(); tH0+=t1-t0; t0 = gettime();
                V_div_WH(d_V, d_WH, N, M );			/* WH = (V./(W*H) */
        t1 = gettime(); tH1+=t1-t0; t0 = gettime();
                accum(d_accW, d_W, N, K, Kpad); 		/* Reducir a una columna */
        t1 = gettime(); tH2+=t1-t0; t0 = gettime();
                Wt_mult_WH(d_Haux, d_W, d_WH, N, M, K, Kpad);	/* Haux = (W'* {V./(WH)} */
        t1 = gettime(); tH3+=t1-t0; t0 = gettime();
                mult_M_div_vect(d_Htras, d_Haux, d_accW, M, K, Kpad);/* H = H .* (Haux) ./ accum_W */
        t1 = gettime(); tH4+=t1-t0;

                /*******************************************/
                /*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
                /*******************************************/
        t0 = gettime();
                W_mult_H(d_WH, d_W, d_Htras, N, M, K, Kpad);	/* WH = W*H */
        t1 = gettime(); tW0+=t1-t0; t0 = gettime();
                V_div_WH(d_V, d_WH, N, M );			/* WH = (V./(W*H) */
        t1 = gettime(); tW1+=t1-t0; t0 = gettime();
                WH_mult_Ht(d_Waux, d_WH, d_Htras, N, M, K, Kpad);/* Waux =  {V./(W*H)} *H' */
        t1 = gettime(); tW2+=t1-t0; t0 = gettime();
                accum(d_accH, d_Htras, M, K, Kpad);		/* Reducir a una columna */
        t1 = gettime(); tW3+=t1-t0; t0 = gettime();
                mult_M_div_vect(d_W, d_Waux, d_accH, N, K, Kpad);/* W = W .* Waux ./ accum_H */
        t1 = gettime(); tW4+=t1-t0;
    }
}


int main(int argc, char *argv[]) {
	int nTests, niters;
	int i,j;

	real **W_best, **Htras_best;
	unsigned char *classification, *last_classification;
	unsigned char *consensus;

	int N;
	int M;
	int K;
	int Kpad;
	int stop_threshold, stop;
	char file_name[255];
	int test, iter;
	int diff, inc;
	
	double time0, time1;
	
	real error;
	real error_old = 9.99e+50;

    setbuf( stdout, NULL );
	
	if (argc == 7) {
		strcpy(file_name, argv[1]);
		N              = atoi(argv[2]);
		M              = atoi(argv[3]);
		K              = atoi(argv[4]);
		Kpad           = K + (PAD - K % PAD);
		nTests         = atoi(argv[5]);
		stop_threshold = atoi(argv[6]);
	} 
    else {
	 	printf("./exec dataInput.bin N M K nTests stop_threshold (argc=%i %i)\n", argc, atoi(argv[2]));
		return(0);
	}

    printf("file=%s\nN=%i M=%i K=%i nTests=%i stop_threshold=%i\n", file_name, N, M, K, nTests, stop_threshold);

    buffer<real, 2> b_V{ range<2>{N, M} };
    buffer<real, 2> b_W{ range<2>{N, Kpad} };
    buffer<real, 2> b_Htras{ range<2>{M, Kpad} };
    buffer<real, 2> b_WH{ range<2>{N, M} };
    buffer<real, 2> b_Haux{ range<2>{M, Kpad} };
    buffer<real, 2> b_Waux{ range<2>{N, Kpad} };
    buffer<real, 1> b_acumm_W{ range<1>{Kpad} };
    buffer<real, 1> b_acumm_H{ range<1>{Kpad} };

    initV(N, M, file_name, &b_V);
    init_memory2D(N, Kpad, &b_W);
    init_memory2D(M, Kpad, &b_Htras);
    init_memory2D(N, M, &b_WH);
    init_memory2D(M, Kpad, &b_Haux);
    init_memory2D(N, Kpad, &b_Waux);
    init_memory1D(Kpad, &b_acumm_W);
    init_memory1D(Kpad, &b_acumm_H);

    W_best              = get_memory2D(N, Kpad);
    Htras_best          = get_memory2D(M, Kpad);
    classification      = get_memory1D_uchar(M);
	last_classification = get_memory1D_uchar(M);
	consensus           = get_memory1D_uchar(M*(M-1)/2);

	/**********************************/
	/******     MAIN PROGRAM     ******/
	/**********************************/
	time0 = gettime();

	for (test=0; test<nTests; test++) {
		/* Init W and H */
		initWH(&b_W, &b_Htras, N, M, K, Kpad);

		niters = 2000/NITER_TEST_CONV;

		stop   = 0;
		iter   = 0;
		inc    = 0;
		while (iter<niters && !stop) {
			iter++;

			/* Main Proccess of NMF Brunet */
			nmf(NITER_TEST_CONV, 
				d_V, d_WH, d_W, d_Htras, d_Waux, d_Haux, d_acumm_W, d_acumm_H,
				N, M, K, Kpad);

			/* Adjust small values to avoid undeflow: h=max(h,eps);w=max(w,eps); */
			adjust_WH(&b_W, &b_Htras, N, M, K, Kpad);

			/* Test of convergence: construct connectivity matrix */
			get_classification(&b_Htras, classification, M, K);

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
		error = get_Error(&b_V, &b_W, &b_Htras, N, M, K);
		if (error < error_old) {
			printf("Better W and H, Error %e Test=%i, Iter=%i\n", error, test, iter*NITER_TEST_CONV);
			matrix_copy2D(&b_W, W_best, N, K);
			matrix_copy2D(&b_Htras, Htras_best, M, K);
			error_old = error;
		}		
	}
	time1 = gettime();
	/**********************************/
	/**********************************/

	printf("\n\n\n EXEC TIME %f (s). (CPU2GPU %f s)      N=%i M=%i K=%i Tests=%i (%i)\n", (time1-time0)/1000000, timeGPU2CPU/1000000, N, M, K, nTests, sizeof(real));
	printf("tH0=%f tH1=%f tH2=%f tH3=%f\n", tH0/1000000, tH1/1000000, tH2/1000000, tH3/1000000);
	printf("tW0=%f tW1=%f tW2=%f tW3=%f\n", tW0/1000000, tW1/1000000, tW2/1000000, tW3/1000000);

	/* Write the solution of the problem */
	writeSolution(W_best, Htras_best, consensus, N, M, K, nTests);

    /* Free memory used */
	delete_memory2D(W_best);
	delete_memory2D(Htras_best);
	delete_memory1D_uchar(classification);
	delete_memory1D_uchar(last_classification);
	delete_memory1D_uchar(consensus);
}
