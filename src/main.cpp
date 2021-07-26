#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "common.hpp"
#include "./kernels/kernels.hpp"
#include "./queue_data/queue_data.hpp"

int IntelGPUSelector::gpus_taken = 0;
int IntelGPUSelector::gpu_counter = 0;


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
		W[i] = ((C_REAL)(rand())) / ((C_REAL) RAND_MAX);

	for (int i = 0; i < M*K; i++)
		Htras[i] = ((C_REAL)(rand())) / ((C_REAL) RAND_MAX);
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


void init_V(C_REAL *V, char* file_name, int n_queues, queue_data* qd) {
	int N = qd[0].N;
	int M = qd[0].M;

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
        V[i] = ((C_REAL)(rand()))/ ((C_REAL) RAND_MAX);
#endif

	// copy V by columns
	int M_acc{0};
	int M_split{0};

	for(int q = 0; q < n_queues; q++) {
		M_split = qd[q].M_split;

		for (int i = 0; i < N; i++)
			for (int j = 0; j < M_split; j++)
				qd[q].V_col[i*M_split + j] = V[i*M + j + M_acc];

		M_acc += M_split;
	}

	// copy V by rows
	int pad{0};
	for(int q = 0; q < n_queues; q++) {
		for(int i = 0; i < qd[q].N_split*M; i++)
			qd[q].V_row[i] = V[pad + i];

		//std::copy(V + pad, V + (qd[q].N_split*M), qd[q].V_row);
		pad += qd[q].N_split * M;
	}
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


void copy_WH_to(int n_queues, queue_data* qd) {
	int N = qd[CPU_QUEUE_IND].N;
	int M = qd[CPU_QUEUE_IND].M;
	int K = qd[CPU_QUEUE_IND].K;

	std::copy(qd[CPU_QUEUE_IND].W, qd[CPU_QUEUE_IND].W + (N*K), qd[IGPU_QUEUE_IND].W);
	std::copy(qd[CPU_QUEUE_IND].Htras, qd[CPU_QUEUE_IND].Htras + (M*K), qd[IGPU_QUEUE_IND].Htras);
}


void copy_WH_from(int n_queues, queue_data* qd) {
	int N = qd[0].N;
	int M = qd[0].M;
	int K = qd[0].K;
	C_REAL* W = qd[CPU_QUEUE_IND].W;
	C_REAL* Htras = qd[CPU_QUEUE_IND].Htras;

	std::copy(qd[IGPU_QUEUE_IND].W , qd[IGPU_QUEUE_IND].W + (qd[IGPU_QUEUE_IND].N_split * K), W);
	std::copy(qd[IGPU_QUEUE_IND].Htras, qd[IGPU_QUEUE_IND].Htras + (qd[IGPU_QUEUE_IND].M_split * K), Htras);
}


void sync_queues(int queues, queue_data* qd) {
	for (size_t i = 0; i < queues; i++)
		qd[i].q.wait();
}


void nmf(int niter, int n_queues, queue_data* qd) {
	/*************************************/
	/*                                   */
	/*      Main Iterative Process       */
	/*                                   */
	/*************************************/
	int padding{0};

	for (int iter = 0; iter < niter; iter++) {
		/*******************************************/
		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
		/*******************************************/

        /* WH = W*H */
		padding = 0;
		for(int i = 0; i < n_queues; i++){
			W_mult_H(qd[i].q, qd[i].WH_col, qd[i].W, qd[i].Htras + padding, qd[i].N, qd[i].M_split, qd[i].K);
			padding += qd[i].M_split * qd[i].K;
		}
		qd[CPU_QUEUE_IND].q.wait();

		/* WH = (V./(W*H) */
		for(int i = 0; i < n_queues; i++)
			V_div_WH(qd[i].q, qd[i].V_col, qd[i].WH_col, qd[i].N, qd[i].M_split);
		
		qd[CPU_QUEUE_IND].q.wait();

		/* Shrink into one column */
		padding = 0;
		for(int i = 0; i < n_queues; i++) {
        	accum(qd[i].q, qd[i].accW, qd[i].W + padding, qd[i].N_split, qd[i].K);
			padding += qd[i].N_split * qd[i].K;
		}
		qd[CPU_QUEUE_IND].q.wait();

		/* Haux = (W'* {V./(WH)} */
		for(int i = 0; i < n_queues; i++)
			Wt_mult_WH(qd[i].q, qd[i].Haux, qd[i].W, qd[i].WH_col, qd[i].N, qd[i].M_split, qd[i].K);

		qd[CPU_QUEUE_IND].q.wait();

		/* H = H .* (Haux) ./ accum_W */
		padding = 0;
		for(int i = 0; i < n_queues; i++) {
        	mult_M_div_vect(qd[i].q, qd[i].Htras + padding, qd[i].Haux, qd[i].accW, qd[i].M_split, qd[i].K);
			padding += qd[i].M_split * qd[i].K;
		}

		/* gather and scatter H */
		sync_queues(n_queues, qd);

		std::copy(
			qd[IGPU_QUEUE_IND].Htras, 
			qd[IGPU_QUEUE_IND].Htras + (qd[IGPU_QUEUE_IND].M_split * qd[IGPU_QUEUE_IND].K), 
			qd[CPU_QUEUE_IND].Htras);

		std::copy(
			qd[CPU_QUEUE_IND].Htras, 
			qd[CPU_QUEUE_IND].Htras + (qd[CPU_QUEUE_IND].M * qd[CPU_QUEUE_IND].K),
			qd[IGPU_QUEUE_IND].Htras);

		/*******************************************/
		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
		/*******************************************/

		/* WH = W*H */
		padding = 0;
		for(int i = 0; i < n_queues; i++) {
			W_mult_H(qd[i].q, qd[i].WH_row, qd[i].W + padding, qd[i].Htras, qd[i].N_split, qd[i].M, qd[i].K);
			padding += qd[i].N_split * qd[i].K;
		}
		qd[CPU_QUEUE_IND].q.wait();

		/* WH = (V./(W*H) */
		for(int i = 0; i < n_queues; i++)
			V_div_WH(qd[i].q, qd[i].V_row, qd[i].WH_row, qd[i].N_split, qd[i].M);

		qd[CPU_QUEUE_IND].q.wait();

		/* Waux =  {V./(W*H)} *H' */
		for(int i = 0; i < n_queues; i++)
        	WH_mult_Ht(qd[i].q, qd[i].Waux, qd[i].WH_row, qd[i].Htras, qd[i].N_split, qd[i].M, qd[i].K);

		qd[CPU_QUEUE_IND].q.wait();

		/* Shrink into one column */
		padding = 0;
		for(int i = 0; i < n_queues; i++) {
        	accum(qd[i].q, qd[i].accH, qd[i].Htras + padding, qd[i].M_split, qd[i].K);
			padding += qd[i].M_split * qd[i].K;
		}
		qd[CPU_QUEUE_IND].q.wait();

		/* W = W .* Waux ./ accum_H */
		padding = 0;
		for(int i = 0; i < n_queues; i++) {
			mult_M_div_vect(qd[i].q, qd[i].W + padding, qd[i].Waux, qd[i].accH, qd[i].N_split, qd[i].K);
			padding += qd[i].N_split * qd[i].K;
		}

		/* gather and scatter W */
		sync_queues(n_queues, qd);

		std::copy(
			qd[IGPU_QUEUE_IND].W, 
			qd[IGPU_QUEUE_IND].W + (qd[IGPU_QUEUE_IND].N_split * qd[IGPU_QUEUE_IND].K), 
			qd[CPU_QUEUE_IND].W);

		std::copy(
			qd[CPU_QUEUE_IND].W,
			qd[CPU_QUEUE_IND].W + (qd[CPU_QUEUE_IND].N * qd[CPU_QUEUE_IND].K),
			qd[IGPU_QUEUE_IND].W);
    }
	/* Adjust small values to avoid undeflow: h=max(h,eps);w=max(w,eps); */
	padding = 0;
	int padding2{0};
	for(int i = 0; i < n_queues; i++) {
		adjust_WH(qd[i].q, qd[i].W + padding, qd[i].Htras + padding2, qd[i].N_split, qd[i].M_split, qd[i].K);
		padding += qd[i].N_split * qd[i].K;
		padding2 += qd[i].M_split * qd[i].K;
	}

	sync_queues(n_queues, qd);
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
	//int N_pad = pow2roundup(N);
	//int M_pad = pow2roundup(M);
	
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
	int M              = atoi(argv[3]);
	int K              = atoi(argv[4]);
	int nTests         = atoi(argv[5]);
	int stop_threshold = atoi(argv[6]);

    printf("file=%s N=%i M=%i K=%i nTests=%i stop_threshold=%i\n\n", file_name, N, M, K, nTests, stop_threshold);

	int n_queues{2};

	// split N and M into the
	int N_1 = N / 2;
	int N_2 = N - N_1;
	int M_1 = M / 2;
	int M_2 = M - M_1;

	// create all the queue_data
	queue_data qd[] = {
		queue_data(N, N_1, M, M_1, K, "IntelGPU"),
		queue_data(N, N_2, M, M_2, K, "cpu"),
	};

	for(int i = 0; i < n_queues; i++)
		std::cout << "Running on "
				  << qd[i].q.get_device().get_info<sycl::info::device::name>()
				  << std::endl << std::endl;

	// host variables
	V                   = new C_REAL[N*M];
    W_best              = new C_REAL[N*K];
    Htras_best          = new C_REAL[M*K];
    classification      = new unsigned char[M];
	last_classification = new unsigned char[M];
	consensus           = new unsigned char[M*(M-1)/2];

	init_V(V, file_name, n_queues, qd);
	/**********************************/
	/******     MAIN PROGRAM     ******/
	/**********************************/
	time0 = gettime();

	for(int test = 0; test < nTests; test++) {
		/* Copy W and H to devices*/
		initWH(N, M, K, qd[CPU_QUEUE_IND].W, qd[CPU_QUEUE_IND].Htras);
		copy_WH_to(n_queues, qd);

		niters = 2000 / NITER_TEST_CONV;
		stop   = 0;
		iter   = 0;
		inc    = 0;
		while(iter < niters && !stop) {
			iter++;

			/* Main Proccess of NMF Brunet */
			nmf(NITER_TEST_CONV, n_queues, qd);

			/* Copy back W and H from devices*/
			copy_WH_from(n_queues, qd);

			/* Test of convergence: construct connectivity matrix */
			get_classification(qd[CPU_QUEUE_IND].Htras, classification, M, K);

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

		printMATRIX(qd[CPU_QUEUE_IND].Htras, M, K);
		printMATRIX(qd[CPU_QUEUE_IND].W, N, K);
		return 0;

		/* Get Matrix consensus */
		get_consensus(classification, consensus, M);

		/* Get variance of the method error = |V-W*H| */
		error = get_Error(V, qd[CPU_QUEUE_IND].W, qd[CPU_QUEUE_IND].Htras, N, M, K);
		if (error < error_old) {
			printf("Better W and H, Error %e Test=%i, Iter=%i\n", error, test, iter);
			matrix_copy2D(qd[CPU_QUEUE_IND].W, W_best, N, K);
			matrix_copy2D(qd[CPU_QUEUE_IND].Htras, Htras_best, M, K);
			error_old = error;
		}
	}
	time1 = gettime();
	/**********************************/
	/**********************************/

	printf("\n\n\nEXEC TIME %f (us).       N=%i M=%i K=%i Tests=%i (%lu)\n", time1-time0, N, M, K, nTests, sizeof(C_REAL));
	printf("Final error %e \n", error);
	
	/* Write the solution of the problem */
	//writeSolution(W_best, Htras_best, consensus, N, M, K, nTests);

	//printMATRIX(W_best, N, K);

    /* Free memory used */
	delete[] V;
	delete[] W_best;
	delete[] Htras_best;
	delete[] classification;
	delete[] last_classification;
	delete[] consensus;

	return 0;
}
