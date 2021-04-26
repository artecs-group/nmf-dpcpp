#include "../common.h"

#ifdef REAL_S
	#define cblas_rgemm cblas_sgemm
#else
	#define cblas_rgemm cblas_dgemm
#endif

// void cpu_nmf(int niter, C_REAL *V, C_REAL *WH, 
// 	C_REAL *W, C_REAL *Htras, C_REAL *Waux, C_REAL *Haux,
// 	C_REAL *accW, C_REAL *accH, int N, int M, int K)
// {
// 	/*************************************/
// 	/*                                   */
// 	/*      Main Iterative Process       */
// 	/*                                   */
// 	/*************************************/
	
// 	for (int iter=0; iter<niter; iter++)
// 	{
	
// 		/*******************************************/
// 		/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
// 		/*******************************************/
// 		/* WH = W*H */

//         cblas_rgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
//             N,				/* [m] */ 
//             M,				/* [n] */
//             K,				/* [k] */
//             1, 				/* alfa */ 
//             Htras, K, 			/* A[m][k], num columnas (lda) */
//             W, K,		/* B[k][n], num columnas (ldb) */
//             0,				/* beta */ 
//             WH, M			/* C[m][n], num columnas (ldc) */
//         );

// 		for (int i=0; i<N; i++){
// 			for (int j=0; j<M; j++)
// 			{
// 				WH[i][j] = V[i][j]/WH[i][j]; /* V./(W*H) */
// 			}
// 		}

// 		/* Reducir a una columna */
// 		cblas_saxpyi(K, -1.0, acumm_W, 1, acumm_W);

// 		for (int i=1;i<N; i++)
// 			for (int j=0; j<K; j++)
// 				acumm_W[j] += W[i][j];

// 		cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
// 			K,				/* [m] */
// 			M,				/* [n] */
// 			N,				/* [k] */
// 			1,				/* alfa */
// 			W, K,			/* A[m][k], num columnas (lda) */
// 			WH, M,			/* B[k][n], num columnas (ldb) */
// 			0,                      	/* beta */
// 			Haux, K			/* C[m][n], num columnas (ldc) */
// 		);

// 		for (int j=0; j<M; j++){
// 			for (int i=0; i<K; i++)
// 				Htras[j][i] = Htras[j][i]*Haux[j][i]/acumm_W[i]; /* H = H .* (Haux) ./ accum_W */
// 		}

// 		/* Reducir a una columna */
// 		cblas_saxpyi(K, -1.0, acumm_H, 1, acumm_H);

// 		for (int i=1;i<M; i++)
// 			for (int j=0; j<K; j++)
// 			acumm_H[j] += Htras[i][j];

// 		/*******************************************/
// 		/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
// 		/*******************************************/


// 		/* WH = W*H */
// 		/* V./(W*H) */
// 		cblas_rgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
// 			M,				/* [m] */ 
// 			N, 				/* [n] */
// 			K,				/* [k] */
// 			1, 				/* alfa */ 
// 			Htras, K,		 	/* A[m][k], num columnas (lda) */
// 			W, K,		/* B[k][n], num columnas (ldb) */
// 			0,				/* beta */ 
// 			WH, M			/* C[m][n], num columnas (ldc) */
// 		);

// 		for (int i=0; i<N; i++) {
// 			for (int j=0; j<M; j++)
// 			{
// 				WH[i][j] = V[i][j]/WH[i][j]; /* V./(W*H) */
// 			}
// 		}

// 		/* Waux =  {V./(W*H)} *H' */
// 		/* W = W .* Waux ./ accum_H */
// 		cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
// 			K,				/* [m] */ 
// 			N, 				/* [n] */
// 			M,				/* [k] */
// 			1, 				/* alfa */ 
// 			Htras, K,		 	/* A[m][k], num columnas (lda) */
// 			WH, M,		/* B[k][n], num columnas (ldb) */
// 			0,				/* beta */ 
// 			Waux, K			/* C[m][n], num columnas (ldc) */
// 		);

// 		for (int i=0; i<N; i++)
// 		{
// 			for (int j=0; j<K; j++)
// 			{
// 				W[i][j] = W[i][j]*Waux[i][j]/acumm_H[j]; /* W = W .* Waux ./ accum_H */
// 			}
// 		}
// }


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

	#pragma omp target enter data map(alloc:WH[0:N*M], Waux[0:N*K], Haux[0:M*K], acumm_W[0:K], acumm_H[0:K]) \
	map(to:W[0:N*K], Htras[0:M*K], V[0:N*M])
	{
		for (int iter = 0; iter < niter; iter++) {
		
			/*******************************************/
			/*** H = H .* (W'*(V./(W*H))) ./ accum_W ***/
			/*******************************************/
			/* WH = W*H */
			#pragma omp task shared(Htras, W, WH) depend(out: WH)
			{
				#pragma omp target variant dispatch
				{
					cblas_rgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
						N,				/* [m] */ 
						M,				/* [n] */
						K,				/* [k] */
						1, 				/* alfa */ 
						Htras, K, 			/* A[m][k], num columnas (lda) */
						W, K,		/* B[k][n], num columnas (ldb) */
						0,				/* beta */ 
						WH, M			/* C[m][n], num columnas (ldc) */
					);
				}
			}
			//#pragma omptaskwait


			#pragma omp target teams distribute parallel for simd
			//num_teams(num_blocks)
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < M; j++) {
					WH[i*M + j] = V[i*M + j] / WH[i*M + j]; /* V./(W*H) */
				}
			}

			/* Reducir a una columna */
			#pragma omp target teams distribute parallel for simd
			for(int i = 0; i < K; i++) {
				acumm_W[i] = 0;
			}

			#pragma omp target teams distribute parallel for simd
			for (int i = 1; i < N; i++) {
				for (int j = 0; j < K; j++) {
					acumm_W[j] += W[i*K + j];
				}
			}

			#pragma omp task shared(W, WH, Haux) depend(out: Haux)
			{
				#pragma omp target variant dispatch
				{
					cblas_rgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
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
			}

			#pragma omp target teams distribute parallel for simd
			for (int j = 0; j < M; j++) {
				for (int i = 0; i < K; i++) {
					Htras[j*K + i] = Htras[j*K + i] * Haux[j*K + i] / acumm_W[i]; /* H = H .* (Haux) ./ accum_W */
				}
			}

			/* Reducir a una columna */
			#pragma omp target teams distribute parallel for simd
			for(int i = 0; i < K; i++) {
				acumm_H[i] = 0;
			}

			#pragma omp target teams distribute parallel for simd
			for (int i = 1; i < N; i++) {
				for (int j = 0; j < K; j++) {
					acumm_H[j] += Htras[i*K + j];
				}
			}

			/*******************************************/
			/*** W = W .* ((V./(W*H))*H') ./ accum_H ***/
			/*******************************************/


			/* WH = W*H */
			/* V./(W*H) */
			#pragma omp task shared(Htras, W, WH) depend(out: WH)
			{
				#pragma omp target variant dispatch
				{
					cblas_rgemm( CblasRowMajor, CblasTrans, CblasNoTrans, 
						M,				/* [m] */ 
						N, 				/* [n] */
						K,				/* [k] */
						1, 				/* alfa */ 
						Htras, K,		 	/* A[m][k], num columnas (lda) */
						W, K,		/* B[k][n], num columnas (ldb) */
						0,				/* beta */ 
						WH, M			/* C[m][n], num columnas (ldc) */
					);
				}
			}

			#pragma omp target teams distribute parallel for simd
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < M; j++) {
					WH[i*M + j] = V[i*M + j] / WH[i*M + j]; /* V./(W*H) */
				}
			}

			/* Waux =  {V./(W*H)} *H' */
			/* W = W .* Waux ./ accum_H */
			#pragma omp task shared(Htras, WH, Waux) depend(out: Waux)
			{
				#pragma omp target variant dispatch
				{
					cblas_rgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 
						K,				/* [m] */ 
						N, 				/* [n] */
						M,				/* [k] */
						1, 				/* alfa */ 
						Htras, K,		 	/* A[m][k], num columnas (lda) */
						WH, M,		/* B[k][n], num columnas (ldb) */
						0,				/* beta */ 
						Waux, K			/* C[m][n], num columnas (ldc) */
					);
				}
			}


			#pragma omp target teams distribute parallel for simd
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < K; j++) {
					W[i*K + j] = W[i*K + j] * Waux[i*K + j] / acumm_H[j]; /* W = W .* Waux ./ accum_H */
				}
			}
		}
	}
	#pragma omp target exit data map(from:W[0:N*K], Htras[0:M*K])
}