#include "kernels.h"
#include "cublas.h"
#include "stdio.h"


void W_mult_H(real *WH, real *W, real *Htras, int N, int M, int K)
{
	cublasRgemm( 'T', 'n', 
		M,				/* [m] */ 
		N,				/* [n] */  
		K,				/* [k] */ 
		1,				/* alfa */ 
		Htras, K,			/* A[m][k], num columnas (lda) */ 
		W, K,			/* B[k][n], num columnas (ldb) */
		0,				/* beta */
		WH, M				/* C[m][n], num columnas (ldc) */
	);
}


__global__ void V_div_WH_device( real* V, real* WH, int ny, int nx)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	WH[id] = V[id] / WH[id];
}


void V_div_WH( real* V, real* WH, int ny, int nx )
{
	int block_size = BLOCK_SIZE * BLOCK_SIZE;
	block_size = block_size < nx ? block_size : nx;
	int remainder = (nx == block_size) ? 0 : block_size - (nx % block_size);
	int threads = ny * (nx + remainder); 

	dim3 dimBlock(block_size);
	dim3 dimGrid(threads);

	V_div_WH_device<<<dimGrid, dimBlock>>>(V, WH, ny, nx);
}


void Wt_mult_WH( real *Haux, real *W, real *WH, int N, int M, int K)
{
	cublasRgemm( 'n', 'T', 
		K,				/* [m] */ 
		M,				/* [n] */  
		N,				/* [k] */ 
		1,				/* alfa */ 
		W, K,			/* A[m][k], num columnas (lda) */ 
		WH, M,				/* B[k][n], num columnas (ldb) */
		0,				/* beta */
		Haux, K			/* C[m][n], num columnas (ldc) */
	);
}

void WH_mult_Ht( real *Waux, real *WH, real *Htras, int N, int M, int K)
{
	cublasRgemm( 'n', 'n', 
		K,				/* [m] */ 
		N,				/* [n] */  
		M,				/* [k] */ 
		1,				/* alfa */ 
		Htras, K,			/* A[m][k], num columnas (lda) */ 
		WH, M,				/* B[k][n], num columnas (ldb) */
		0,				/* beta */
		Waux, K			/* C[m][n], num columnas (ldc) */
	);
}

__global__ void mult_M_div_vect_device(real *M, real *Maux, real *acc, int ny, int nx)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	int id = idy*nx+idx;

	// Make sure we do not go out of bounds
	if (idx<nx && idy<ny)
		M[id] = M[id]*Maux[id]/acc[idx];
}

void mult_M_div_vect(real *M, real *Maux, real *acc, int ny, int nx)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int a=nx/BLOCK_SIZE; if (nx % BLOCK_SIZE > 0) a++;
	int b=ny/BLOCK_SIZE; if (ny % BLOCK_SIZE > 0) b++;
	dim3 dimGrid(a,b);

	mult_M_div_vect_device<<<dimGrid, dimBlock>>>( M, Maux, acc, ny, nx);
}


__global__ void adjust_WH_device(real *M, int ny, int nx)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	int id = idy*nx+idx;

	// Make sure we do not go out of bounds
	if (idx<nx && idy<ny)
		if (M[id]<EPS)
			M[id] = EPS;
}

void adjust_WH_GPU(real *W, real *Htras, int N, int M, int K)
{
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	int a=K/BLOCK_SIZE;  if (K % BLOCK_SIZE > 0) a++;
	int bW=N/BLOCK_SIZE; if (N % BLOCK_SIZE > 0) bW++;
	int bH=M/BLOCK_SIZE; if (M % BLOCK_SIZE > 0) bH++;
	dim3 dimGridW(a,bW);
	dim3 dimGridH(a,bH);

	adjust_WH_device<<<dimGridW, dimBlock>>>( W, N, K);
	adjust_WH_device<<<dimGridH, dimBlock>>>( Htras, M, K);
}


__global__ void init_accum_device(real *acc)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	acc[idx] = 0;
}


__global__ void reduction_device(int n, int nx, int block_size, int threads, real* acc, real* X)
{
	extern __shared__ real scratch[];
	int local_id = threadIdx.x;
	int block_id = blockIdx.x;
	int blocks = 0;
	int offset;
	int global_id = local_id * (nx-1) + local_id + block_id;
	int global_id_offset;
	
	for(int i = 0; i < n; i += block_size){
		offset = threads * blocks;
		global_id_offset = global_id + offset;

		scratch[local_id] = X[global_id_offset];

		// Tree reduction
		for(int j = block_size / 2; j > 0; j >>= 1) {
			__syncthreads();

			if(local_id < j)
				scratch[local_id] += scratch[local_id + j];
		}

		if (local_id == 0)
			acc[block_id] += scratch[0];
		
		blocks++;
		__syncthreads();
	}
}


void accum( real* acc, real* X, int n, int nx)
{
	// init acc with 0s
	int block_size = BLOCK_SIZE < nx ? BLOCK_SIZE : nx;
	dim3 dimBlock1(block_size);
	dim3 dimGrid1(nx);
	init_accum_device<<<dimGrid1, dimBlock1>>>(acc);

	// reduction
	block_size = BLOCK_SIZE * BLOCK_SIZE;
	block_size = block_size < n ? block_size : n;
	int threads = block_size * nx;
	dim3 dimBlock2(block_size);
	dim3 dimGrid2(threads);
	reduction_device<<<dimGrid2, dimBlock2, block_size*sizeof(real)>>>(n, nx, block_size, threads, acc, X);
}

// __global__ void init_accum_device( real *acc, real *X, int n, int nx)
// {
// 	int idx = blockIdx.x*blockDim.x+threadIdx.x;
// 	int i;

// 	// Make sure we do not go out of bounds
// 	if (idx<nx){
// 		acc[idx] = 0.0;
// 		for (i=0; i<n; i++)
// 			acc[idx] += X[i*nx_pad+idx];
// 	}
// }

// void accum( real *acc, real* X, int n, int nx)
// {

// 	int i;

// 	dim3 dimBlock(BLOCK_SIZE);
// 	int a=nx/BLOCK_SIZE; if (nx % BLOCK_SIZE > 0) a++;
// 	dim3 dimGrid(a);

// 	/* Init acc with 0s */
// #if 1
// 	init_accum_device<<<dimGrid, dimBlock>>>( acc, X, n, nx);

// #else
// 	cublasRaxpy(nx,		/* n, num de elementos del vector*/
// 		0.0,		/* alpha*/
// 		acc, 1,		/* vector x, incx */
// 		acc, 0		/* vector y, incy */ 		
// 	);
// 	for (i=0; i<n; i++)
// 		cublasRaxpy(nx,		/* n, num de elementos del vector*/
// 			1,		/* alpha*/
// 			X+i*nx, 1,	/* vector x, incx */
// 			acc, 1		/* vector y, incy */ 		
// 		);

// #endif
// }
