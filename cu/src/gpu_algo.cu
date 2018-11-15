#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#include "gpu_algo.h"

namespace gpu
{

__global__ void print_cuda(char *a, int N)
{
    char p[11]="Hello CUDA";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if(idx < N) 
    {
        a[idx]=p[idx];
    }
}

__global__ void gpu_count_occurrences_global( double *dM, int size, int start_index, int stop_index, int *d_counter )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid > size) return;

	int bin = dM[tid] - start_index;

	atomicAdd( &d_counter[bin], 1);

}

__global__ void gpu_count_occurrences_shared( double * pM, int size, int start_index, int stop_index, int *d_counter )
{

	extern __shared__ int s_counter[];

	int tid 	= blockIdx.x * blockDim.x + threadIdx.x;
	int N = stop_index - start_index + 1;

	if (tid < (stop_index - start_index + 1))
	{	
		s_counter[tid] = 0;
		d_counter[tid] = 0;
	}

	__syncthreads();
	
	if (tid < size)
	{
		int sbin = pM[tid] - start_index;
		atomicAdd( &s_counter[sbin], 1);
	}

	__syncthreads();

	if ((threadIdx.x < N) && (tid < size))
	{
		atomicAdd( &d_counter[threadIdx.x], s_counter[threadIdx.x] );
	}

}

__global__ void gpu_rms_filter_shared( double * _d_M, int filtNRows, int filtNCols, double * _d_MResult )
{
	
	extern __shared__ int _l_M[];
	
	int subMatIdx = blockIdx.x;
	
	int blockSize	 = blockDim.x * blockDim.y;

	int row = threadIdx.x;
	int col = threadIdx.y;

	_l_M[subMatIdx * blockSize + row * blockDim.y + col] = *(_d_M + subMatIdx * blockSize + row * blockDim.y + col );

	__syncthreads();

	double this_result(0);

	int start_row 	= max(row - (filtNRows - 1)/2, 0);
	int stop_row 	= min(row + (filtNRows - 1)/2, blockDim.x);
	
	int start_col 	= max(col - (filtNCols - 1)/2, 0);
	int stop_col 	= min(col + (filtNCols - 1)/2, blockDim.y);

	double temp(0);

	for (int iRow = start_row; iRow <= stop_row; iRow++)
	{
		for (int iCol = start_col; iCol <= stop_col; iCol++)
		{

			temp 		= _l_M[ subMatIdx * blockSize + iRow * blockDim.y + iCol ];
			this_result += temp*temp;

		}	
	}
	
	*(_d_MResult + subMatIdx * blockSize + row * blockDim.y + col ) = sqrt( this_result );
}

__global__ void gpu_rms_filter_global( double * _d_M, int filtNRows, int filtNCols, double * _d_MResult )
{
	int subMatIdx = blockIdx.x;
	
	int blockSize	 = blockDim.x * blockDim.y;

	int row = threadIdx.x;
	int col = threadIdx.y;

	double this_result(0);

	int start_row 	= max(row - (filtNRows - 1)/2, 0);
	int stop_row 	= min(row + (filtNRows - 1)/2, blockDim.x);
	
	int start_col 	= max(col - (filtNCols - 1)/2, 0);
	int stop_col 	= min(col + (filtNCols - 1)/2, blockDim.y);

	double temp(0);

	for (int iRow = start_row; iRow <= stop_row; iRow++)
	{
		for (int iCol = start_col; iCol <= stop_col; iCol++)
		{
			temp 		= *(_d_M + subMatIdx * blockSize + iRow * blockDim.y + iCol );
			this_result +=  temp*temp;
		}	
	}
	
	*(_d_MResult + subMatIdx * blockSize + row * blockDim.y + col ) =  sqrt( this_result );
}

void cuda_init()
{
	double *dM;
	cudaMalloc( (void **) &dM, 10*sizeof(double));
	cudaFree(dM);
}

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count, bool bGlobal )
{
	
	// Copy the matrix data to the gpu
	double *d_M;
	int *d_counter;
		
	int nbins 	= stop_count - start_count + 1;

	size_t counter_size = nbins * sizeof( int );
	int *h_counter = (int *) malloc( counter_size );

	for (int ii = 0; ii < nbins; ii++)
	{
		h_counter[ii] = 0;
	}

	int N = nRows * nCols;

	size_t size = N * sizeof( double );

	cudaMalloc((void **)&d_M, size);
	cudaMalloc((void **)&d_counter, counter_size);

	cudaMemcpy( d_M, h_M, size, cudaMemcpyHostToDevice);
	
	int factor = 4;

	int blockSize = (factor * nCols < 1000 ? factor * nCols : 1000);
	int nBlock = N / blockSize + (N%blockSize == 0 ? 0 : 1);

	if (bGlobal)
	{
		gpu_count_occurrences_global<<< nBlock, blockSize  >>>(d_M, size, start_count, stop_count, d_counter );
	}
	else
	{
		gpu_count_occurrences_shared<<< nBlock, blockSize, nbins*sizeof(int)  >>>(d_M, size, start_count, stop_count, d_counter );
	}


	cudaMemcpy( h_counter, d_counter, counter_size, cudaMemcpyDeviceToHost);

	//for (int ii = start_count; ii <= stop_count; ii++)
	//{
	//	printf("%4d | ", ii);
	//}
	//printf("\n");
	//for (int ii = 0; ii < nbins; ii++)
	//{
	//	printf("%3d |  ", h_counter[ii]);
	//}
	//printf("\n");

	free(h_counter);
	cudaFree(d_M);
	cudaFree(d_counter);
}	

void rms_filter( double *hBlockMResult, double *_h_BlockM, int nRowBreak, int nColBreak, int subMatNumRows, int subMatNumCols, int nFiltRows, int nFiltCols, bool bGlobal )
{
	double *_d_BlockM, *_d_BlockMResult, *_d_BlockMResultShared;

	size_t __blockSize = nRowBreak * nColBreak * subMatNumRows * subMatNumCols * sizeof(double);

	cudaMalloc((void **)&_d_BlockM, __blockSize );
	cudaMalloc((void **)&_d_BlockMResult, __blockSize );
	cudaMalloc((void **)&_d_BlockMResultShared, __blockSize );

	cudaMemcpy( _d_BlockM, _h_BlockM, __blockSize, cudaMemcpyHostToDevice );

	dim3 threadsPerBlock(subMatNumRows, subMatNumCols );

	if (bGlobal)
	{
		gpu_rms_filter_global<<< nRowBreak * nColBreak, threadsPerBlock >>>( _d_BlockM, nFiltRows, nFiltCols, _d_BlockMResult );
	}
	else
	{
		gpu_rms_filter_shared<<< nRowBreak * nColBreak, threadsPerBlock, __blockSize  >>>( _d_BlockM, nFiltRows, nFiltCols, _d_BlockMResult );	
	}

	cudaMemcpy( hBlockMResult, _d_BlockMResult, __blockSize, cudaMemcpyDeviceToHost );
	
	cudaFree(_d_BlockM);
	cudaFree(_d_BlockMResult);
	cudaFree(_d_BlockMResultShared);	
}

void hello_cuda(void)
{
    char *a_h, *a_d; // _h for the host and _d for the device based pointers
    const int N = 11;
    size_t size = N * sizeof(char);

    a_h = (char *) malloc(size); // allocating the array on the host
    cudaMalloc((void **) &a_d, size); // allocating the array on the device
    
    // initialize the host array
    for (int i = 0; i < N; i++)
    {
        a_h[i] = 0;
    }

    // Copy the array on the host to the device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

    int blocksize = 4;
    int nblock = N/blocksize + (N%blocksize == 0 ? 0 : 1);
	nblock = 100;

    print_cuda <<< nblock, blocksize >>>(a_d, N); // Run the kernel on the device

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
    // copy from the device memory to the host memory
    cudaMemcpy(a_h, a_d, sizeof(char) * N, cudaMemcpyDeviceToHost);


	// print the array on the host
    for (int i = 0; i < N; i++)
    {
        printf("%c", a_h[i]);
    }
	printf("\n");
    free(a_h);
	cudaFree(a_d);
}


void particle_swarm_eval( double (*f)(int dim, double * vec), 
							int dim, 
							int numParticles, 
							double * pos_vec_array, 
							double * vel_vec_array, 
							double a_1, double a_2, 
							double * P_b, double * P_g, 
							double *next_pos_vec_array)
{
	// Initialize the particles

	// Using the current position of the particles (from pos_vec_array), compute the score at each particle

	// Using the current position of the particles (from pos_vec_array), Update the Personal best for each particle

	// Of all the particles, do a maximum reduction on global data to find the global max

	// Randomly generate the two random vectors [0,1]

	// Move the particles and update the positions

	// Compute the convergence metric

	// If done, then exit

	// Else, repeat up to max num times
	

}

}
