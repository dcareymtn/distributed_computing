#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "gpu_algo.h"

namespace gpu
{

__global__ void print_cuda(char *a, int N)
{
    char p[11]="Hello CUDA";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("Hello\n");
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

__global__ void gpu_count_occurrences_shared( double * pM, int size, int start_index, int stop_index, int *d_counter_partial )
{

	extern __shared__ int s_counter[];

	int tid 	= blockIdx.x * blockDim.x + threadIdx.x;
	int N = stop_index - start_index + 1;

	if (blockIdx.x < (stop_index - start_index + 1))
	{	
		s_counter[blockIdx.x] = 0;
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
		atomicAdd( &d_counter_partial[threadIdx.x], s_counter[threadIdx.x] );
	}
	//d_counter_partial[blockIdx.x*N + threadIdx.x] = s_counter[threadIdx.x];

}

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count )
{
	
	// Copy the matrix data to the gpu
	double *d_M;
	int *d_counter_partial;
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
	
	int blockSize = nCols;
	int nBlock = N / blockSize + (N%blockSize == 0 ? 0 : 1);
	
	//cudaMalloc((void **)&d_counter_partial, nBlock*counter_size);
	
	//gpu_count_occurrences_global<<< nBlock, blockSize >>>( d_M, size, start_count, stop_count, d_counter );
	int test[nbins];
	size_t testsize = nbins*sizeof(int);

	printf("Size of array = %d\n", sizeof(test));
	printf("Sizeof array  = %d\n", testsize );
	gpu_count_occurrences_shared<<< nBlock, blockSize, 64*sizeof(int)  >>>(d_M, size, start_count, stop_count, d_counter );
	
	// gpu_combine_count<<< nRows, nbins >>> (d_counter_partial, start_count, stop_count, d_counter );

	cudaMemcpy( h_counter, d_counter, counter_size, cudaMemcpyDeviceToHost);

	for (int ii = start_count; ii <= stop_count; ii++)
	{
		printf("%4d | ", ii);
	}
	printf("\n");
	for (int ii = 0; ii < nbins; ii++)
	{
		printf("%3d |  ", h_counter[ii]);
	}
	printf("\n");

	free(h_counter);
	cudaFree(d_M);
	cudaFree(d_counter);
	//cudaFree(d_counter_partial);
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

}
