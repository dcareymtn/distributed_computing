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

__global__ void gpu_count_occurrences( double * pM, int size, int start_index, int *d_counter )
{

	int tid 	= blockIdx.x * blockDim.x + threadIdx.x;
	
	// Check that the thread ID is in the matrix
	if (tid >= size) 
	{ 
		return;
	}

	int bin = pM[tid] - start_index;
	
	atomicAdd( &d_counter[bin], 1 );
}

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count )
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
	
	int blockSize = nCols;
	int nBlock = N / blockSize + (N%blockSize == 0 ? 0 : 1);

	gpu_count_occurrences<<< nBlock, blockSize >>>(d_M, size, start_count, d_counter);

	cudaMemcpy( h_counter, d_counter, counter_size, cudaMemcpyDeviceToHost);

	for (int ii = start_count; ii <= stop_count; ii++)
	{
		printf("%d ", ii);
	}
	printf("\n");
	for (int ii = 0; ii < nbins; ii++)
	{
		printf("%d ", h_counter[ii]);
	}
	printf("\n");

	free(h_counter);
	cudaFree(d_M);
	cudaFree(d_counter);
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
