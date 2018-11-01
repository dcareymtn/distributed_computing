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

__global__ void gpu_count_occurrences( double * pM, int start_index, int stop_index )
{

	// Get the number of points
	int nCount 	= stop_index - start_index + 1;

	// Get the row and column in the larger array
	int x 		= blockIdx.x * blockDim.x + threadIdx.x;
	int y 		= blockIdx.y * blockDim.y + threadIdx.y;

	// Get the grid dimension
	int nx 		= blockDim.x * gridDim.x;
	int ny 		= blockDim.y * gridDim.y;
	int npix 	= nx * ny;

	// Get linear thread Index in the block
	int t 	= threadIdx.x + threadIdx.y * blockDim.x;

	// Get total number of threads in 2D block
	int nt = blockDim.x * blockDim.y;
	
	// Get the linear block index within the 2D grid
	int g = blockIdx.x + blockIdx.y * gridDim.x;

	// Initialize temporary accumulation array in global memory
	unsigned int *gmem = out + g * nCount;

	for (int i = t; i < npix; i += nt) 
	{
		gmem[i] = 0;
	}

	for (int col = x; col < nCols; col += nx)
	{
		for (int row = y; row < nRows; row+= ny)
		{
			double val = pM[row * width + col];
			atomicAdd(&gmem[pM[ row * width + col]
		}

}

void count_occurrences( double *M, int nRows, int nCols, int start_count, int stop_count )
{
	
	// Copy the matrix data to the gpu
	

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
