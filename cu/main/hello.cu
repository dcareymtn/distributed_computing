#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void print(char *a, int N)
{
    char p[11]="Hello CUDA";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N) 
    {
        a[idx]=p[idx];
    }
}

int main(void)
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

    print <<< nblock, blocksize >>>(a_d, N); // Run the kernel on the device

    // copy from the device memory to the host memory
    cudaMemcpy(a_h, a_d, sizeof(char) * N, cudaMemcpyDeviceToHost);

    // print the array on the host
    for (int i = 0; i < N; i++)
    {
        printf("%c", a_h[i]);
    }

    // Clean
    free(a_h);
    cudaFree(a_d);
}
