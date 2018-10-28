#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "gpu_algo.h"

int main(void)
{
	printf("Hello Cpp\n");
	gpu::hello_cuda();
}
