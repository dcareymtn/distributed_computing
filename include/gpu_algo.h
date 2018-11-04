#ifndef GPU_ALGO_H
#define GPU_ALGO_H

#include <cuda.h>

namespace gpu
{

void hello_cuda();

void cuda_init();

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count );

void rms_filter( double *h_M, int filtNRows, int filtNCols ); 

}

#endif
