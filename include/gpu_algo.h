#ifndef GPU_ALGO_H
#define GPU_ALGO_H

#include <cuda.h>

namespace gpu
{

void hello_cuda();

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count );

}

#endif
