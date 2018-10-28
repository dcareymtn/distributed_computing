#ifndef GPU_ALGO_H
#define GPU_ALGO_H

namespace gpu
{

void hello_cuda();

void count_occurrences( double *M, int nRows, int nCols, int start_count, int stop_count );

}

#endif
