#ifndef GPU_ALGO_H
#define GPU_ALGO_H

namespace gpu
{

void hello_cuda();

void cuda_init();

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count );

void rms_filter( double *hBlockMResult, double *_h_BlockM, int nRowBreak, int subMatNumRows, int subMatNumCols, int filtNRows, int filtNCols ); 

}

#endif
