#include <omp.h>
#include <math.h>

#include "matrix.h"
#include "gpu_algo.h"

int main(int argc, char **argv)
{ 
	double start_s, stop_s;
	int counterMRows(256), counterMCols(256);
	int start_count(0), stop_count(10);
	Matrix M = Matrix::randi(counterMRows,counterMCols,start_count,stop_count);

	double *pM = (double *) malloc(counterMRows*counterMCols*sizeof(double));
	M.copy_to_cptr( pM );	

	start_s = omp_get_wtime();
	gpu::cuda_init();
	stop_s 	= omp_get_wtime();
	printf("CUDA Initialization Took: %f\n", stop_s - start_s);
	
	start_s = omp_get_wtime();
	gpu::count_occurrences( pM, counterMRows, counterMCols, start_count, stop_count);
	stop_s = omp_get_wtime();
	printf("Compute Time: %1.20f\n", stop_s - start_s );


	start_s = omp_get_wtime();
	int temp_counter[stop_count - start_count + 1];
	for (int i = 0; i < 128*128; i++)
	{
		int temp_bin = pM[i] - start_count;
		temp_counter[temp_bin] += 1;
	}
	stop_s = omp_get_wtime();
	printf("Compute Time: %1.20f\n", stop_s - start_s );


	/***********************************************\
	 * Part 2
	\***********************************************/

	printf(" ------------- Part 2 ------------------\n");

	int filtMRows(4), filtMCols(4); // Matrix Size
	int filtNRows(3), filtNCols(3); // Filter Size
	int filt_start_count(0), filt_stop_count(5);
	int nRowBreak = 2;

	Matrix M2 = Matrix::randi( filtMRows, filtMCols, filt_start_count, filt_stop_count );

	M2.write(stdout);
	
	int blockSize;
	int subMatNumRows;
	int subMatNumCols;

	M2.getParFiltBlockSize( nRowBreak, filtNRows, filtNCols, blockSize, subMatNumRows, subMatNumCols );

	double *pBlockM 		= (double *) malloc(blockSize * sizeof(double) );
	double *pBlockMResult 	= (double *) malloc(blockSize * sizeof(double) );

	M2.copy_to_c_zero_padded_blocks( pBlockM, nRowBreak, filtNRows, filtNCols );

	gpu::rms_filter( pBlockMResult, pBlockM, nRowBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols );	

	
	start_s = omp_get_wtime();
	Matrix MResult( pBlockMResult, nRowBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols );
	stop_s = omp_get_wtime();
	printf("Compute Time: %1.20f\n", stop_s - start_s );
	
	MResult.write(stdout);

	free(pM);
	free(pBlockM);	
	free(pBlockMResult);

	return 0;

}
