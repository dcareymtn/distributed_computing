#include <omp.h>
#include <math.h>

#include "matrix.h"
#include "gpu_algo.h"
#include "algo.h"

int main(int argc, char **argv)
{ 
	double start_s, stop_s;
	if (false)
	{

		int counterMRows(256), counterMCols(256);
		int start_count(0), stop_count(10);
		printf("here1\n");
		Matrix M = Matrix::randi(counterMRows,counterMCols,start_count,stop_count);
		printf("here\n");
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
	
		free(pM);
	}

	/***********************************************\
	 * Part 2
	\***********************************************/

	if (true)
	{
		printf(" ------------- Part 2 ------------------\n");

		int filtMRows(128), filtMCols(128); // Matrix Size
		int filtNRows(13), filtNCols(13); // Filter Size
		int filt_start_count(0), filt_stop_count(5);
		int nRowBreak = 64;
		Matrix M2 = Matrix::randi( filtMRows, filtMCols, filt_start_count, filt_stop_count );

		//M2.write(stdout);
		
		int blockSize;
		int subMatNumRows;
		int subMatNumCols;

		M2.getParFiltBlockSize( nRowBreak, filtNRows, filtNCols, blockSize, subMatNumRows, subMatNumCols );
	

		double *pBlockM 		= (double *) malloc(blockSize * sizeof(double) );
		double *pBlockMResult 	= (double *) malloc(blockSize * sizeof(double) );
		double *pBlockMResult2 	= (double *) malloc(blockSize * sizeof(double) );

		M2.copy_to_c_zero_padded_blocks( pBlockM, nRowBreak, filtNRows, filtNCols );
		
		start_s = omp_get_wtime();
		gpu::rms_filter( pBlockMResult, pBlockM, nRowBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols, true );	
		stop_s = omp_get_wtime();
		printf("Global Compute Time: %1.20f\n", stop_s - start_s );

		//Matrix MResult1( pBlockMResult, nRowBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols );
		//MResult1.write(stdout);
		
		start_s = omp_get_wtime();
		gpu::rms_filter( pBlockMResult2, pBlockM, nRowBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols, false );	
		stop_s = omp_get_wtime();
		
		//Matrix MResult( pBlockMResult2, nRowBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols );
		//MResult.write(stdout);

		//printf("Number Block = %d\n", nRowBreak);
		//printf("Number Threads = %d\n", subMatNumRows * subMatNumCols);
		printf("Shared Compute Time: %1.20f\n", stop_s - start_s );
		
		start_s = omp_get_wtime();
		RMS_filter2( M2, filtNRows, filtNCols );
		stop_s = omp_get_wtime();
		printf("Serial Compute Time: %1.20f\n", stop_s - start_s );

		free(pBlockM);	
		free(pBlockMResult);
		free(pBlockMResult2);
	}

	return 0;

}
