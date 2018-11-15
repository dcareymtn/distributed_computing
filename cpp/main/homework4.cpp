#include <omp.h>
#include <math.h>

#include "matrix.hpp"
#include "gpu_algo.h"
#include "algo.hpp"

int main(int argc, char **argv)
{ 
	double start_s, stop_s;

	start_s = omp_get_wtime();
	gpu::cuda_init();
	stop_s 	= omp_get_wtime();
	printf("CUDA Initialization Took: %f\n", stop_s - start_s);
	
	if (false)
	{

		int counterMRows(256), counterMCols(256);
		int start_count(0), stop_count(10);
		Matrix M = Matrix::randi(counterMRows,counterMCols,start_count,stop_count);
		double *pM = (double *) malloc(counterMRows*counterMCols*sizeof(double));
		M.copy_to_cptr( pM );	
		
		start_s = omp_get_wtime();
		gpu::count_occurrences( pM, counterMRows, counterMCols, start_count, stop_count, true);
		stop_s = omp_get_wtime();
		printf("Global Compute Time: %1.20f\n", stop_s - start_s );

		start_s = omp_get_wtime();
		gpu::count_occurrences( pM, counterMRows, counterMCols, start_count, stop_count, false);
		stop_s = omp_get_wtime();
		printf("Shared Compute Time: %1.20f\n", stop_s - start_s );
		
		start_s = omp_get_wtime();
		int temp_counter[stop_count - start_count + 1];
		for (int i = 0; i < 128*128; i++)
		{
			int temp_bin = pM[i] - start_count;
			temp_counter[temp_bin] += 1;
		}
		stop_s = omp_get_wtime();
		printf("Serial Compute Time: %1.20f\n", stop_s - start_s );
	
		free(pM);
	}

	/***********************************************\
	 * Part 2
	\***********************************************/

	if (true)
	{
		printf(" ------------- Part 2 ------------------\n");

		int filtMRows(256), filtMCols(256); // Matrix Size
		int filtNRows(3), filtNCols(3); // Filter Size
		int filt_start_count(0), filt_stop_count(5);
		int squareBreak = 16;
		int nRowBreak = squareBreak;
		int nColBreak = squareBreak;
		int nOMP_Par = 4;
		Matrix M2 = Matrix::randi( filtMRows, filtMCols, filt_start_count, filt_stop_count );

		//M2.write(stdout);
		
		int blockSize;
		int subMatNumRows;
		int subMatNumCols;

		M2.getParFiltBlockSize( nRowBreak, nColBreak, filtNRows, filtNCols, blockSize, subMatNumRows, subMatNumCols );	


		double *pBlockM 		= (double *) malloc(blockSize * sizeof(double) );
		double *pBlockMResult 	= (double *) malloc(blockSize * sizeof(double) );
		double *pBlockMResult2 	= (double *) malloc(blockSize * sizeof(double) );

		M2.copy_to_c_zero_padded_blocks( pBlockM, nRowBreak, nColBreak, filtNRows, filtNCols );

		start_s = omp_get_wtime();
		gpu::rms_filter( pBlockMResult, pBlockM, nRowBreak, nColBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols, true );	
		stop_s = omp_get_wtime();
		double global_proc_time	= stop_s - start_s;

		Matrix MResult1( pBlockMResult, nRowBreak, nColBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols );
		//MResult1.write(stdout);

		start_s = omp_get_wtime();
		gpu::rms_filter( pBlockMResult2, pBlockM, nRowBreak, nColBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols, false );	
		stop_s = omp_get_wtime();
		double shared_proc_time = stop_s - start_s;
		
		Matrix MResult( pBlockMResult2, nRowBreak, nColBreak, subMatNumRows, subMatNumCols, filtNRows, filtNCols );
		//MResult.write(stdout);

		start_s = omp_get_wtime();
		Matrix M3Result = RMS_filter2( M2, filtNRows, filtNCols );
		stop_s = omp_get_wtime();
		double serial_proc_time = stop_s - start_s;
		//M3Result.write(stdout);

		start_s = omp_get_wtime();
		Matrix M4Result = RMS_filter2_par( M2, nOMP_Par, filtNRows, filtNCols );
		stop_s = omp_get_wtime();
		double omp_proc_time = stop_s - start_s;

		printf("Number Block = %d\n", nRowBreak*nColBreak);
		printf("Number Threads = %d\n", subMatNumRows * subMatNumCols);
		
		printf("Global Compute Time: %1.20f\n", global_proc_time );	
		printf("Shared Compute Time: %1.20f\n", shared_proc_time );
		printf("Serial Compute Time: %1.20f\n", serial_proc_time );
		printf("OMP    Compute Time: %1.20f\n", omp_proc_time );

		free(pBlockM);	
		free(pBlockMResult);
		free(pBlockMResult2);
	}

	return 0;

}
