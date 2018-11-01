#include "matrix.h"
#include "gpu_algo.h"

int main(int argc, char **argv)
{  
	int counterMRows(128), counterMCols(128);
	int start_count(0), stop_count(64);
	Matrix M = Matrix::randi(counterMRows,counterMCols,start_count,stop_count);
	double *pM = (double *) malloc(counterMRows*counterMCols*sizeof(double));
	M.copy_to_cptr( pM );	
		
	gpu::count_occurrences( pM, counterMRows, counterMCols, start_count, stop_count);
	
    free(pM);

	return 0;

}
