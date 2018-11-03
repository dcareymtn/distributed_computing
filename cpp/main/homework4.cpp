#include "matrix.h"
#include "gpu_algo.h"

int main(int argc, char **argv)
{  
	int counterMRows(6), counterMCols(6);
	int start_count(0), stop_count(3);
	Matrix M = Matrix::randi(counterMRows,counterMCols,start_count,stop_count);

	M.write(stdout);

	double *pM = (double *) malloc(counterMRows*counterMCols*sizeof(double));
	M.copy_to_cptr( pM );	
		
	gpu::count_occurrences( pM, counterMRows, counterMCols, start_count, stop_count);
	
    free(pM);

	return 0;

}
