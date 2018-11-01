#include "matrix.h"
#include "gpu_algo.h"

int main(int argc, char **argv)
{  

	Matrix M = Matrix::randi(12,12,0,5);
	double *pM = (double *) malloc(12*12*sizeof(double));
	M.copy_to_cptr( pM );	
		
	M.write(stdout);
	gpu::count_occurrences( pM, 12, 12, 1, 2);
	
    free(pM);

	return 0;

}
