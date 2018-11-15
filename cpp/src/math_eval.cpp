#include <math.h>

#include "math_eval.hpp"
#include "matrix.hpp"

Matrix meshgrid( int dim,  double start_val, double step_val, double stop_val)
{
	int nStep =	((stop_val - start_val)/step_val) + 1;

	Matrix M(nStep, nStep);

	if (dim == 2)
	{
		for (int iRow = 0; iRow < nStep; iRow++)
		{
			for (int iCol = 0; iCol < nStep; iCol++)
			{
				M[iRow][iCol] = start_val + step_val * iCol;
			}
		}
	}
	else if (dim == 1)
	{
		for (int iRow = 0; iRow < nStep; iRow++)
		{
			for (int iCol = 0; iCol < nStep; iCol++)
			{
				M[iRow][iCol] = start_val + step_val * iRow;
			}
		}
	}
	else
	{
		fprintf(stderr, "Error:math_eval::meshgrid. Dim must be 1 or 2");
	}

	return M;

}

Matrix feval( double (*f)( double x, double y), const Matrix & X, const Matrix & Y)
{
	
	int nRows, nCols;

	// Handle errors
	if (X.getRows() == Y.getRows())
	{
		nRows 	= X.getRows();
	}
	else
	{
		fprintf(stderr, "Error:math_eval::feval matrix size does not match" );
	}

	if (X.getCols() == Y.getCols())
	{
		nCols 	= Y.getCols();
	}
	else
	{
		fprintf(stderr, "Error:math_Eval::feval matrix size does not match");
	}
	
	Matrix Z(nRows, nCols);

	for (int iRow = 0; iRow < X.getRows(); iRow++)
	{
		for (int iCol = 0; iCol < X.getCols(); iCol++)
		{
			Z[iRow][iCol] 	= f(X[iRow][iCol], Y[iRow][iCol]);
		}
	}

	return Z;
}

