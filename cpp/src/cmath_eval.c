
#include "cmath_eval.h"

#include <math.h>
#include <stdio.h>

void c_hello()
{
	printf("Hello C\n");
}

double feval_c( double (*f)(int dim, double * vec), int dim, double * vec)
{
	return f( dim, vec);
}

double root_mean_square( int dim, double * vec)
{
	double res = 0;

	for (int i = 0; i < dim; i++)
	{
		res += vec[i]*vec[i];
	}
}

double x2_plus_y2( double x, double y)
{
	return x*x+y*y;
}

double rastrigin( double x, double y)
{
	return (10 * 2 + x*x - 10 * cos( 2 * M_PI * x ) + y*y - 10 * cos(2 * M_PI * y));
}

double rastrigin( int dim, double *vec)
{
	double res = 10 * dim;

	for (int i = 0; i < dim; i++)
	{
		res += vec[i]*vec[i] - 10 * cos( 2 * M_PI * vec[i] );
	}
}
