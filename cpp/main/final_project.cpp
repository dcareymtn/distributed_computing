#include <iostream>
#include <stdio>

#include "matrix.hpp"
#include "math_eval.hpp"

#include "cmath_eval.h"

int main()
{
	FILE *pFileX;
	FILE *pFileY;
	FILE *pFileZ;
	
	pFileX 	= fopen( "X.csv", "w");
	pFileY 	= fopen( "Y.csv", "w");
	pFileZ 	= fopen( "Z.csv", "w");

	Matrix X = meshgrid( 2, -5.12,0.01,5.12);
	Matrix Y = meshgrid( 1, -5.12,0.01,5.12);

	double (*f)(double, double);

	// f = &x2_plus_y2;
	f = &rastrigin;

	Matrix Z = feval(f, X, Y);
	
//	X.write(pFileX);
//	Y.write(pFileY); 
//	Z.write(pFileZ);

	double (*g)(int, double*);

	double vec[4];
	vec[0] = 1;
	vec[1] = 2;
	vec[2] = 3;
	vec[3] = 4;

	g = &root_mean_square;
	double res = g(4, vec);
	c_hello();

	printf("Result = %f\n", res);
	return 0;
}

