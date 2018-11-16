#include <iostream>
#include <stdio>

#include "matrix.hpp"
#include "math_eval.hpp"
#include "algo.hpp"
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

	int dim = 3;
	g = &sum_of_the_squares;
	g = &rastrigin;
	int numParticles = 100;
	double pos_lower_bound = -10;
	double pos_upper_bound = 10;
	double a_1 = 0.1;
	double a_2 = 1.0;
	double max_vel = 0.2;
	int max_iter 	= 1000;

	particle_swarm_eval( 	g, 
							dim, 
							numParticles, 
							pos_lower_bound,
							pos_upper_bound,
							a_1, a_2,
							max_vel,
							max_iter, 
							false); 
	
	return 0;
}

