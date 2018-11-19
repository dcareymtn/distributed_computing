#include <iostream>
#include <stdio>

#include "matrix.hpp"
#include "math_eval.hpp"
#include "algo.hpp"

#include "cmath_eval.h"
#include "gpu_algo.h"

int main()
{
	double opt_limit = 5;
	FILE *pFileX;
	FILE *pFileY;
	FILE *pFileZ;
	
	pFileX 	= fopen( "X.csv", "w");
	pFileY 	= fopen( "Y.csv", "w");
	pFileZ 	= fopen( "Z.csv", "w");

	Matrix X = meshgrid( 2, -opt_limit,0.1,opt_limit);
	Matrix Y = meshgrid( 1, -opt_limit,0.1,opt_limit);

	double (*f)(double, double);

	// f = &x2_plus_y2;
	f = &rastrigin;

	Matrix Z = feval(f, X, Y);
	
	X.write(pFileX);
	Y.write(pFileY); 
	Z.write(pFileZ);

	double (*g)(int, double*);

	int dim = 2;
	g = &sum_of_the_squares;
	//g = &rastrigin;
	int numParticles = 50;
	double pos_lower_bound = -opt_limit;
	double pos_upper_bound = opt_limit;
	double a_1 = 0.2;
	double a_2 = 1.0;
	double max_vel = 0.2;
	int max_iter 	= 50;
	int numSwarms  = 1;
	int numParticlesPerSwarm = numParticles;

	gpu::particle_swarm_eval( 	g,  
								dim, 
								numSwarms, 
								numParticlesPerSwarm,
								pos_lower_bound,
								pos_upper_bound,
								a_1, a_2,
								max_vel,
								max_iter, 
								false); 

	//particle_swarm_eval( 	g, 
	//						dim, 
	//						numParticles, 
	//						pos_lower_bound,
	//						pos_upper_bound,
	//						a_1, a_2,
	//						max_vel,
	//						max_iter, 
	//						false); 
	
	return 0;
}

