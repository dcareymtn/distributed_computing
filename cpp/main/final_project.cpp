#include <omp.h>
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

	int dim = 3;
	g = &sum_of_the_squares;
	//g = &rastrigin;
	int numParticles = 128;
	double pos_lower_bound = -opt_limit;
	double pos_upper_bound = opt_limit;
	double a_1 = 0.2;
	double a_2 = 1.0;
	double max_vel = 10000000000;
	int max_iter 	= 1000;
	int numSwarms  = 400;
	int numParticlesPerSwarm = numParticles;
	double score1, score2;
	double x_hat1[10], x_hat2[10];
	double start_s, stop_s;

	gpu::cuda_init();
	
	start_s = omp_get_wtime();
	gpu::particle_swarm_eval( 	dim, 
								numSwarms, 
								numParticlesPerSwarm,
								pos_lower_bound,
								pos_upper_bound,
								a_1, a_2,
								max_vel,
								max_iter, 
								score1,
								&x_hat1[0],
								false);

	stop_s = omp_get_wtime();
	printf("GPU Time: %1.10f\n", stop_s - start_s);

	start_s = omp_get_wtime();
	particle_swarm_eval( 	g, 
							dim, 
							numSwarms * numParticles, 
							pos_lower_bound,
							pos_upper_bound,
							a_1, a_2,
							max_vel,
							max_iter, 
							score1,
							&x_hat1[0], 
							false); 
	stop_s = omp_get_wtime();
	printf("Serial Time: %1.10f\n", stop_s - start_s);
	
	return 0;
}

