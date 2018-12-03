#include <omp.h>
#include <iostream>
#include <stdio>
#include <math.h>

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
	//g = &sum_of_the_squares;
	g = &rastrigin;
	int numParticles = 32;
	double pos_lower_bound = -opt_limit;
	double pos_upper_bound = opt_limit;
	double a_1 = 0.2;
	double a_2 = 1.0;
	double max_vel = 10000;
	int max_iter 	= 1000;
	int numSwarms  = 20;
	int numParticlesPerSwarm = numParticles;
	double score1, score2;
	double x_hat1[10], x_hat2[10];
	double start_s, stop_s;

	gpu::cuda_init();
	
//	start_s = omp_get_wtime();
//	gpu::particle_swarm_eval( 	dim, 
//								numSwarms, 
//								numParticlesPerSwarm,
//								pos_lower_bound,
//								pos_upper_bound,
//								a_1, a_2,
//								max_vel,
//								max_iter, 
//								score1,
//								&x_hat1[0],
//								false);
//
//	stop_s = omp_get_wtime();
//	printf("GPU Time: %1.10f\n", stop_s - start_s);
//
//	start_s = omp_get_wtime();
//	particle_swarm_eval( 	g, 
//							dim, 
//							numSwarms * numParticles, 
//							pos_lower_bound,
//							pos_upper_bound,
//							a_1, a_2,
//							max_vel,
//							max_iter, 
//							score1,
//							&x_hat1[0], 
//							false,
//							false);// Writing results 
//	stop_s = omp_get_wtime();
//	printf("Serial Time: %1.10f\n", stop_s - start_s);

	// Design experiment
	dim = 2;
	double gpu_time, cpu_time;
	int tSwarmSize;
	int startFac = 5;
	int numFactors = 5;
	int nMC = 100;
	double score1MC(0), score2MC(0);
	double time1MC(0), time2MC(0);
	double miss1MC(0), miss2MC(0);

	srand(1000*time(NULL));

	printf("%-20s%-20s%-20s%-20s%-20s%-20s%-20s%-20s\n", "Num Swarms", "Num Particles", "CPU Time", "GPU Time", "CPU Score", "GPU Score" , "CPU Miss", "GPU Miss");
	for (int iFac = startFac; iFac <= numFactors; iFac++)
	{
		
		// Experiment swarm size
		tSwarmSize 	= pow(2, iFac);

		// Reinitialize
		score1MC 	= 0;
		score2MC 	= 0;
		time1MC 	= 0;
		time2MC 	= 0;
		miss1MC 	= 0;
		miss2MC 	= 0;

		for (int iMC = 0; iMC < nMC; iMC++)
		{
			// Evaluate the cpu
			start_s = omp_get_wtime();
			particle_swarm_eval( 	g, 
									dim, 
									numParticlesPerSwarm * tSwarmSize, 
									pos_lower_bound,
									pos_upper_bound,
									a_1, a_2,
									max_vel,
									max_iter, 
									score1,
									&x_hat1[0], 
									false,
									false);// Writing results 
			stop_s = omp_get_wtime();
			cpu_time = stop_s - start_s;

			// Evaluate the gpu
			start_s = omp_get_wtime();
			gpu::particle_swarm_eval( 	dim, 
										tSwarmSize, 
										numParticlesPerSwarm,
										pos_lower_bound,
										pos_upper_bound,
										a_1, a_2,
										max_vel,
										max_iter, 
										score2,
										&x_hat2[0],
										true,
										false);

			stop_s = omp_get_wtime();
			gpu_time = stop_s - start_s;
			
			score1MC += score1;
			score2MC += score2;
			time1MC += cpu_time;
			time2MC += gpu_time;

			miss1MC += sqrt(x_hat1[0]*x_hat1[0] + x_hat1[1]*x_hat1[1]);
			miss2MC += sqrt(x_hat2[0]*x_hat2[0] + x_hat2[1]*x_hat2[1]);

//			printf("score1 = %-20.10fscore2 = %-20.10f\n", score1, score2);

		}

		printf("%-20d%-20d%-20f%-20f%-20.10f%-20.10f%-20.10f%-20.10f\n", tSwarmSize, numParticlesPerSwarm,time1MC/nMC, time2MC/nMC, score1MC/nMC, score2MC/nMC, miss1MC/nMC, miss2MC/nMC );
		
	}
	
	return 0;
}

