#ifndef ALGO_H
#define ALGO_H

#include "matrix.hpp"

// Function to count the occurences of integers in a Matrix
std::vector<int> count_occurrences( const Matrix & mat , int start_count, int stop_count );

std::vector<int> count_occurrences_par( const Matrix & mat, int start_count, int stop_count, int nPar );

std::vector<int> count_occurrences_par_mpi( const Matrix & mat, int start_count, int stop_count, int nPar );

Matrix RMS_filter2( const Matrix & M, int filtNRows, int filtNCols );

Matrix RMS_filter2_par( const Matrix & M, int nPar, int filtNRows, int filtNCols );

Matrix RMS_filter2_par_mpi( const Matrix & M, int nPar, int filtNRows, int filtNCols );


void particle_swarm_eval( double (*f)(int dim, double * vec), 
							int dim, 
							int numParticles, 
							double pos_lower_bound,
							double pos_upper_bound,
							double a_1, double a_2,
							double max_vel, int max_iter, 
							double &score,
							double *x_hat, 
							bool bHighIsGood = false,
							bool bWriteResults = false);

#endif
