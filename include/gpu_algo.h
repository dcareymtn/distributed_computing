#ifndef GPU_ALGO_H
#define GPU_ALGO_H

namespace gpu
{

void hello_cuda();

void cuda_init();

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count, bool bGlobal );

void rms_filter( double *hBlockMResult, double *_h_BlockM, int nRowBreak, int nColBreak, int subMatNumRows, int subMatNumCols, int filtNRows, int filtNCols, bool bGlobal ); 

void particle_swarm_eval( 	int dim,
							int numSwarms, 
							int numParticlesPerSwarm, 
							double pos_lower_bound,
							double pos_upper_bound,
							double a_1, double a_2,
							double max_vel, int max_iter,
							double &result_score,
							double *result_vec, 
							bool bHighIsGood = false); 

}

#endif
