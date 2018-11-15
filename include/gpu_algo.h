#ifndef GPU_ALGO_H
#define GPU_ALGO_H

namespace gpu
{

void hello_cuda();

void cuda_init();

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count, bool bGlobal );

void rms_filter( double *hBlockMResult, double *_h_BlockM, int nRowBreak, int nColBreak, int subMatNumRows, int subMatNumCols, int filtNRows, int filtNCols, bool bGlobal ); 

/* Function to optimize a 2D function using PSO algorithm*\
\*                                                       */
void particle_swarm_eval( double (*f)(int dim, double * vec), 
							int dim, 
							int numParticles, 
							double * pos_vec_array, 
							double * vel_vec_array, 
							double a_1, double a_2, 
							double * P_b, double * P_g, 
							double *next_pos_vec_array);

}

#endif
