#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#include "curand.h"
#include "curand_kernel.h"

#include "gpu_algo.h"
#include "util.hpp"
#include "cmath_eval.h"

#define MAX_DIM 2
#define MAX_NUM_PARTICLES 1000

namespace gpu
{

__device__ double gpu_sum_of_the_squares( int dim, double * vec)
{
	double res = 0;
	
	for (int i = 0; i < dim; i++)
	{
		res += (vec[i]-0.5)*(vec[i]-0.5);
	}
	
	return res;
}

__device__ double gpu_rastrigin( int dim, double * vec)
{
	double res = 10 * dim;

	for (int i = 0; i < dim; i++)
	{
		res += vec[i]*vec[i] - 10 * cos( 2 * M_PI * vec[i] );
	}

	return res;

}

__global__ void setup_kernel( unsigned long seed, curandState *state )
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, tid, 0, &state[tid]);
}

__global__ void print_cuda(char *a, int N)
{
    char p[11]="Hello CUDA";
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
	if(idx < N) 
    {
        a[idx]=p[idx];
    }
}

__global__ void gpu_count_occurrences_global( double *dM, int size, int start_index, int stop_index, int *d_counter )
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid > size) return;

	int bin = dM[tid] - start_index;

	atomicAdd( &d_counter[bin], 1);

}

__global__ void gpu_count_occurrences_shared( double * pM, int size, int start_index, int stop_index, int *d_counter )
{

	extern __shared__ int s_counter[];

	int tid 	= blockIdx.x * blockDim.x + threadIdx.x;
	int N = stop_index - start_index + 1;

	if (tid < (stop_index - start_index + 1))
	{	
		s_counter[tid] = 0;
		d_counter[tid] = 0;
	}

	__syncthreads();
	
	if (tid < size)
	{
		int sbin = pM[tid] - start_index;
		atomicAdd( &s_counter[sbin], 1);
	}

	__syncthreads();

	if ((threadIdx.x < N) && (tid < size))
	{
		atomicAdd( &d_counter[threadIdx.x], s_counter[threadIdx.x] );
	}

}

__global__ void gpu_rms_filter_shared( double * _d_M, int filtNRows, int filtNCols, double * _d_MResult )
{
	
	extern __shared__ int _l_M[];
	
	int subMatIdx = blockIdx.x;
	
	int blockSize	 = blockDim.x * blockDim.y;

	int row = threadIdx.x;
	int col = threadIdx.y;

	_l_M[subMatIdx * blockSize + row * blockDim.y + col] = *(_d_M + subMatIdx * blockSize + row * blockDim.y + col );

	__syncthreads();

	double this_result(0);

	int start_row 	= max(row - (filtNRows - 1)/2, 0);
	int stop_row 	= min(row + (filtNRows - 1)/2, blockDim.x);
	
	int start_col 	= max(col - (filtNCols - 1)/2, 0);
	int stop_col 	= min(col + (filtNCols - 1)/2, blockDim.y);

	double temp(0);

	for (int iRow = start_row; iRow <= stop_row; iRow++)
	{
		for (int iCol = start_col; iCol <= stop_col; iCol++)
		{

			temp 		= _l_M[ subMatIdx * blockSize + iRow * blockDim.y + iCol ];
			this_result += temp*temp;

		}	
	}
	
	*(_d_MResult + subMatIdx * blockSize + row * blockDim.y + col ) = sqrt( this_result );
}

__global__ void gpu_rms_filter_global( double * _d_M, int filtNRows, int filtNCols, double * _d_MResult )
{
	int subMatIdx = blockIdx.x;
	
	int blockSize	 = blockDim.x * blockDim.y;

	int row = threadIdx.x;
	int col = threadIdx.y;

	double this_result(0);

	int start_row 	= max(row - (filtNRows - 1)/2, 0);
	int stop_row 	= min(row + (filtNRows - 1)/2, blockDim.x);
	
	int start_col 	= max(col - (filtNCols - 1)/2, 0);
	int stop_col 	= min(col + (filtNCols - 1)/2, blockDim.y);

	double temp(0);

	for (int iRow = start_row; iRow <= stop_row; iRow++)
	{
		for (int iCol = start_col; iCol <= stop_col; iCol++)
		{
			temp 		= *(_d_M + subMatIdx * blockSize + iRow * blockDim.y + iCol );
			this_result +=  temp*temp;
		}	
	}
	
	*(_d_MResult + subMatIdx * blockSize + row * blockDim.y + col ) =  sqrt( this_result );
}

// Test of a reduction
__device__ void max_int( int dim, int * _s_map )
{
	unsigned int tid = threadIdx.x;
	int left_value;
	int right_value;

	// Use the reduction tricks from CUDA reduction
	for (unsigned int s = dim / 2; s > 0; s>>=1)
	{
		if (tid < s)
		{
			// Get the values using the map
			left_value 	= _s_map[tid];
			right_value	= _s_map[tid + s];

			// If the score on the right is better than the left
			if (right_value > left_value)
			{

				// The left one is better, so switch them
				_s_map[tid] 	= right_value;
				_s_map[tid + s] = left_value;

			}
		}

		__syncthreads();

	}
}

// Device Shared Memory min but only an ordering of the particles personal best to get the max
__device__ void reduce_swarm( int numParticles, double * _s_array,  int * _s_map, int pb_score_idx )
{
	// Get the thread id (can only use this for shared memory
	unsigned int pId = threadIdx.x;
	unsigned int left_idx, right_idx;
	double left_value, right_value;

	// Use the reduction tricks from CUDA reduction
	for (int s = numParticles / 2; s > 0; s>>=1)
	{
		if (pId < s)
		{
			// Get the values using the map
			left_idx 	= _s_map[pId];
			right_idx	= _s_map[pId + s];
			left_value 	= _s_array[pb_score_idx + left_idx ];
			right_value	= _s_array[pb_score_idx + right_idx];

			// If the score on the right is better than the left
			if ( right_value < left_value)
			{

				// The left one is better, so switch them
				_s_map[pId] 	= right_idx;
				_s_map[pId + s] = left_idx;

			}
		}

		__syncthreads();

	}


}

__global__ void gpu_particle_swarm_opt( const int dim,
										curandState *state,  
										const double pos_lower_bound, 
										const double pos_upper_bound, 
										int iterations,
										double a_1, double a_2,
										double max_vel,
										double *_d_results,
										double *_d_test )
{
	
	// Initialize some arrays
	__shared__ int *smem_int;
	__shared__ double *smem;

	unsigned int numParticles 	= blockDim.x;
	unsigned int c_pos_idx 		= 0;
	unsigned int c_vel_idx 		= c_pos_idx + dim * numParticles;
	unsigned int pb_pos_idx 	= c_vel_idx + dim * numParticles;
	unsigned int pb_score_idx 	= pb_pos_idx + dim*numParticles;
	unsigned int gb_pos_idx 	= pb_score_idx + numParticles;
	unsigned int gb_score_idx 	= gb_pos_idx + dim;
	
	double current_score;
	double r_1, r_2;
	size_t smemsize = (gb_score_idx + 1) * sizeof(double);
	size_t smemintsize = numParticles * sizeof(int);

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int idx; // tempory idx
	
	unsigned int thread_map;
	// Allocate shared memory for use on all threads
	if (threadIdx.x == 0)
	{	
		// current position (dim x numParticles), current velocity (dim x numParticles), then personal best position (dim x numParticles)
		// plus personnal_best_score (numParticles) + global_best_pos (dim) + global_best_score (1)
		smem 	= (double*)malloc(smemsize);
		smem[gb_score_idx] = INFINITY;
		smem_int = (int*)malloc(smemintsize);
	}
	
	__syncthreads(); // sync to ensure shared memory
	smem_int[threadIdx.x]= threadIdx.x; // smem_int[threadIdx.x];
	 
	// Check for failure
	if (smem == NULL)
		return;
	
	// Make a pointer for each thread
	double *data = smem;
	int *hash = smem_int;
	
	double *pCurPos 	= smem + c_pos_idx;
	double *pCurVel 	= smem + c_vel_idx;
	double *pPbPos 		= smem + pb_pos_idx;
	double *pPbScore 	= smem + pb_score_idx;
	double *pGbPos		= smem + gb_pos_idx;
	double *pGbScore 	= smem + gb_score_idx;
	
	// Initialize the swarm
	data[pb_score_idx + threadIdx.x] = INFINITY;
	
	for (int iDim = 0; iDim < dim; iDim++)
	{
		thread_map 	= smem_int[threadIdx.x];
		pCurPos[ thread_map * dim + iDim] = (curand_uniform_double( &state[tid] ) * (pos_upper_bound - pos_lower_bound)) + pos_lower_bound;
		pCurVel[ thread_map * dim + iDim] = 0;
		pPbPos[ thread_map * dim + iDim ] = pPbPos[ thread_map * dim + iDim];	
	}
	
	// Start the optimization
	for (int iIter = 0; iIter < iterations; iIter++)
	{

		// Get the thread map
		thread_map 		= hash[threadIdx.x];
		
		// Get the score of the current particle and was it a personal best
//		current_score 	= gpu_sum_of_the_squares( dim, &pCurPos[thread_map*dim]);
		current_score 	= gpu_rastrigin( dim, &pCurPos[thread_map*dim] );
		
		if (current_score < pPbScore[thread_map])
		{
			pPbScore[thread_map] = current_score;

			for (int iDim = 0; iDim < dim; iDim++)
			{
				pPbPos[ thread_map * dim + iDim] = pCurPos[ thread_map * dim + iDim];
			}
		}
		//The IF STATEMENT ABOVE IS SLOW	
			
		// Sync the threads in order to confirm that all of the scores are complete
		__syncthreads();

		// Reduce the swarm
		reduce_swarm( numParticles, smem, smem_int, pb_score_idx ); 
	
		// The best personal best is the best
		if (threadIdx.x == 0)
		{
			pGbScore[0] = pPbScore[smem_int[0]];
			for (int iDim = 0; iDim < dim; iDim++)
			{
				pGbPos[iDim] = pPbPos[smem_int[0]*dim + iDim];
			}
		}
		
		__syncthreads();

		// Apply the stochastic motion
		r_1 	= curand_uniform_double( &state[tid]);
		r_2 	= curand_uniform_double( &state[tid]);
		
		// Apply the random motion to the terms	
		for (int iDim = 0; iDim < dim; iDim++)
		{

			idx 	= smem_int[threadIdx.x] * dim + iDim;
			
			pCurVel[idx] = pCurVel[idx] + a_1 * r_1 * (pPbPos[idx] - pCurPos[idx]) + a_2 * r_2 * (pGbPos[iDim] - pCurPos[idx]);
			pCurVel[idx] = pCurVel[idx] * (fabs(pCurVel[idx]) > max_vel ? max_vel/fabs(pCurVel[idx]) : 1);
			pCurPos[idx] = pCurPos[idx] + pCurVel[idx];
		}
		
		__syncthreads();

	}

	// Write the result
	if (threadIdx.x == 0)
	{
		_d_results[blockIdx.x*(dim+1)] = pGbScore[0];
		for (int iDim = 0; iDim < dim; iDim++)
		{
			_d_results[(blockIdx.x*(dim+1)) + iDim + 1] = pGbPos[iDim];
		}
	}

	// Testing 
	_d_test[threadIdx.x] 	= smem_int[threadIdx.x];
	_d_test[threadIdx.x + numParticles] = data[pb_score_idx + threadIdx.x];

}

//////////////////////////////////////////////////////////////////////////////////
// CUDA Main code below

void cuda_init()
{
	double *dM;
	cudaMalloc( (void **) &dM, 10*sizeof(double));
	cudaFree(dM);
}

void count_occurrences( double *h_M, int nRows, int nCols, int start_count, int stop_count, bool bGlobal )
{
	
	// Copy the matrix data to the gpu
	double *d_M;
	int *d_counter;
		
	int nbins 	= stop_count - start_count + 1;

	size_t counter_size = nbins * sizeof( int );
	int *h_counter = (int *) malloc( counter_size );

	for (int ii = 0; ii < nbins; ii++)
	{
		h_counter[ii] = 0;
	}

	int N = nRows * nCols;

	size_t size = N * sizeof( double );

	cudaMalloc((void **)&d_M, size);
	cudaMalloc((void **)&d_counter, counter_size);

	cudaMemcpy( d_M, h_M, size, cudaMemcpyHostToDevice);
	
	int factor = 4;

	int blockSize = (factor * nCols < 1000 ? factor * nCols : 1000);
	int nBlock = N / blockSize + (N%blockSize == 0 ? 0 : 1);

	if (bGlobal)
	{
		gpu_count_occurrences_global<<< nBlock, blockSize  >>>(d_M, size, start_count, stop_count, d_counter );
	}
	else
	{
		gpu_count_occurrences_shared<<< nBlock, blockSize, nbins*sizeof(int)  >>>(d_M, size, start_count, stop_count, d_counter );
	}


	cudaMemcpy( h_counter, d_counter, counter_size, cudaMemcpyDeviceToHost);

	//for (int ii = start_count; ii <= stop_count; ii++)
	//{
	//	printf("%4d | ", ii);
	//}
	//printf("\n");
	//for (int ii = 0; ii < nbins; ii++)
	//{
	//	printf("%3d |  ", h_counter[ii]);
	//}
	//printf("\n");

	free(h_counter);
	cudaFree(d_M);
	cudaFree(d_counter);
}	

void rms_filter( double *hBlockMResult, double *_h_BlockM, int nRowBreak, int nColBreak, int subMatNumRows, int subMatNumCols, int nFiltRows, int nFiltCols, bool bGlobal )
{
	double *_d_BlockM, *_d_BlockMResult, *_d_BlockMResultShared;

	size_t __blockSize = nRowBreak * nColBreak * subMatNumRows * subMatNumCols * sizeof(double);

	cudaMalloc((void **)&_d_BlockM, __blockSize );
	cudaMalloc((void **)&_d_BlockMResult, __blockSize );
	cudaMalloc((void **)&_d_BlockMResultShared, __blockSize );

	cudaMemcpy( _d_BlockM, _h_BlockM, __blockSize, cudaMemcpyHostToDevice );

	dim3 threadsPerBlock(subMatNumRows, subMatNumCols );

	if (bGlobal)
	{
		gpu_rms_filter_global<<< nRowBreak * nColBreak, threadsPerBlock >>>( _d_BlockM, nFiltRows, nFiltCols, _d_BlockMResult );
	}
	else
	{
		gpu_rms_filter_shared<<< nRowBreak * nColBreak, threadsPerBlock, __blockSize  >>>( _d_BlockM, nFiltRows, nFiltCols, _d_BlockMResult );	
	}

	cudaMemcpy( hBlockMResult, _d_BlockMResult, __blockSize, cudaMemcpyDeviceToHost );
	
	cudaFree(_d_BlockM);
	cudaFree(_d_BlockMResult);
	cudaFree(_d_BlockMResultShared);	
}

void hello_cuda(void)
{
    char *a_h, *a_d; // _h for the host and _d for the device based pointers
    const int N = 11;
    size_t size = N * sizeof(char);

    a_h = (char *) malloc(size); // allocating the array on the host
    cudaMalloc((void **) &a_d, size); // allocating the array on the device
    
    // initialize the host array
    for (int i = 0; i < N; i++)
    {
        a_h[i] = 0;
    }

    // Copy the array on the host to the device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

    int blocksize = 4;
    int nblock = N/blocksize + (N%blocksize == 0 ? 0 : 1);
	nblock = 100;

    print_cuda <<< nblock, blocksize >>>(a_d, N); // Run the kernel on the device

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
    // copy from the device memory to the host memory
    cudaMemcpy(a_h, a_d, sizeof(char) * N, cudaMemcpyDeviceToHost);


	// print the array on the host
    for (int i = 0; i < N; i++)
    {
        printf("%c", a_h[i]);
    }
	printf("\n");
    free(a_h);
	cudaFree(a_d);
}


void particle_swarm_eval( 	int dim,
							int numSwarms,  
							int numParticlesPerSwarm,
							double pos_lower_bound,
							double pos_upper_bound, 
							double a_1, double a_2,
							double max_vel,
							int iterations,
							double &result_score,
							double *result_vec, 
							bool bHighIsGood,
							bool bWriteResults )
{

	// Create the random state for random number generation on each thread (global)
	curandState *_d_state;
	cudaMalloc((void **)&_d_state, numParticlesPerSwarm * numSwarms* sizeof(curandState));

	// Create memory for testing on host and device;
	double *_h_test =  (double *) malloc(numParticlesPerSwarm * numSwarms*2 * sizeof(double));

	// Create memory for the result
	double *_h_result = (double *) malloc( numSwarms*(dim + 1) * sizeof(double));

	double * _d_result;
	cudaMalloc((void **)&_d_result, numSwarms*(dim+1) * sizeof(double));
	
	double *_d_test; 
	cudaMalloc((void **)&_d_test, numParticlesPerSwarm * 2 * sizeof(double));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

			
	// Set up the kernel for number generation on each particle
	setup_kernel<<< numSwarms, numParticlesPerSwarm >>>((unsigned long long)clock(), _d_state);

	// Call the kernel test
	cudaEventRecord(start);
	gpu_particle_swarm_opt<<< numSwarms, numParticlesPerSwarm >>>(	dim,
																	_d_state, 
																	pos_lower_bound, pos_upper_bound, 
																	iterations,
																	a_1, a_2,
																	max_vel,
																	_d_result,
																	_d_test );
	
	cudaEventRecord(stop);
	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess)
//		printf("Error: %s\n", cudaGetErrorString(err));

	// Copy the result
	cudaMemcpy( _h_result, _d_result, numSwarms*(dim + 1) * sizeof(double), cudaMemcpyDeviceToHost );
	
	// Copy the test
	cudaMemcpy( _h_test, _d_test, (numParticlesPerSwarm*numSwarms*2 * sizeof(double)), cudaMemcpyDeviceToHost );

	cudaEventSynchronize(stop);
	float msec = 0;
	cudaEventElapsedTime(&msec, start, stop);
	
	int best_idx;
	double this_result;
	result_score = INFINITY;

	for (int iSwarm = 0; iSwarm < numSwarms; iSwarm++)
	{
		this_result = _h_result[iSwarm*(dim+1)];
		
		if (this_result < result_score)
		{
			result_score = this_result;
			best_idx = iSwarm;
		}

		if (bWriteResults) 
		{
			printf("Result Score: %1.10f\n", this_result);
			for (int i = 0; i<dim; i++)
			{
				printf("%1.10f ", _h_result[iSwarm*(dim+1)+i+1]);
			}
			printf("\n\n");
		}

	}

	for (int i = 0; i < dim; i++)
	{
		result_vec[i] = _h_result[best_idx*(dim+1)+i+1];
	}
	
	if (bWriteResults)	
	{
		printf("The Best Swarm Idx is %d\n\n", best_idx);
		printf("The Best Swarm Score is %1.10f\n", result_score);
		printf("Best Pos = ("); 

		for (int i = 0; i < (dim-1); i++)
		{
			printf("%1.10f, ", result_vec[i]);
		}
		printf("%1.10f)\n\n", result_vec[dim-1]);
	}
			
	cudaFree(_d_state);
	cudaFree(_d_test);
	cudaFree(_d_result);	
	free(_h_result);
	free(_h_test);
		
}// End GPU Name space

}
