#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#include "curand.h"
#include "curand_kernel.h"

#include "gpu_algo.h"
#include "util.hpp"
#include "cmath_eval.h"

namespace gpu
{

__device__ double gpu_sum_of_the_squares( int dim, double * vec)
{
	double res = 0;
	
	for (int i = 0; i < dim; i++)
	{
		res += vec[i]*vec[i];
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

__global__ void max( const int num, double * _d_array_in, double *_d_array_out )
{
	// 		
}

__global__ void gpu_particle_swarm_opt( double (*f)(int dim, double * vec), 
										const int dim,
										curandState *state,  
										const double pos_lower_bound, 
										const double pos_upper_bound, 
										int iterations,
										double *_d_test )
{
	
	// Initialize some arrays
	__shared__ double *smem;

	unsigned int numParticles 	= blockDim.x;
	unsigned int c_pos_idx 		= 0;
	unsigned int c_vel_idx 		= c_pos_idx + dim * numParticles;
	unsigned int pb_pos_idx 	= c_vel_idx + dim * numParticles;
	unsigned int pb_score_idx 	= pb_pos_idx + dim*numParticles;
	unsigned int gb_pos_idx 	= pb_score_idx + numParticles;
	unsigned int gb_score_idx 	= gb_pos_idx + dim;
	double current_score;
	size_t smemsize = (gb_score_idx + 1) * sizeof(double);

	double r_1, r_2;

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	double dum[2];
	dum[0]=tid;
	dum[1]=tid;
	_d_test[tid]= gpu_sum_of_the_squares( dim, &dum[0]);
			
	// Allocate shared memory for us on all threads
	if (threadIdx.x == 0)
	{	
		// current position (dim x numParticles), current velocity (dim x numParticles), then personal best position (dim x numParticles)
		// plus personnal_best_score (numParticles) + global_best_pos (dim) + global_best_score (1)
		smem 	= (double*)malloc(smemsize);
	}
	
	__syncthreads();

	// Check for failure
	if (smem == NULL)
		return;
	
	// Make a pointer for each thread
	double *data = smem;
	
	// Initialize the swarm
	data[pb_score_idx + threadIdx.x] = INFINITY;
	
	for (int iDim = 0; iDim < dim; iDim++)
	{
		data[c_pos_idx + iDim*numParticles + threadIdx.x] = (curand_uniform_double( &state[tid] ) * (pos_upper_bound - pos_lower_bound)) + pos_lower_bound;
		data[c_vel_idx + iDim*numParticles + threadIdx.x] = 0; 
		data[pb_pos_idx + iDim*numParticles + threadIdx.x] = data[c_pos_idx + iDim*numParticles + threadIdx.x];
	}

	// Start the optimization
	for (int iIter = 0; iIter < iterations; iIter++)
	{
		// Get the score of the current particle
		current_score 	= gpu_sum_of_the_squares( dim, &data[c_pos_idx + threadIdx.x*dim]);
	
		// Was this a personal best?
		data[pb_pos_idx + threadIdx.x] = min( data[pb_pos_idx + threadIdx.x], current_score );

		__syncthreads();

	}

	// Evaluate this thread's score at the initial position

	// 

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


void particle_swarm_eval( double (*f)(int dim, double * vec), 
							int dim,
							int numSwarms,  
							int numParticlesPerSwarm,
							double pos_lower_bound,
							double pos_upper_bound, 
							double a_1, double a_2,
							double max_vel,
							int iterations, 
							bool bHighIsGood )
{

	// Create the random state for random number generation on each thread (global)
	curandState *_d_state;
	cudaMalloc((void **)&_d_state, numParticlesPerSwarm * numSwarms* sizeof(curandState));

	// Create memory for testing on host and device;
	double *_h_test =  (double *) malloc(numParticlesPerSwarm * numSwarms * sizeof(double));
	
	double *_d_test; 
	cudaMalloc((void **)&_d_test, numParticlesPerSwarm * numSwarms * sizeof(double));
		
	// Set up the kernel for number generation on each particle
	setup_kernel<<< numSwarms, numParticlesPerSwarm >>>(time(NULL), _d_state);
	
	// Call the kernel test
	printf("Num Swarms = %d; Num Particles = %d\n", numSwarms, numParticlesPerSwarm);
	gpu_particle_swarm_opt<<< numSwarms, numParticlesPerSwarm >>>(	f,
																	dim,
																	_d_state, 
																	pos_lower_bound, pos_upper_bound, 
																	iterations,
																	_d_test );
	
	// Copy the test
	cudaMemcpy( _h_test, _d_test, (numParticlesPerSwarm*numSwarms * sizeof(double)), cudaMemcpyDeviceToHost );
	
	for (int i = 0; i < numParticlesPerSwarm*numSwarms; i++)
	{
		printf("%f ", _h_test[i]);
	}
	printf("\n");
		
	cudaFree(_d_state);	
	free(_h_test);
		
	// OLD CODE STARTS HERE

	//srand(time(NULL));
	//
	//int score_fac = bHighIsGood ? -1 : 1;
	//
	//// Using the current position of the particles (from pos_vec_array), compute the score at each particle
	//// Using the current position of the particles (from pos_vec_array), Update the Personal best for each particle
	//double *c_pos 		= (double *)malloc( dim * numParticles * sizeof(double));
	//double *c_vel 		= (double *)malloc( dim * numParticles * sizeof(double));
	//double *pb_pos 		= (double *)malloc( dim * numParticles * sizeof(double));
	//double *pb_score 	= (double *)malloc( numParticles * sizeof(double));
	//double *gb_pos 		= (double *)malloc( dim * sizeof( double ));
	//double r_1;
	//double r_2;
	//double gb_score 	= INFINITY;
	//double this_score = 0;
	//int idx 	= 0;

	//FILE * fParticles;
	//FILE * fScore;
	//
	//fParticles 	= fopen( "particles.csv", "w");
	//fScore 		= fopen( "score.csv", "w");
	//
	//// Initialize scoreing
	//	for (int iParticle = 0; iParticle < numParticles; iParticle++)
	//	{
	//		pb_score[iParticle] 	= INFINITY;
	//		for (int iDim = 0; iDim < dim; iDim++)
	//		{
	//			c_pos[iParticle * dim + iDim] 	= unifrand( pos_lower_bound, pos_upper_bound ); 
	//			c_vel[iParticle * dim + iDim] 	= 0;
	//			pb_pos[iParticle * dim + iDim] 	= c_pos[iParticle * dim + iDim];
	//		}
	//	}

	//// Start the optimization
	//	for (int iter = 0; iter < max_iter; iter++)
	//	{
	//		printf("--------------------------------------------\n");
	//		printf("            Iteration %d   \n", iter);
	//		printf("--------------------------------------------\n");

	//		for (int iParticle = 0; iParticle < numParticles; iParticle++)
	//		{
	//			this_score = score_fac * feval_c( f, dim, &c_pos[iParticle * dim] );
	//			pb_score[iParticle] = min( pb_score[iParticle], this_score );
	//		}

	//		printf("Current Position\n");	
	//		printMatrix( stdout, numParticles, dim, c_pos, true );
	//		printMatrix( fParticles, numParticles, dim, c_pos, true );
	//		printf("Current Velocity\n"    );
	//		printMatrix( stdout, numParticles, dim, c_vel, true );
	//		printf("Personal Best Position\n"    );
	//		printMatrix( stdout, numParticles, dim, pb_pos, true );
	//		printf("Personal Best Score\n");
	//		printMatrix( stdout, 1, numParticles, pb_score );
	//		printMatrix( fParticles, 1, numParticles, pb_score );
	//		
	//		// Of all the particles, do a maximum reduction on global data to find the global max
	//		for (int iParticle = 0; iParticle < numParticles; iParticle++)
	//		{
	//			if (pb_score[iParticle] < gb_score)
	//			{
	//				gb_score 	= min( pb_score[iParticle], gb_score);
	//				for (int iDim = 0; iDim < dim; iDim++)
	//				{
	//					gb_pos[iDim] 	= c_pos[iParticle * dim + iDim];
	//				}
	//			}
	//		}

	//		fprintf( stdout, "global score = %f\n", gb_score);
	//		fprintf( fScore, "%f\n");
	//		printf("Global Best Position\n");
	//		printMatrix( stdout, dim, 1, gb_pos );	

	//		// Randomly generate the two random vectors [0,1]
	//		// Move the particles and update the positions	
	//		
	//		for (int iParticle = 0; iParticle < numParticles; iParticle++)
	//		{
	//			r_1 	= unifrand(0.0,1.0);
	//			r_2 	= unifrand(0.0,1.0);
	//
	//			for (int iDim = 0; iDim < dim; iDim++)
	//			{
	//				idx 	= iParticle * dim + iDim;
	//				
	//				c_vel[idx] 	= c_vel[idx] +  a_1 * r_1 * (pb_pos[idx] - c_pos[idx]) + a_2 * r_2 * (gb_pos[iDim] - c_pos[idx]);
	//				c_vel[idx] 	= c_vel[idx] * (fabs(c_vel[idx]) > max_vel ? max_vel/fabs(c_vel[idx]) : 1);
	//				c_pos[idx] 	= c_pos[idx] + c_vel[idx];
	//			}

	//		}

	//	}



	//// Compute the convergence metric

	//// If done, then exit

	//// Else, repeat up to max num times
	//
	//free(c_pos);
	//free(c_vel);
	//free(pb_pos);
	//free(pb_score);
	//free(gb_pos);

}// End GPU Name space

}
