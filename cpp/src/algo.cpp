#include <stdio.h>
#include <omp.h>
#include <vector>
#include <math.h>
#include <mpi.h>
#include <time.h>

#include "algo.hpp"
#include "matrix.hpp"
#include "cmath_eval.h"
#include "util.hpp"

std::vector<int> count_occurrences( const Matrix & mat, int start_count, int stop_count )
{
    std::vector<int> counter ((stop_count - start_count + 1), 0);
    
    int count_idx;

    for (int iRow = 0; iRow < mat.getRows(); iRow++)
    {
        for ( int iCol = 0; iCol < mat.getCols(); iCol++ )
        {
            count_idx = 0; 
            for (int this_int = start_count; this_int <= stop_count; this_int++)
            {
                if (mat[iRow][iCol] == this_int)
                {
                    counter[count_idx] = counter[count_idx] + 1;
                }
                count_idx += 1;
            }
        }
    }
    return counter;
}

std::vector<int> count_occurrences_par( const Matrix & mat, int start_count, int stop_count, int nPar )
{
int num_int = (stop_count - start_count + 1);
std::vector<int> count (num_int, 0);

std::vector<std::vector<int> > count_par(nPar, std::vector<int> (num_int, 0)); 

std::vector<std::vector<Matrix> > PMat = mat.parBreak( nPar );

    omp_set_num_threads(nPar);

    #pragma omp parallel for
    for (int iPar = 0; iPar < nPar; iPar++)
    {
        count_par[iPar] = count_occurrences( PMat[iPar][0], start_count, stop_count ); 
    }

    for (int iInt = 0; iInt < num_int; iInt++)
    {
        for (int iPar = 0; iPar < nPar; iPar++)
        {
            count[iInt] = count[iInt] + count_par[iPar][iInt];
        }
    }

    return count;

}


std::vector<int> count_occurrences_par_mpi( const Matrix & mat, int start_count, int stop_count, int nPar )
{
    int num_int = (stop_count - start_count + 1);
    std::vector<int> count (num_int, 0);

    std::vector<std::vector<int> > count_par(nPar, std::vector<int> (num_int, 0)); 

    std::vector<std::vector<Matrix> > PMat = mat.parBreak( nPar );

    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Matrix recvMatrix(mat.getRows(), mat.getCols());
    int testint;
    if (rank == 0)
    {   
        for (int iPar = 1; iPar < size; iPar++)
        {
            //mat.sendMPI(iPar, 0, MPI_COMM_WORLD);
            PMat[iPar][0].sendMPI(iPar, 0, MPI_COMM_WORLD);
        }

        // Process some of the data on the main thread
        count_par[0] = count_occurrences( PMat[0][0], start_count, stop_count ); 

        // Receive the processed data on the parallel threads
        for (int iPar = 1; iPar < size; iPar++)
        {
            MPI_Recv( &count_par[iPar][0], num_int, MPI_INT, iPar, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Combine the results
        for (int iInt = 0; iInt < num_int; iInt++)
        {
            for (int iPar = 0; iPar < nPar; iPar++)
            {
                count[iInt] = count[iInt] + count_par[iPar][iInt];
            }
        }

        return count;

    }
    else
    {
        // Receive the sub matrix on the parallel thread
        recvMatrix.recvMPI( PMat[rank][0].getRows(), PMat[0][0].getCols(), 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Count the sub matrix on this parallel thread
        std::vector<int> par_count = count_occurrences( recvMatrix, start_count, stop_count);

        MPI_Send( &par_count[0], par_count.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);

    }

    // return count;

}

Matrix RMS_filter2( const Matrix & M, int filtNRows, int filtNCols )
{

    Matrix filtM( M.getRows(), M.getCols() );
    
    int numRowsUD = (filtNRows - 1)/2;
    int numColsLR = (filtNCols - 1)/2;

    double rssq     = 0;

    for (int iRow = (filtNRows + 1)/2 - 1; iRow < (M.getRows() - (filtNRows + 1)/2) + 1; iRow++)
    {
        for (int iCol = (filtNCols + 1)/2 - 1; iCol < (M.getCols() - (filtNCols + 1)/2) + 1; iCol++)
        {
            rssq = 0;
            for (int iFiltRow = iRow - numRowsUD; iFiltRow <= iRow + numRowsUD; iFiltRow++)
            {
                for (int iFiltCol = iCol - numColsLR; iFiltCol <= iCol + numColsLR; iFiltCol++)
                {
                    rssq += M[iFiltRow][iFiltCol] * M[iFiltRow][iFiltCol];
                }
            }
            rssq = sqrt( rssq );
            filtM[iRow][iCol] = rssq;
        }
        
    }
    
    return filtM;
}

Matrix RMS_filter2_par( const Matrix & M, int nPar, int filtNRows, int filtNCols )
{
    Matrix filtM( M.getRows(), M.getCols() );

    std::vector<std::vector<Matrix> > PMat = M.parBreakZeroPadForFilt( nPar, 1, filtNRows, filtNCols );

    int sub_row_size = M.getRows()/nPar;
    int sub_col_size = M.getCols();

    int new_row, new_col;

    #pragma omp parallel for
    for (int iPar = 0; iPar < nPar; iPar++)
    {
        Matrix temp = RMS_filter2( PMat[iPar][0], filtNRows, filtNCols);
       
        for (int iRow = 0; iRow < sub_row_size; iRow++)
        {
            for (int iCol = 0; iCol < sub_col_size; iCol++)
            {
                new_row = iRow + (filtNRows-1)/2;
                new_col = iCol + (filtNCols-1)/2;
                filtM[iPar*sub_row_size + iRow][iCol] = temp[new_row][new_col];
            }
        }
    }
return filtM;
}

Matrix RMS_filter2_par_mpi( const Matrix & M, int nPar, int filtNRows, int filtNCols )
{
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        Matrix filtM( M.getRows(), M.getCols() );
        std::vector<std::vector<Matrix> > PMat = M.parBreakZeroPadForFilt( nPar, filtNRows, 1, filtNCols);
        
        int sub_row_size = M.getRows()/size;
        int sub_col_size = M.getCols();

        int new_row, new_col;

        int num_rows = PMat[0][0].getRows();
        int num_cols = PMat[0][0].getCols();
        
        // Send some processes to the helpers
        for (int iPar = 1; iPar < size; iPar++)
        {

            MPI_Send( &num_rows, 1, MPI_INT, iPar, 0, MPI_COMM_WORLD);
            MPI_Send( &num_cols, 1, MPI_INT, iPar, 0, MPI_COMM_WORLD);

            PMat[iPar][0].sendMPI(iPar, 0, MPI_COMM_WORLD);
        
        }

        std::vector<Matrix> temp_vecMat(nPar, PMat[0][0]);
        
        temp_vecMat[0] = RMS_filter2( PMat[0][0], filtNRows, filtNCols);

        for (int iPar = 1; iPar < size; iPar++)
        {
            temp_vecMat[iPar].recvMPI( num_rows, num_cols, iPar, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
   
        for (int iPar = 0; iPar < nPar; iPar++)
        {
            for (int iRow = 0; iRow < sub_row_size; iRow++)
            {
                for (int iCol = 0; iCol < sub_col_size; iCol++)
                {
                    new_row = iRow + (filtNRows-1)/2;
                    new_col = iCol + (filtNCols-1)/2;
                    filtM[iPar*sub_row_size + iRow][iCol] = temp_vecMat[iPar][new_row][new_col];
                }
            }
        }

        return filtM;
    }
    else
    {
        int num_rows, num_cols;
        Matrix recvMatrix;
        MPI_Recv( &num_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv( &num_cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        recvMatrix.recvMPI( num_rows, num_cols, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Matrix temp = RMS_filter2( recvMatrix, filtNRows, filtNCols);

        temp.sendMPI( 0, 0, MPI_COMM_WORLD );

    }

    return Matrix(2,2);
}


void particle_swarm_eval( double (*f)(int dim, double * vec), 
							int dim, 
							int numParticles,
							double pos_lower_bound,
							double pos_upper_bound, 
							double a_1, double a_2,
							double max_vel,
							int max_iter, 
							bool bHighIsGood )
{
	
	srand(time(NULL));

	// Initialize the particles and their scoring
	Matrix vel = Matrix::zeros( numParticles, dim);

	double score_best 	= INFINITY;

	int score_fac = bHighIsGood ? -1 : 1;
	
	// Using the current position of the particles (from pos_vec_array), compute the score at each particle
	// Using the current position of the particles (from pos_vec_array), Update the Personal best for each particle
	double *c_pos 		= (double *)malloc( dim * numParticles * sizeof(double));
	double *c_vel 		= (double *)malloc( dim * numParticles * sizeof(double));
	double *pb_pos 		= (double *)malloc( dim * numParticles * sizeof(double));
	double *pb_score 	= (double *)malloc( numParticles * sizeof(double));
	double *gb_pos 		= (double *)malloc( dim * sizeof( double ));
	double r_1;
	double r_2;
	double gb_score 	= INFINITY;
	double this_score = 0;
	int idx 	= 0;

	// Initialize scoreing
		for (int iParticle = 0; iParticle < numParticles; iParticle++)
		{
			pb_score[iParticle] 	= INFINITY;
			for (int iDim = 0; iDim < dim; iDim++)
			{
				c_pos[iParticle * dim + iDim] 	= unifrand( pos_lower_bound, pos_upper_bound ); 
				c_vel[iParticle * dim + iDim] 	= 0;
				pb_pos[iParticle * dim + iDim] 	= c_pos[iParticle * dim + iDim];
			}
		}

	// Start the optimization
		for (int iter = 0; iter < max_iter; iter++)
		{
			printf("--------------------------------------------\n");
			printf("            Iteration %d   \n", iter);
			printf("--------------------------------------------\n");

			for (int iParticle = 0; iParticle < numParticles; iParticle++)
			{
				this_score = score_fac * feval_c( f, dim, &c_pos[iParticle * dim] );
				pb_score[iParticle] = min( pb_score[iParticle], this_score );
			}

			printf("Current Position\n");	
			printMatrix( stdout, numParticles, dim, c_pos, true );
			printf("Current Velocity\n"    );
			printMatrix( stdout, numParticles, dim, c_vel, true );
			printf("Personal Best Position\n"    );
			printMatrix( stdout, numParticles, dim, pb_pos, true );
			printf("Personal Best Score\n");
			printMatrix( stdout, 1, numParticles, pb_score );
			
			// Of all the particles, do a maximum reduction on global data to find the global max
			for (int iParticle = 0; iParticle < numParticles; iParticle++)
			{
				if (pb_score[iParticle] < gb_score)
				{
					gb_score 	= min( pb_score[iParticle], gb_score);
					for (int iDim = 0; iDim < dim; iDim++)
					{
						gb_pos[iDim] 	= c_pos[iParticle * dim + iDim];
					}
				}
			}

			fprintf( stdout, "global score = %f\n", gb_score);
			printf("Global Best Position\n");
			printMatrix( stdout, dim, 1, gb_pos );	

			// Randomly generate the two random vectors [0,1]
			// Move the particles and update the positions	
			
			for (int iParticle = 0; iParticle < numParticles; iParticle++)
			{
				r_1 	= unifrand(0.0,1.0);
				r_2 	= unifrand(0.0,1.0);
	
				for (int iDim = 0; iDim < dim; iDim++)
				{
					idx 	= iParticle * dim + iDim;
					
					c_vel[idx] 	= c_vel[idx] +  a_1 * r_1 * (pb_pos[idx] - c_pos[idx]) + a_2 * r_2 * (gb_pos[iDim] - c_pos[idx]);
					c_vel[idx] 	= c_vel[idx] * (fabs(c_vel[idx]) > max_vel ? max_vel/fabs(c_vel[idx]) : 1);
					c_pos[idx] 	= c_pos[idx] + c_vel[idx];
				}

			}

		}



	// Compute the convergence metric

	// If done, then exit

	// Else, repeat up to max num times
	
	free(c_pos);
	free(c_vel);
	free(pb_pos);
	free(pb_score);
	free(gb_pos);

}
