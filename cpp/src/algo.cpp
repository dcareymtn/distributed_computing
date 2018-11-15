#include <stdio.h>
#include <omp.h>
#include <vector>
#include <math.h>
#include <mpi.h>

#include "algo.hpp"
#include "matrix.hpp"

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
