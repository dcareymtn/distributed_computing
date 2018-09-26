#include <stdio.h>
#include <omp.h>
#include <vector>
#include <math.h>

#include "algo.h"
#include "matrix.h"


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

    std::vector<std::vector<Matrix> > PMat = M.parBreakZeroPadForFilt( nPar, filtNRows, filtNCols );

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
