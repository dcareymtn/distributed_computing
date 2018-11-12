#include <mpi.h>

#include "matrix.h"
#include "matrix.h"

using namespace std;

Matrix::Matrix( void )
{
}


Matrix::Matrix( const int rows, const int cols ) :
    M(rows, vector<double> (cols, 0)),
    rows(rows), cols(cols)
{
}

Matrix::Matrix( double *pBlockM, int nRowBreak, int nColBreak, int subMatNumRows, int subMatNumCols, int nFiltRows, int nFiltCols )
{
	if ((nFiltRows % 2) != 1)
	{
		fprintf(stderr, "Error: Matrix::Matrix ... nFiltRows must be odd");
		exit(0);
	}

	if ((nFiltCols % 2) != 1)
	{
		fprintf(stderr, "Error: Matrix ... n FiltCols must be odd");
	}

	this->rows = nRowBreak * (subMatNumRows - nFiltRows + 1);
	this->cols = nColBreak * (subMatNumCols - nFiltRows + 1);

	this->M = std::vector< std::vector< double > > ( this->rows, std::vector< double > (this->cols, 0));

	int extraRowIdx 	= (nFiltRows - 1)/2;
	int extraColIdx 	= (nFiltCols - 1)/2;

	int blockSize 		= subMatNumRows * subMatNumCols;
	
	int currRow, currCol;
	
	for (int iRowBreak = 0; iRowBreak < nRowBreak; iRowBreak++)
	{
		for (int iColBreak = 0; iColBreak < nColBreak; iColBreak++)
		{
			for (int iSubRow = extraRowIdx; iSubRow < subMatNumRows - extraRowIdx; iSubRow++)
			{
				for (int iSubCol = extraColIdx; iSubCol < subMatNumCols - extraColIdx; iSubCol++)
				{
					currRow 	= iRowBreak*(subMatNumRows - nFiltRows + 1 ) + iSubRow - extraRowIdx;
					currCol 	= iColBreak*(subMatNumCols - nFiltCols + 1 ) + iSubCol - extraColIdx;
					this->M[currRow][currCol] 	= *(pBlockM + iRowBreak * blockSize * nColBreak + iColBreak*blockSize + iSubRow * subMatNumCols  + iSubCol );
				}
			}
		}
	}

}

int Matrix::getRows( void ) const
{
    return this->rows;
}

int Matrix::getCols( void ) const
{
    return this->cols;
}

void Matrix::getParFiltBlockSize(int nRowBreak, int nColBreak, int nRowFilt, int nColFilt, int &blockSize, int &subMatNumRows, int &subMatNumCols ) const
{

	if ((nRowFilt %2 != 1) || (nColFilt %2 != 1))
	{
		fprintf(stderr, "Filter Size must be odd");
	}

	if ((this->rows % 2) == 0)
	{
		subMatNumCols  	= this->cols/nColBreak + nColFilt - 1;
		subMatNumRows  	= this->rows/nRowBreak + nRowFilt -1;
		blockSize 		= subMatNumRows * subMatNumCols * nRowBreak * nColBreak;
	}
	else
	{
		fprintf(stderr, "Matrix Rows size most be 2^N");
		exit(0);
	}

}

void Matrix::write( FILE * os ) const
{
    fprintf(os, "nrows = %d\n", rows);
    fprintf(os, "ncols = %d\n", cols);

    for (int iRow = 0; iRow < rows; iRow++ )
    {
        for (int iCol = 0; iCol < cols; iCol++ )
        {
            fprintf(os, "%5.1f ", M[iRow][iCol]);
        }
        fprintf(os, "\n");
    }
    fprintf(os, "\n");
}



vector<double>& Matrix::operator[](int row)
{
    return M[row];
}
vector<double> Matrix::operator[](int row) const
{
    return M[row];
}

void Matrix::set_all_to_zeros( void )
{
    for (int iRow = 0; iRow < this->rows; iRow++)
    {
        for (int iCol = 0; iCol < this->cols; iCol++)
        {
            this->M[iRow][iCol] = 0;
        }
    }
}

Matrix Matrix::getSubMatrix( int start_row, int stop_row, int start_col, int stop_col ) const
{
    int nRow = stop_row - start_row + 1;
    int nCol = stop_col - start_col + 1;
    
    Matrix newMat( nRow, nCol );

    for (int iRow = 0; iRow < nRow; iRow++)
    {   
        for (int iCol = 0; iCol < nCol; iCol++)
        {
            newMat[iRow][iCol] = this->M[start_row + iRow][start_col + iCol];
        }
    }

    return newMat;
}

std::vector<std::vector<Matrix> > Matrix::parBreak( int nRowBreak ) const
{
    std::vector<std::vector<Matrix> > Mpar(nRowBreak, std::vector<Matrix> (1));

    if (this->rows%nRowBreak != 0)
    {
        fprintf(stderr, "%d is Not-divisible by %d\n", this->rows, nRowBreak);
        exit(0);
    }
    
    int newRowSize = this->rows / nRowBreak;

    for (int iPar = 0; iPar<nRowBreak; iPar++)
    {
        Mpar[iPar][0] = this->getSubMatrix( iPar*newRowSize, (iPar+1)*newRowSize-1, 0, this->cols);
    }

    return Mpar;    
    
}

std::vector<std::vector< Matrix> > Matrix::parBreakZeroPadForFilt( int nRowBreak, int nColBreak, int filtNRows, int filtNCols ) const
{
    std::vector<std::vector<Matrix> > Mpar( nRowBreak, std::vector<Matrix> (nColBreak) );
    if (this->rows%nRowBreak != 0)
    {
        fprintf( stderr, "%d is Not divisible by %d\n", this->rows, nRowBreak);
        exit(0);
    }
    
	int newRowSize  = this->rows / nRowBreak;
	int newColSize 	= this->cols / nColBreak;

    int newRowSizeOverlap = newRowSize + filtNRows - 1;
    int newColSizeOverlap = newColSize + filtNCols - 1;
    
	Matrix newOverlap = Matrix( newRowSizeOverlap, newColSizeOverlap );
    
	int start_row_idx, stop_row_idx;
	int start_col_idx, stop_col_idx;

    int this_row, this_col;

    for (int iRowBreak = 0; iRowBreak < nRowBreak; iRowBreak++)
    {
		for (int iColBreak = 0; iColBreak < nColBreak; iColBreak++)
		{
			start_row_idx  	= iRowBreak*newRowSize - (filtNRows - 1)/2;
			stop_row_idx   	= iRowBreak*newRowSize + newRowSize + (filtNRows - 1)/2;

   			start_col_idx   = iColBreak*newColSize - (filtNCols - 1)/2;
    		stop_col_idx    = iColBreak*newColSize + newColSize + (filtNCols - 1)/2;

			for (int iNewRow = start_row_idx; iNewRow < stop_row_idx; iNewRow++)
			{
				for (int iNewCol = start_col_idx; iNewCol < stop_col_idx; iNewCol++)
				{
					this_row = iNewRow + (filtNRows-1)/2 - iRowBreak*newRowSize;
					this_col = iNewCol + (filtNCols-1)/2 - iColBreak*newColSize;

					if (iNewRow < 0 || iNewRow >= this->rows || iNewCol < 0 || iNewCol >= this->cols)
					{
						newOverlap[this_row][this_col] = 0;
					}
					else
					{
						newOverlap[this_row][this_col] = this->M[iNewRow][iNewCol];
					}
				}
			}
	        
			Mpar[iRowBreak][iColBreak] = newOverlap;
		
		}

    }

    return Mpar;
}

void Matrix::sendMPI( const int proc, const int tag, const MPI_Comm comm) const
{
    for (int iRow = 0; iRow < rows; iRow++)
    {
        MPI_Send( &M[iRow][0], cols, MPI_DOUBLE, proc, tag, comm);
    }
}

void Matrix::recvMPI( const int num_rows, const int num_cols, const int from_proc, const int tag, const MPI_Comm comm, MPI_Status *status)
{

    rows = num_rows;
    cols = num_cols;

    M.resize( num_rows, std::vector<double>( num_cols, 0)); 

    for (int iRow = 0; iRow < rows; iRow++)
    {   
        MPI_Recv( &M[iRow][0], cols, MPI_DOUBLE, from_proc, tag, comm, status);
    }  
}

void Matrix::copy_to_cptr( double *newM )
{
	for (int iRow = 0; iRow < rows; iRow++)
	{
		for (int iCol = 0; iCol < cols; iCol++)
		{
			*(newM + iRow * cols + iCol) = M[iRow][iCol];
		}
	}
}

void Matrix::copy_to_c_zero_padded_blocks( double *newMArray, int nRowBreak, int nColBreak, int filtNRows, int filtNCols ) const
{
	std::vector<std::vector< Matrix > > MBlock = this->parBreakZeroPadForFilt( nRowBreak, nColBreak, filtNRows, filtNCols );

	int fRows = MBlock[0][0].getRows(); 
	int fCols = MBlock[0][0].getCols();
	int fSize = fRows * fCols;
	
	for (int iRowBreak = 0; iRowBreak < nRowBreak; iRowBreak++)
	{
		for (int iColBreak = 0; iColBreak < nColBreak; iColBreak++)
		{
			Matrix thisM = MBlock[iRowBreak][0];
			for (int iRow = 0; iRow < fRows; iRow++)
			{
				for (int iCol = 0; iCol < fCols; iCol++)
				{
					*(newMArray + iRowBreak * fSize * nColBreak + iColBreak * fSize + iRow*fCols + iCol) = MBlock[iRowBreak][iColBreak][iRow][iCol];
				}
			}
		}
	}

}

Matrix Matrix::zeros( int rows, int cols )
{
    return Matrix( rows, cols );
}

Matrix Matrix::eye( int n )
{
    Matrix new_mat = Matrix( n, n );

    for (int r = 0; r < n; r++)
    {
        new_mat[r][r] = 1.0;
    }
    
    return new_mat;
}

Matrix Matrix::randi( int rows, int cols, int low_int, int high_int )
{
    Matrix new_mat = Matrix( rows, cols );

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            new_mat[r][c]   = -1 +low_int + (rand() % (high_int - low_int + 1) + 1);
        }
    }

    return new_mat;
}

Matrix Matrix::stack( const Matrix & top, const Matrix & middle, const Matrix & bottom )
{
    Matrix stackMat = Matrix(   top.getRows() + middle.getRows() + bottom.getRows(), 
                                top.getCols());
    
    for (int iRow = 0; iRow < top.getRows(); iRow++)
    {
        for (int iCol = 0; iCol < top.getCols(); iCol++)
        {
            stackMat[iRow][iCol] = top[iRow][iCol];
        }
    }

    for (int iRow = 0; iRow < middle.getRows(); iRow++)
    {
        for (int iCol = 0; iCol < middle.getCols(); iCol++)
        {
            stackMat[iRow+top.getRows()][iCol] = middle[iRow][iCol];
        }
    }

    for (int iRow = 0; iRow < bottom.getRows(); iRow++)
    {
        for (int iCol = 0; iCol < bottom.getCols(); iCol++)
        {
            stackMat[iRow + top.getRows() + middle.getRows()][iCol] = bottom[iRow][iCol];
        }
    }

    return stackMat;
}
