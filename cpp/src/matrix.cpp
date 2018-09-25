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

int Matrix::getRows( void ) const
{
    return this->rows;
}

int Matrix::getCols( void ) const
{
    return this->cols;
}

void Matrix::write( FILE * os )
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

std::vector<std::vector< Matrix> > Matrix::parBreakZeroPadForFilt( int nRowBreak, int filtNRows, int filtNCols ) const
{
    std::vector<std::vector<Matrix> > Mpar( nRowBreak, std::vector<Matrix> (1) );

    if (this->rows%nRowBreak != 0)
    {
        fprintf( stderr, "%d is Not divisible by %d\n", this->rows, nRowBreak);
        exit(0);
    }

    int newRowSize = this->rows / nRowBreak;
    int newColSize = this->cols + ((filtNRows -1) );

    int newRowSizeOverlap = newRowSize + filtNRows - 1;

    Matrix newOverlap = Matrix( newRowSizeOverlap, newColSize );
    int start_row_idx, stop_row_idx;

    int start_col_idx   = -(filtNCols - 1)/2;
    int stop_col_idx    = this->cols + (filtNCols - 1)/2;

    int this_row, this_col;

    for (int iPar = 0; iPar < nRowBreak; iPar++)
    {
        start_row_idx  = iPar*newRowSize - (filtNRows - 1)/2;
        stop_row_idx   = iPar*newRowSize + newRowSize + (filtNRows-1)/2;

        for (int iNewRow = start_row_idx; iNewRow < stop_row_idx; iNewRow++)
        {
            for (int iNewCol = start_col_idx; iNewCol < newColSize; iNewCol++)
            {
                this_row = iNewRow + (filtNRows-1)/2 - iPar*newRowSize;
                this_col = iNewCol + (filtNCols-1)/2;

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
        Mpar[iPar][0] = newOverlap;
    }

    return Mpar;
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