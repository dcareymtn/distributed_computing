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

vector<double>& Matrix::operator[](int row)
{
    return M[row];
}
vector<double> Matrix::operator[](int row) const
{
    return M[row];
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
        Mpar[iPar][0] = this->getSubMatrix(iPar*newRowSize, (iPar+1)*newRowSize-1, 0, this->cols);
    }

    return Mpar;    
    
}
