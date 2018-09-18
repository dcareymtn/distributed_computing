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

void Matrix::write( FILE * os )
{
    printf( "nrows = %d\n", rows);
    printf( "ncols = %d\n", cols);

    for (int iRow = 0; iRow < rows; iRow++ )
    {
        for (int iCol = 0; iCol < cols; iCol++ )
        {
            printf("%2.0f ", M[iRow][iCol]);
        }
        printf("\n");
    }

}

Matrix Matrix::eye( int n )
{
    Matrix new_mat = Matrix( n, n );

    for (int r = 0; r < n; r++)
    {
        new_mat[r][r] = 1.0;
    }
    
    new_mat.write( stdout );

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

    new_mat.write( stdout );

    return new_mat;
}

vector<double>& Matrix::operator[](int row)
{
    return M[row];
}
