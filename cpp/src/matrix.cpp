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
    printf( "nrows = %d\n\n", rows);
    for (int iRow = 0; iRow++; iRow < rows)
    {
        for (int iCol = 0; iCol++; iCol < cols)
        {
            printf("%2.0f ", M[iRow][iCol]);
        }
        printf("\n");
    }

}
