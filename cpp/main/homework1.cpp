#include <stdio.h>
#include "matrix.h"
#include "algo.h"
#include "util.h"

void problem1()
{
    printf("Working Problem 1:\n");

    // Generate a matrix of size 128 x 128
    Matrix M = Matrix::randi(128, 128, 0, 31);
    std::vector<int> counter = count_occurrences( M, 0, 31);
    printVectorTableCSV( stdout, counter, 0, 31 );
   
    int count =0;
    for (int ii = 0; ii < counter.size(); ii++)
    {
        count += counter[ii];
    }
    FILE * pFile = fopen("result.txt", "w");

    printVectorTableCSV( pFile, counter, 0, 31);

    Matrix T = Matrix::randi(8,8,1,8);
    std::vector<std::vector< Matrix > > P = T.parBreak( 4 );

    T.write(stdout);
    P[0][0].write(stdout);
    P[1][0].write(stdout);
    P[2][0].write(stdout);
    P[3][0].write(stdout);

    std::vector<int> Tcount     = count_occurrences( T, 1, 8);
    std::vector<int> TcountPar  = count_occurrences_par( T, 1, 8, 4 );

    printVectorTableCSV( stdout, Tcount, 1, 8);
    printVectorTableCSV( stdout, TcountPar, 1, 8);

    RMS_filter2( M, 3, 3 ).write(stdout);
    
}

int main(int argc, char **argv)
{  

    problem1();
    
    Matrix A(3,3);
    Matrix::eye(3);
    Matrix::randi(5,5,11,20);
    return 0;

}
