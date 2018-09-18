#include <stdio.h>
#include "test.h"
#include "matrix.h"

void problem1()
{
    printf("Working Problem 1:\n");

    // Generate a matrix of size 128 x 128
    Matrix M = Matrix::randi(128, 128, 0, 31);

}

int main(int argc, char **argv)
{  

    problem1();
    
    Matrix A(3,3);
    A.write( stdout );
    Matrix::eye(3);
    Matrix::randi(5,5,11,20);
    return 0;

}
