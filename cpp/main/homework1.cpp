#include <stdio.h>
#include "test.h"
#include "matrix.h"

int main(int argc, char **argv)
{  
    
    Matrix A(3,3);
    A.write( stdout );
    
    return 0;

}
