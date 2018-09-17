#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>

using namespace std;

class Matrix
{
    public:
    Matrix( void );
    Matrix( const int rows, const int cols );
    void write( FILE * os );
    private:
        vector<vector<double> > M;
        int rows;
        int cols;

		
};

#endif
