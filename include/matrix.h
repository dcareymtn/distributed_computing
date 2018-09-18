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
    static Matrix eye(int n);
    static Matrix randi( int rows, int cols, int low_int, int high_int );
    vector<double>& operator[](int row);    
    private:
        vector<vector<double> > M;
        int rows;
        int cols;

		
};

#endif
