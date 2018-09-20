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
    vector<double> operator[](int row) const;
    int getRows( void ) const;
    int getCols( void ) const;
    Matrix getSubMatrix( int start_row, int stop_row, int start_col, int stop_col) const;
    std::vector<std::vector<Matrix> > parBreak(int nRowBreak ) const;
    private:
        vector<vector<double> > M;
        int rows;
        int cols;

		
};

#endif
