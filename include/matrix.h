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
    vector<double>& operator[](int row);
    vector<double> operator[](int row) const;

    void set_all_to_zeros( void );

    int getRows( void ) const;
    int getCols( void ) const;
    Matrix getSubMatrix( int start_row, int stop_row, int start_col, int stop_col) const;
    
    std::vector<std::vector<Matrix> > parBreak(int nRowBreak ) const;
    std::vector<std::vector<Matrix> > parBreakZeroPadForFilt( int nRowBreak, int filtNRows, int FiltNCols ) const;

    static Matrix zeros( const int rows, const int cols );
    static Matrix eye(int n);
    static Matrix randi( int rows, int cols, int low_int, int high_int );
    static Matrix stack( const Matrix & top, const Matrix & middle, const Matrix & bottom );

    private:
        vector<vector<double> > M;
        int rows;
        int cols;

		
};

#endif
