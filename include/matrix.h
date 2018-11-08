#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <mpi.h>

using namespace std;

class Matrix

{
    public:

    Matrix( void );
    Matrix( const int rows, const int cols );
	Matrix( double *pBlockM, int nRowBreak, int subMatNumRows, int subMatNumCols, int nFiltRows, int nFiltCols );
    void write( FILE * os ) const;
    vector<double>& operator[](int row);
    vector<double> operator[](int row) const;

	// Set methods

    void set_all_to_zeros( void );

	// Get methods

    int getRows( void ) const;
    int getCols( void ) const;
	void getParFiltBlockSize(int nRowBreak, int nRowFilt, int nColFilt, int &blockSize, int &subMatNumRows, int & subMatNumCols ) const;
	Matrix getSubMatrix( int start_row, int stop_row, int start_col, int stop_col) const;
    
	// Parallel support function

    std::vector<std::vector<Matrix> > parBreak(int nRowBreak ) const;
    std::vector<std::vector<Matrix> > parBreakZeroPadForFilt( int nRowBreak, int filtNRows, int FiltNCols ) const;
    void sendMPI( const int proc, const int tag, const MPI_Comm comm) const;
    void recvMPI( const int num_rows, const int num_cols, const int from_proc, const int tag, const MPI_Comm comm, MPI_Status *status);

	// Interfaces
	void copy_to_cptr( double * newM );
	void copy_to_c_zero_padded_blocks( double *newMArray, int nRowBreak, int filtNRows, int filtNCols ) const;

	// Static construction methods

    static Matrix zeros( const int rows, const int cols );
    static Matrix eye(int n);
    static Matrix randi( int rows, int cols, int low_int, int high_int );

	// Static operations
    static Matrix stack( const Matrix & top, const Matrix & middle, const Matrix & bottom );

    private:
        vector<vector<double> > M;
        int rows;
        int cols;

		
};

#endif
