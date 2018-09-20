#ifndef ALGO_H
#define ALGO_H

#include "matrix.h"

// Function to count the occurences of integers in a Matrix
std::vector<int> count_occurrences( const Matrix & mat , int start_count, int stop_count );

std::vector<int> count_occurrences_par( const Matrix & mat, int start_count, int stop_count, int nPar );

Matrix RMS_filter2( const Matrix & M, int filtNRows, int filtNCols );

#endif
