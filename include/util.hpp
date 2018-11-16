#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <iostream>

void printVectorTableCSV( FILE * f, std::vector<int>, int start, int stop );
void printVectorTableCSV( FILE * f, std::vector<int> row1, std::vector<double> row2);
void printMatrix( FILE * f, double * pM, int nRows, int nCols);
void printVector( FILE * f, int dim, double * vec);
void printMatrix( FILE * f, int rows, int cols, double * mat, bool bTranspose = false);
double unifrand( double min, double max);
#endif
