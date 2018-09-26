#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <iostream>

void printVectorTableCSV( FILE * f, std::vector<int>, int start, int stop );
void printVectorTableCSV( FILE * f, std::vector<int> row1, std::vector<double> row2);

#endif
