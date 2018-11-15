#include <vector>
#include <iostream>

#include "util.hpp"

void printVectorTableCSV( FILE * f, std::vector<int> v, int start_int, int stop_int )
{
    for (int this_int = start_int; this_int <= stop_int; this_int++)
    {
        fprintf( f, "   %4d   |", this_int);
    }
    fprintf( f, "\n");
    
    int count_idx = 0;
    for (int this_int = start_int; this_int <= stop_int; this_int++)
    {
        fprintf( f, "   %4d   |", v[count_idx]);
        count_idx += 1;
    }
    fprintf(f, "\n\n\n");
}

void printVectorTableCSV( FILE * f, std::vector<int> row1, std::vector<double> row2 )
{
    for (int iCol = 0; iCol < row1.size(); iCol++)
    {
        fprintf( f, "   %6d   |", row1[iCol]);
    }
    fprintf( f, "\n");

    for (int iCol = 0; iCol < row2.size(); iCol++)
    {
        fprintf( f, "   %6.4f   |", row2[iCol]);
    }
    fprintf( f, "\n\n\n");

}

void printMatrix( FILE * f, double * pM, int nRows, int nCols )
{
	
	fprintf(f, "nRows = %d\n", nRows);
	fprintf(f, "nRows = %d\n", nCols);
	for (int iRow = 0; iRow < nRows; iRow++)
	{
		for (int iCol = 0; iCol < nCols; iCol++)
		{
			fprintf(f, "%1.2f ", *((pM + iRow*(nCols)) + iCol));
		}
		fprintf(f, "\n");
	}

}
