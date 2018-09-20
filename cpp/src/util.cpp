#include "util.h"
#include <vector>
#include <iostream>

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
