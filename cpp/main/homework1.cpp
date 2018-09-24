#include <stdio.h>
#include <ctime>

#include "matrix.h"
#include "algo.h"
#include "util.h"

void problem1()
{
    printf("Working Problem 1:\n");

    // Generate a matrix of size 128 x 128
    Matrix M = Matrix::randi(128, 128, 0, 31);
    std::vector<int> counter = count_occurrences( M, 0, 31);
    printVectorTableCSV( stdout, counter, 0, 31 );
   
    int count =0;
    for (int ii = 0; ii < counter.size(); ii++)
    {
        count += counter[ii];
    }
    FILE * pFile = fopen("result.txt", "w");

    printVectorTableCSV( pFile, counter, 0, 31);

    Matrix T = Matrix::randi(128,128,1,8);
    std::vector<std::vector< Matrix > > P = T.parBreak( 4 );

    printf("======================================\n");
    printf("Starting Small Experiment\n");
    printf("======================================\n");

    printf("\n\nExperiment 1) Counter:\n\n");

    printf("\nInput Matrix:\\n\n");
    T.write(stdout);

    int start_s = clock();
    std::vector<int> Tcount     = count_occurrences( T, 1, 8);
    int stop_s = clock();
    double time_count_serial = (stop_s - start_s)/double(CLOCKS_PER_SEC)*1000;
    
    start_s = clock();
    std::vector<int> TcountPar  = count_occurrences_par( T, 1, 8, 4 );
    stop_s = clock();
    double time_count_par   = (stop_s - start_s)/double(CLOCKS_PER_SEC)*1000;

    printf("Serial: %3.5f seconds\n\n", time_count_serial);
    printVectorTableCSV( stdout, Tcount, 1, 8);
    
    printf("Paralellel: %3.5f seconds\n\n", time_count_par);
    printVectorTableCSV( stdout, TcountPar, 1, 8);

    printf("\n\nExperiment 2) RMS Filter\n\n");

    printf("Input:\n");
    T.write(stdout);
    printf("Serial:\n");
    start_s     = clock();
    RMS_filter2(T, 3, 3).write(stdout);
    stop_s  = clock();
    double time_rms_serial = (stop_s - start_s)/double(CLOCKS_PER_SEC)*1000;

    printf("Parallel:\n");

    start_s = clock();
    RMS_filter2_par( T, 2, 3, 3).write(stdout);
    stop_s = clock();
    double time_rms_par = (stop_s - start_s)/double(CLOCKS_PER_SEC)*1000;


    printf("Counter Serial : %3.5f seconds\n", time_count_serial);
    printf("Counter Parall : %3.5f seconds\n\n", time_count_par);
    printf("RMS Series took: %3.5f seconds\n", time_rms_serial);
    printf("RMS Parall took: %3.5f seconds\n", time_rms_par);

}

int main(int argc, char **argv)
{  

    problem1();
    
    return 0;

}
