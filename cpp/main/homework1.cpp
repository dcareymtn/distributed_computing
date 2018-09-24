#include <stdio.h>
#include <omp.h>

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

    printf("======================================\n");
    printf("Starting Small Experiment\n");
    printf("======================================\n");

    int start_int(1), stop_int(31);
    int nPar = 32;
    int counter_size = 256;
    int filter_size = 3;
    double start_s, stop_s;
    Matrix T = Matrix::randi(counter_size, counter_size, start_int, stop_int);


    printf("\nParallel: %d\n", nPar);

    printf("\n\nExperiment 1) Counter:\n\n");

    printf("\nInput Matrix:\\n\n");
    T.write(stdout);

    start_s = omp_get_wtime();
    std::vector<int> Tcount     = count_occurrences( T, start_int, stop_int);
    stop_s = omp_get_wtime();
    double time_count_serial = (stop_s - start_s);
    
    start_s = omp_get_wtime();
    std::vector<int> TcountPar  = count_occurrences_par( T, start_int, stop_int, nPar );
    stop_s = omp_get_wtime();
    double time_count_par   = (stop_s - start_s);

    printf("Serial: %3.5f seconds\n\n", time_count_serial);
    printVectorTableCSV( stdout, Tcount, 1, 8);
    
    printf("Paralellel: %3.5f seconds\n\n", time_count_par);
    printVectorTableCSV( stdout, TcountPar, 1, 8);

    printf("\n\nExperiment 2) RMS Filter\n\n");

    printf("Input:\n");
    T.write(stdout);
    printf("Serial:\n");
    start_s     = omp_get_wtime();
    Matrix filter_series = RMS_filter2(T, filter_size, filter_size);
    stop_s  = omp_get_wtime();
    filter_series.write(stdout);
    double time_rms_serial = (stop_s - start_s);

    printf("Parallel:\n");

    start_s = omp_get_wtime();
    Matrix filter_par = RMS_filter2_par( T, nPar, filter_size, filter_size);
    stop_s = omp_get_wtime();
    filter_par.write(stdout);
    double time_rms_par = (stop_s - start_s);


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
