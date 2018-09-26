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
    std::vector<int> nPar = {1,1,2,4,8,16,32};
    int counter_size = 128;
    int image_size  = 128;
    int filter_size = 3;
    double start_s, stop_s;
    std::vector<double> counter_time, rms_filter_time;

    Matrix T = Matrix::randi(counter_size, counter_size, start_int, stop_int);
    Matrix I = Matrix::randi(image_size, image_size, start_int, stop_int);

    printf("Experiment 1) Counter:\n\n");

    start_s = omp_get_wtime();
    std::vector<int> Tcount     = count_occurrences( T, start_int, stop_int);
    stop_s = omp_get_wtime();
    counter_time.push_back(stop_s - start_s);
    for (int iPar = 1; iPar < nPar.size(); iPar++)
    {
        start_s = omp_get_wtime();
        std::vector<int> TcountPar  = count_occurrences_par( T, start_int, stop_int, nPar[iPar]);
        stop_s = omp_get_wtime();
        counter_time.push_back(stop_s - start_s);
    }

    printf("Counter Results\n\n");
    printVectorTableCSV( stdout, nPar, counter_time );

    printf("\n\nExperiment 2) RMS Filter\n\n");

    start_s     = omp_get_wtime();
    Matrix filter_series = RMS_filter2(I, filter_size, filter_size);
    stop_s  = omp_get_wtime();
    rms_filter_time.push_back(stop_s - start_s);

    for (int iPar = 1; iPar < nPar.size(); iPar++)
    {
        start_s = omp_get_wtime();
        Matrix filter_par = RMS_filter2_par( I, nPar[iPar], filter_size, filter_size);
        stop_s = omp_get_wtime();
        rms_filter_time.push_back(stop_s - start_s);
    }

    printf("RMS Filter Results\n\n");
    printVectorTableCSV( stdout, nPar, rms_filter_time );
    
}

int main(int argc, char **argv)
{  

    problem1();
    
    return 0;

}
