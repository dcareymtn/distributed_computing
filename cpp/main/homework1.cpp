#include <stdio.h>
#include <omp.h>
#include <mpi.h>

#include "matrix.h"
#include "algo.h"
#include "util.h"

void problem1()
{
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int start_int(1), stop_int(31);
    int counter_size = 128;
    
    Matrix T = Matrix::randi(counter_size, counter_size, start_int, stop_int);

    if (rank == 0)
    {

        printf("Working Problem 1:\n");

        // Generate a matrix of size 128 x 128
        Matrix M = Matrix::randi(8, 8, 0, 5);
        std::vector<int> counter = count_occurrences( M, 0, 5);
        //std::vector<int> counter_mpi = count_occurrences_par_mpi( M, 0, 5, 2222
        M.write(stdout);
        printVectorTableCSV( stdout, counter, 0, 5 );
        // printVectorTableCSV( stdout, counter_mpi, 0, 5);

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

        std::vector<int> nPar = {1,1,2,4,8,16,32};

        int image_size  = 128;
        int filter_size = 3;
        double start_s, stop_s;
        std::vector<double> counter_time, rms_filter_time;


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

    int done = 1;
    if (rank == 0)
    {
        
        double start_s, stop_s;
        start_s = omp_get_wtime();

        std::vector<int> TcountParMPI = count_occurrences_par_mpi( T, start_int, stop_int, size);
        MPI_Send( &done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        MPI_Recv( &done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stop_s  = omp_get_wtime();
        printVectorTableCSV( stdout, TcountParMPI, start_int, stop_int);

        printf("MPI Took: %3.3f sec\n", stop_s - start_s);
    }
    else
    {
        std::vector<int> TcountParMPI = count_occurrences_par_mpi( T, start_int, stop_int, size);
        MPI_Send( &done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
// printVectorTableCSV( stdout, TcountParMPI, start_int, stop_int); 
        
}

int main(int argc, char **argv)
{  
    MPI_Init(&argc, &argv); 
    problem1();
    MPI_Finalize();
    
    return 0;

}
