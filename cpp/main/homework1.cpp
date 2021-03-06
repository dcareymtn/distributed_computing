#include <stdio.h>
#include <omp.h>
#include <mpi.h>

#include "matrix.hpp"
#include "algo.hpp"
#include "util.hpp"

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
        
        FILE * pFile = fopen("result.txt", "w");
        printVectorTableCSV( pFile, TcountParMPI, start_int, stop_int);

        printf("MPI Running with %d cores\n", size);
        printf("MPI Occurrence Counting Took: %3.3f sec\n", stop_s - start_s);
    }
    else
    {
        std::vector<int> TcountParMPI = count_occurrences_par_mpi( T, start_int, stop_int, size);
        MPI_Send( &done, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    double start_s = omp_get_wtime();
    Matrix filtM_MPI = RMS_filter2_par_mpi( T, size, 3, 3 );

    if (rank == 0)
    {
        double stop_s = omp_get_wtime();
        printf("MPI RMS Filtering Took: %3.3f sec\n", stop_s - start_s);
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
