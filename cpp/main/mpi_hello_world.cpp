#include<mpi.h>
#include<stdio.h>
#include<unistd.h>

int main(int argc, char **argv)
{  

  int rank, size;  
  char hostname[256];  
  
  MPI_Init(&argc,&argv);  
  MPI_Comm_size(MPI_COMM_WORLD, &size);  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
  gethostname(hostname,255);  
  
  printf("Hello world! I am process number: %d out of %d on host %s\n", rank, size, hostname);  
  
  MPI_Finalize();  
  
  return 0;

}
