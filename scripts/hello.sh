#!/bin/bash
#BSUB -n 2 # number of cores that we are requesting
#BSUB -o scripts/lsf.out # regular output file
#BSUB -e scripts/lsf.err # stdeerr output file
#BSUB -q "windfall" # out group -- don't change
#BSUB -J hello_world # job name
#BSUB -R gpu

rm ~/workspace/distributed_computing/scripts/lsf*
module load cuda
module load openmpi 
time ~/workspace/distributed_computing/build/bin/hello

###end of script
