#!/bin/bash
#BSUB -n 2 # number of cores that we are requesting
#BSUB -o lsf.out # regular output file
#BSUB -e lsf.err # stdeerr output file
#BSUB -q "windfall" # out group -- don't change
#BSUB -J hello_world # job name
#BSUB -R gpu

module load cuda
module load openmpi 
time ~/workspace/distributed_computing/buildcu/bin/testmain

###end of script
