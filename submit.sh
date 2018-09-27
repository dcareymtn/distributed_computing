#!/bin/bash

### Set the job name

#PBS -N hello_world

### Specify the PI group for this job
### List of PI groups available to each user can be found with "va" command
# You can leave this as it is

#PBS -W group_list=ece677f18
### Request email when job begins and ends - commented out in this case
#PBS -m bea
### Specify email address to use for notification - commented out in this case

#PBS -M dcareymtn@email.arizona.edu

### Set the queue for this job as windfall
### We will be using windfall (this is the priority level)
#PBS -q windfall

### Set the number of nodes, cores and memory that will be used for this job
### You will need to adjust the following line
#PBS -l select=1:ncpus=28:mem=168gb

### Specify "wallclock time" required for this job, hhh:mm:ss
### You will need to adjust the following line
#PBS -l walltime=0:0:1

### Specify total cpu time required for this job, hhh:mm:ss
### total cputime = walltime * ncpus
### You will need to adjust the following line
#PBS -l cput=000:00:32

### Load required modules/libraries if needed (openmpi example)
### Use "module avail" command to list all available modules

### Below is what we submit for jobs
### set directory for job execution
cd ~/workspace/distributed_computing
### run your executable program with begin and end date and time outputdate

./build/bin/homework1
date
