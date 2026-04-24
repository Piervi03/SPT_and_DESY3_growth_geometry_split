#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=p.conti@campus.lmu.de

#SBATCH --ntasks=100
#SBATCH --partition=usm-cl-el9
#SBATCH --mem-per-cpu 2000 
#SBATCH --nodes=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo $1

mpirun cosmosis --mpi $1
