#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=p.conti@campus.lmu.de

#SBATCH --ntasks=120
#SBATCH --partition=lsm-rbg
#SBATCH --mem-per-cpu 2000

#SBATCH --time=2-00:00:00

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo $1

mpirun cosmosis --mpi $1
