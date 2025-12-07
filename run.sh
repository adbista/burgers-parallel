#!/bin/bash
#SBATCH -A plgar2025-cpu
#SBATCH -p plgrid
#SBATCH -t 00:20:00
#SBATCH -J burgers_roe
#SBATCH -o burgers.out
#SBATCH -e burgers.err
#SBATCH -N 1
#SBATCH --tasks-per-node=17

if [ $# -ne 2 ]; then
    echo "Użycie: sbatch run_burgers.sh <num_processes> <N>"
    exit 1
fi

NPROC=$1   # liczba procesów MPI
NGRID=$2   # liczba punktów siatki N

module load Python/3.10.8-GCCcore-12.2.0
module load mpi4py/3.1.4-gompi-2022b
module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1

# ważne: ogranicz liczbę wątków OpenMP (jeśli biblioteki używają)
export OMP_NUM_THREADS=1

echo "=== Roe Burgers 1D MPI ==="
echo "Procesy MPI      = ${NPROC}"
echo "Liczba punktów N = ${NGRID}"
echo "================================="

# uruchomienie programu (zmień ścieżkę jeśli plik jest w innym katalogu)
mpiexec -n ${NPROC} python parallel_roe.py ${NGRID} 0.5

echo "=== Job finished ==="
