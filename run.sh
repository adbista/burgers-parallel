#!/bin/bash
#SBATCH --account=plgar2025-cpu
#SBATCH --partition=plgrid
#SBATCH --time=18:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks=32

if [ $# -ne 5 ]; then
  echo "Użycie: $0 <program_name>  <num_processes> <N_grid_points> <T> <dt>"
  echo "  <program_name> - nazwa programu python do uruchomienia; musi byc w folderze lokalnym"
  echo "  <num_processes> - liczba równoległych procesów programu"
  echo "  <N_grid_points> - liczba punktów w siatce 1D"
  echo "  <T> - czas symulacji - od 0 aż do 'T'"
  echo "  <dt> - krok czasowy symulacji"
  exit 1
fi

PROGRAM_NAME=$1    # nazwa programu python do uruchomienia
NPROC=$2           # liczba procesów MPI
NGRID=$3           # liczba punktów siatki N
SIMULATION_TIME=$4 # czas trwania symulacji
SIMULATION_DT=$5   # krok czasowy symulacji

module load mpi4py/3.1.4-gompi-2022b
module load scipy-bundle/2021.10-intel-2021b
export SLURM_OVERLAP=1

echo "=== Roe Burgers 1D MPI ==="
echo "Uruchamiany program    = ${PROGRAM_NAME}"
echo "Procesy MPI            = ${NPROC}"
echo "Liczba punktów N       = ${NGRID}"
echo "Czas symulacji         = ${SIMULATION_TIME}"
echo "Krok czasowy symulacji = ${SIMULATION_DT}"
echo "================================="

mpiexec -n "${NPROC}" python "${PROGRAM_NAME}" "${NGRID}" "${SIMULATION_TIME}" "${SIMULATION_DT}"
