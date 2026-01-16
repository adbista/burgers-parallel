#!/bin/bash
#SBATCH --account=plgar2025-cpu
#SBATCH --partition=plgrid
#SBATCH --time=18:00:00
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks=32

if [ $# -ne 6 ]; then
  echo "Użycie: $0 <program_name>  <num_processes> <N_grid_points> <T> <dt> <simulation_width>"
  echo "  <program_name> - nazwa programu python do uruchomienia; musi byc w folderze lokalnym"
  echo "  <num_processes> - liczba równoległych procesów programu"
  echo "  <N_grid_points> - liczba punktów w siatce 1D"
  echo "  <T> - czas symulacji - od 0 aż do 'T'"
  echo "  <dt> - krok czasowy symulacji"
  echo "  <simulation_width> - szerokość wymiaru przestrzennego; od 0 do <simulation_width>"
  exit 1
fi

PROGRAM_NAME=$1     # nazwa programu python do uruchomienia
NPROC=$2            # liczba procesów MPI
SIMULATION_N=$3     # liczba punktów w wymiarze przestrzennym
SIMULATION_TIME=$4  # czas trwania symulacji
SIMULATION_DT=$5    # krok czasowy symulacji
SIMULATION_WIDTH=$6 # szerokośc wymiaru przestrzennego L

# dx jest oblicznane już w programie jako dx = SIMULATION_WIDTH / SIMULATION_N
# t_steps jest obliczane już w programie jako t_steps = SIMULATION_TIME / SIMULATION_DT
# UWAGA! Żeby symulacja była stabilna, trzeba zadbać, żeby dt było odpowiednio małe względem dx!
# Nie wiem, jaka dokładnie jest zależność, ale jak będzie źle, to symulacja wybuchnie

module load mpi4py/3.1.4-gompi-2022b
module load scipy-bundle/2021.10-intel-2021b

echo "=== Roe Burgers 1D MPI ==="
echo "Uruchamiany program    = ${PROGRAM_NAME}"
echo "Procesy MPI            = ${NPROC}"
echo "Liczba punktów N       = ${SIMULATION_N}"
echo "Czas symulacji         = ${SIMULATION_TIME}"
echo "Krok czasowy symulacji = ${SIMULATION_DT}"
echo "Szerokość symulowanego przedziału = ${SIMULATION_WIDTH}"
echo "================================="

echo "Running the simulation..."

mpiexec -n "${NPROC}" python "${PROGRAM_NAME}" "${SIMULATION_N}" "${SIMULATION_TIME}" "${SIMULATION_DT}" "${SIMULATION_WIDTH}"
