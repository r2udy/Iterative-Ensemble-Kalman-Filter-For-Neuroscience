#!/bin/bash
#
#SBATCH --job-name=fenics
#SBATCH --error=job_error.txt
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

# Unset any Python-related pollution from HPC modules
unset PYTHONPATH
unset LD_LIBRARY_PATH

source ~/miniconda3/bin/activate neur_act
# module load math gmsh
# ml load devel py-h5py/2.7.1_py27
# python3 -u generate_refined_mesh_multiple_holes.py
python3 -u solver_multiple_holes.py Nonlinear_diffusion_multiple_holes 2. 2. 3 4 75.
