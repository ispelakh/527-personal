#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0-1:00:00  # max job runtime
#SBATCH --cpus-per-task=1  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=instruction  # partition(s)
#SBATCH --gres=gpu:6
#SBATCH --mem=1G  # max memory
#SBATCH -J "test527"  # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load ml-gpu/20230427 

# make sure we are in working directory
cd /work/instruction/coms-527-f23/ispelakh/527

# execute python code
ml-gpu python3 main.py
