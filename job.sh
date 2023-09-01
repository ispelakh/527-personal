#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0-1:00:00  # max job runtime
#SBATCH --cpus-per-task=1  # number of processor cores
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=instruction  # partition(s)
#SBATCH --gres=gpu:1
#SBATCH --mem=1G  # max memory
#SBATCH -J "test527"  # job name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load ml-gpu/20230427 
ml-gpu python3 main.py