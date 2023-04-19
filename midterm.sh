#!/bin/bash
#SBATCH --job-name="Midterm Project"
#SBATCH --time=00:60:00
#SBATCH --mem-per-cpu=2GB

module purge
module load Python

srun python3 ./DSF/midtermcode.py
