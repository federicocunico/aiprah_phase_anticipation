#!/bin/bash
#SBATCH --nodes=1 # nodes
#SBATCH --ntasks-per-node=4 # tasks per node
#SBATCH --cpus-per-task=8 # cores per task
#SBATCH --mem=494000MB # memory on RAM
#SBATCH --gres=tmpfs:200GB # memory on $TMPDIR
#SBATCH --time=1:00:00 # time limit (d-hh:mm:ss)
#SBATCH --account=<account_name> # account
#SBATCH --partition=<partition_name> # partition name
#SBATCH --qos=<qos_name> # quality of service
module load profile/deeplrn
source activate 
srun python datasets/cholec80.py
