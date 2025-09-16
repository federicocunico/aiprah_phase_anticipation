#!/bin/bash
#SBATCH --job-name=cholec_extract        # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=12                     # total number of tasks across all nodes
#SBATCH --mem=60G                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2                    # number of gpus per node
#SBATCH --time=25:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send mail when job begins
#SBATCH --mail-type=end                 # send mail when job ends
#SBATCH --mail-type=fail                # send mail if job fails
#SBATCH --mail-user=federico.cunico@univr.it
#SBATCH --output=train_%x.out           
#SBATCH --account=IscrC_SWARAS     # project name
#SBATCH --partition=boost_usr_prod      # https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+LEONARDO+UserGuide#UG3.2:LEONARDOUserGuide-Productionenvironment
#SBATCH --qos=boost_qos_lprod

module load profile/deeplrn
source activate 
srun python datasets/cholec80.py
