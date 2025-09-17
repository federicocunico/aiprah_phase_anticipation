#!/bin/bash
#SBATCH --job-name=cholec_extract        # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=12                     # total number of tasks across all nodes
#SBATCH --mem=60G                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4                    # number of gpus per node
#SBATCH --time=25:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send mail when job begins
#SBATCH --mail-type=end                 # send mail when job ends
#SBATCH --mail-type=fail                # send mail if job fails
#SBATCH --mail-user=federico.cunico@univr.it
#SBATCH --output=train_%x.out           
#SBATCH --account=IscrC_SWARAS     # project name
#SBATCH --partition=boost_usr_prod      # https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+LEONARDO+UserGuide#UG3.2:LEONARDOUserGuide-Productionenvironment
#SBATCH --qos=boost_qos_lprod

# load bashrc
source $HOME/.bashrc

module purge
module load anaconda3/2023.03
module load profile/deeplrn
module load autoload cudnn

# conda activate
conda activate aiprah


##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "NUM gpu per done:=" $NUM_GPU
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date


echo "###### starting ######"
# torchrun --nnodes 1 --nproc_per_node $NUM_GPU --standalone run.py $flag1
srun python datasets/cholec80.py

echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"
