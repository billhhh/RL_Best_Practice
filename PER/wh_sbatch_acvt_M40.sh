#!/bin/bash

# Configure the resources required
#SBATCH -p batch                                                # partition (this is the queue your job will be added to)
#SBATCH -M acvt                                                 # use M40
#SBATCH -n 1              	                                    # number of tasks (sequential job starts 1 task) (check this if your job unexpectedly uses 2 nodes)
#SBATCH -c 4              	                                    # number of cores (sequential job calls a multi-thread program that uses 8 cores)
#SBATCH --time=24:00:00                                         # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:1                                            # generic resource required (here requires 4 GPUs)
#SBATCH --mem=32GB                                              # specify memory required per node (here set to 32 GB)

# Configure notifications 
#SBATCH --mail-type=END                                         # Type of email notifications will be sent (here set to END, which means an email will be sent when the job is done)
#SBATCH --mail-type=FAIL                                        # Type of email notifications will be sent (here set to FAIL, which means an email will be sent when the job is fail to complete)
#SBATCH --mail-user=a1757791@adelaide.edu.au                    # Email to which notification will be sent

# record GPU utilisation
nvidia-smi -l > nv-smi_sa.log.${SLURM_JOB_ID} 2>&1 &

# Execute your script (due to sequential nature, please select proper compiler as your script corresponds to)
python dqn_SpaceInvaders.py
