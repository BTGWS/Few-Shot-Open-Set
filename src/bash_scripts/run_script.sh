#!/bin/bash -l
 
#SBATCH --nodes=1 # Allocate *at least* 5 nodes to this job.
#SBATCH --nodelist=node08
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1 # Allocate *at most* 5 tasks for job steps in the job
#SBATCH --cpus-per-task=1 # Each task needs only one CPU
#SBATCH --mem=64G # This particular job won't need much memory
#SBATCH --time=7-00:01:00  # 7 day and 1 minute 
#SBATCH --mail-user=snag005@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="1shot_rel_net_post_emh_resent12"
#SBATCH -p gpu # You could pick other partitions for other jobs
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.
 
# Place any commands you want to run below
conda activate py37
CUDA_VISIBLE_DEVICES=1 python main.py --config miniimagenet.yml
