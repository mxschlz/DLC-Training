#!/bin/bash
#SBATCH --job-name=DLC_Analysis_SMILE
#SBATCH --output=dlc_output_%j.log     
#SBATCH --partition=defq               
#SBATCH --gpus=1                       
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                      
#SBATCH --time=48:00:00                

# 1. Activate your DeepLabCut environment
# Slurm jobs are non-interactive, so we must explicitly source conda first
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dlc

# 2. Run the cluster-optimized python script
python storage/personal/Projects/DLC-Training/analyze_all_videos_cluster.py
