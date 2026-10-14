#!/bin/bash
#SBATCH --job-name=DLC_Analysis_SMILE
#SBATCH --output=dlc_output_%A_%a.log     
#SBATCH --array=0-9
#SBATCH --partition=defq               
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                      
#SBATCH --time=100:00:00

# 1. Activate your DeepLabCut environment
# Slurm jobs are non-interactive, so we must explicitly source conda first
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dlc

# 1.5 Export library path to ensure TensorFlow finds the CUDA libraries
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/dlc/lib:$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=0

# 2. Run the cluster-optimized python script
python analyze_all_videos_cluster.py
