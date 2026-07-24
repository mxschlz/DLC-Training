#!/bin/bash
#SBATCH --job-name=DLC_Analysis_SMILE
#SBATCH --output=dlc_output_%j.log     
#SBATCH --partition=defq               
#SBATCH --gpus=1                       
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                      
#SBATCH --time=48:00:00                

# 1. Activate your DeepLabCut environment
# (You may need to uncomment and adjust the line below depending on how conda is loaded on the cluster)
# source activate dlc 

# 2. Run the cluster-optimized python script
python analyze_all_videos_cluster.py
