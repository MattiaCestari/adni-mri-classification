#!/bin/bash
#SBATCH --job-name=SMIL_train_encoders
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=6:00:00
#SBATCH --output=./SMIL_train_encoders.out
#SBATCH --error=./SMIL_train_encoders.err
#SBATCH --account=pi-aereditato

# Load your conda environment
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate adni_rcc   

echo "Starting job at $(date)"

python -u train_encoders.py --config encoders.yaml --save-dir .

