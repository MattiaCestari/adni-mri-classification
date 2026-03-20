#!/bin/bash
#SBATCH --job-name=oasis_download
#SBATCH --output=./logs/oasis_download.out
#SBATCH --error=./logs/oasis_download.err
#SBATCH --time=18:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --account=pi-aereditato

set -euo pipefail

mkdir -p ./logs

CSV_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_multimodal/csv/oasis/dataset_multimodal.csv"
DATA_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_multimodal/data_oasis"

source ../../env_setup.sh
cd scripts/ || exit 1

START=600
END=1372

python download_subject_oasis.py "$START" "$END" "$CSV_PATH" "$DATA_PATH"
