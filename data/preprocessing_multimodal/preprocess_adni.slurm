#!/bin/bash
#SBATCH --array=1-100%100
#SBATCH --job-name=preprocessing_multimodal
#SBATCH --output=./logs/preprocessing_multimodal_%A_%a.out
#SBATCH --error=./logs/preprocessing_multimodal_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=amd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --account=pi-aereditato

set -euo pipefail

mkdir -p ./logs

ZIP_PATH1="/project/aereditato/cestari/downloads/ADNI/Multimodal_collection.zip"
ZIP_PATH2="/project/aereditato/cestari/downloads/ADNI/Multimodal_collection_dataset.zip"
CSV_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_multimodal/csv/dataset_multimodal.csv"
DATA_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_multimodal/data"

source ../../env_setup.sh
cd scripts/ || exit 1

# Total work items you want to cover
TOTAL=3289 # Number of unique subjects, as calculated in the notebook
NJOBS=100
CHUNK=$(( (TOTAL + NJOBS - 1) / NJOBS ))   # ceil(TOTAL/NJOBS)

TASK0=$((SLURM_ARRAY_TASK_ID - 1))
START=$((TASK0 * CHUNK))
END=$((START + CHUNK - 1))
if [ $END -ge $((TOTAL - 1)) ]; then END=$((TOTAL - 1)); fi

echo "Task ${SLURM_ARRAY_TASK_ID}: processing subjects ${START}..${END}"

for IDX in $(seq $START $END); do
  python preprocess_subject.py "$IDX" "$CSV_PATH" "$DATA_PATH" "$ZIP_PATH1" "$ZIP_PATH2"
done
