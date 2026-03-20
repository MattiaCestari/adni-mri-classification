#!/bin/bash
#SBATCH --array=1-100%100
#SBATCH --job-name=preprocessing_multimodal_oasis
#SBATCH --output=./logs/preprocessing_multimodal_oasis_%A_%a.out
#SBATCH --error=./logs/preprocessing_multimodal_oasis_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH --partition=caslake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=pi-aereditato

set -euo pipefail

mkdir -p ./logs

CSV_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_multimodal/csv/oasis/dataset_multimodal.csv"
DATA_PATH="/project/aereditato/cestari/adni-mri-classification/data/preprocessing_multimodal/data_oasis"

source ../../env_setup.sh
cd scripts/ || exit 1

# First and last subject indices to be preprocessed (inclusive)
FIRST_SUBJ_IDX=600
LAST_SUBJ_IDX=1372

TOTAL=$((LAST_SUBJ_IDX - FIRST_SUBJ_IDX + 1)) # Number of unique subjects
NJOBS=100
CHUNK=$(( (TOTAL + NJOBS - 1) / NJOBS ))   # ceil(TOTAL/NJOBS)

TASK0=$((SLURM_ARRAY_TASK_ID - 1))
START=$((TASK0 * CHUNK + FIRST_SUBJ_IDX))
END=$((START + CHUNK - 1 ))

if [ "$END" -gt "$LAST_SUBJ_IDX" ]; then
    END=$LAST_SUBJ_IDX
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: processing subjects ${START}..${END}"

for IDX in $(seq $START $END); do
  python preprocess_subject_oasis.py "$IDX" "$CSV_PATH" "$DATA_PATH"
done
