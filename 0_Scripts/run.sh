#!/bin/bash
#SBATCH --job-name=morph1
#SBATCH --output=morph1.out
#SBATCH --error=morph1.err
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:2

echo "LOADING THE ENVIRONMENT"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate subword
echo "Starting"

# Your job commands go here
python tokenizer_TAR_Wordpiece.py > Log_TAR.txt
python tokenizer_SHP_Morfessor_Flatcat.py > Log_SHP.txt
echo "DONE!!"
