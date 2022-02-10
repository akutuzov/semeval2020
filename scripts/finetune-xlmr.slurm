#!/bin/bash

#SBATCH --job-name=XLMR
#SBATCH --account=ec30
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=4         # To specify how many GPUs to use
#SBATCH --time=60:00:00      # Max walltime is 150 hours.
#SBATCH --mem-per-cpu=8G

# Definining resource we want to allocate.
#SBATCH --nodes=1
#SBATCH --ntasks=4

# 6 CPU cores per task to keep the parallel data feeding going. A little overkill, but CPU time is very cheap compared to GPU time.
#SBATCH --cpus-per-task=6

# This is used to make checkpoints and logs to readable and writable by other members in the project.
umask 0007

module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module purge   # Recommended for reproducibility
module load nlpl-datasets/1.17-foss-2019b-Python-3.7.4
module load nlpl-transformers/4.14.1-foss-2019b-Python-3.7.4

module list

# For example, english
language=${1}
model=xlm-roberta-base
epochs=2
bsz=16
max_seq_len=256

python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0  \
        ../code/xlmr/finetune_mlm.py \
        --train_file finetuning_corpora/$language/token/all.txt.gz \
        --targets_file targets/$language/target_forms_udpipe.csv \
        --output_dir finetuned_models/$language \
        --do_train \
        --do_eval \
        --save_steps 100000000 \
        --fp16 \
        --num_train_epochs $epochs \
        --per_device_train_batch_size $bsz \
        --model_name_or_path $model \
        --overwrite_output_dir \
        --max_seq_len $max_seq_len
