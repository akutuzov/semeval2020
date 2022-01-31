#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=tune-en
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

source ~/.bash_profile

# Load all necessary modules.
echo "Loading modules..."
module load 2019

echo "Loading virtual environment..."
source ${HOME}/projects/erp/venv/bin/activate

export HF_DATASETS_CACHE="/project/dmg_data/lsc/.cache/"

language=english
model=xlm-roberta-base
epochs=3
batch=4

python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0  \
        $HOME/projects/semeval2020/code/xlmr/finetune_mlm.py \
        --train_file $HOME/projects/semeval2020/finetuning_corpora/$language/token/1000.txt \
        --targets_file $HOME/projects/semeval2020/finetuning_corpora/$language/targets/targets.txt \
        --output_dir /project/dmg_data/lsc/models/xlmr/finetuned/$language \
        --do_train \
	--do_eval \
        --save_steps 100000000 \
        --fp16 \
        --num_train_epochs $epochs \
        --per_device_train_batch_size $batch \
        --model_name_or_path $model \
	--overwrite_output_dir \
	--max_seq_len 256 \
        &> ${HOME}/projects/semeval2020/out/finetune-xlmr-en_$(date +"%FT%T")


