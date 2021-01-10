#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ft-it
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc

# Load all necessary modules.
echo "Loading modules..."
module load 2019

echo "Loading virtual environment..."
source ${HOME}/projects/erp/venv/bin/activate

cd ${HOME}/projects/semeval2020 || exit

language=italian
preproc=token
model=dbmdz/bert-base-italian-xxl-uncased
epochs=5
batch=8

echo "Run script"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/run_mlm_wwm.py \
	--model_type bert \
	--model_name_or_path $model \
	--train_file finetuning_corpora/${language}/${preproc}/train.txt \
	--validation_file finetuning_corpora/${language}/${preproc}/test.txt \
	--targets_file finetuning_corpora/${language}/targets/target_forms.csv \
	--do_train \
	--do_eval \
	--output_dir finetuned_bert/${language}/ \
	--num_train_epochs ${epochs} \
	--per_device_train_batch_size ${batch} \
	--per_device_eval_batch_size ${batch} \
	--evaluation_strategy epoch \
	--load_best_model_at_end \
	--save_total_limit 1 \
	--fp16 \
	&> out/finetune_${language}_${preproc}_bsz${batch}

