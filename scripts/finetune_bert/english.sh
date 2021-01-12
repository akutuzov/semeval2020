#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ft-en
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/.bashrc

# Load all necessary modules.
echo "Loading modules..."
module load 2019

echo "Loading virtual environment..."
source ${HOME}/projects/erp/venv/bin/activate

cd ${HOME}/projects/semeval2020 || exit

language=english 
preproc=token  # or lemma
model=bert-base-uncased  # bert-base-german-cased, bert-base-multilingual-cased, af-ai-center/bert-base-swedish-uncased
epochs=20
bsz=32
max_seq_len=200
lr=0.0001 
warmup=10000
log_every=10000

echo "Run script"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/run_mlm_wwm.py \
	--model_type bert \
	--model_name_or_path $model \
	--train_file finetuning_corpora/${language}/${preproc}/all.txt \
	--targets_file finetuning_corpora/${language}/targets/target_forms.csv \
	--do_train \
	--output_dir finetuned_bert/${language}/ \
	--overwrite_output_dir \
	--num_train_epochs ${epochs} \
	--per_device_train_batch_size ${bsz} \
	--per_device_eval_batch_size ${bsz} \
	--logging_steps ${log_every} \
	--warmup_steps ${warmup} \
	--learning_rate ${lr} \
	--max_seq_length ${max_seq_len} \
	--load_best_model_at_end \
	--save_total_limit 1 \
	--overwrite_cache \
	--fp16 \
	&> out/finetune_${language}_${preproc}_bsz${bsz}

