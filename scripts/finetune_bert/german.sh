#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=ft-en
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load all necessary modules.
echo "Loading modules..."
module load 2019

cd ${HOME}/projects/semeval2020 || exit

unzip -n test_data_public.zip
data=test_data_public

# English
echo "Downloading data..."
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_ger.zip
unzip -n semeval2020_ulscd_ger.zip
rm -rf $data/german
mv semeval2020_ulscd_ger  $data/german
rm semeval2020_ulscd_ger.zip

echo "Loading virtual environment..."
source ${HOME}/projects/erp/venv/bin/activate

language=german
preproc=lemma
model=bert-base-german-cased
epochs=10
batch=64

echo "Run script"
python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/run_mlm_wwm.py \
	--model_type bert \
	--model_name_or_path $model \
	--train_file finetuning_corpora/${language}/${preproc}/train.txt \
	--validation_file finetuning_corpora/${language}/${preproc}/test.txt \
	--targets_file ${data}/${language}/targets.txt \
	--do_train \
	--do_eval \
	--output_dir finetuned_bert/${language}/ \
	--num_train_epochs ${epochs} \
	--per_device_train_batch_size ${batch} \
	--per_device_eval_batch_size ${batch} \
	--evaluation_strategy epoch \
	--load_best_model_at_end \
	--save_total_limit 1 \
	&> out/finetune_${language}_${preproc}_bsz${batch}

