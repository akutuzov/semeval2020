#!/bin/bash

# Provide project path as argument to this script!
project_path=$1
cd $project_path || exit

unzip -n test_data_public.zip
data=test_data_public

# English
echo "Downloading data..."
wget https://zenodo.org/record/3672950/files/semeval2020_ulscd_swe.zip
unzip -n semeval2020_ulscd_swe.zip
rm -rf $data/swedish
mv semeval2020_ulscd_swe  $data/swedish
rm semeval2020_ulscd_swe.zip

echo "Loading virtual environment..."
source ${HOME}/nlp-env/bin/activate

language=swedish
preproc=lemma
model=af-ai-center/bert-base-swedish-uncased
epochs=10
batch=64

python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/run_mlm_wwm.py \
	--model_type bert \
	--model_name_or_path $model \
	--train_file finetuning_corpora/${language}/${preproc}/train.txt \
	--validation_file finetuning_corpora/${language}/${preproc}/val.txt \
	--targets_file ${data}/${language}/targets.txt \
	--do_train \
	--do_eval \
	--output_dir finetuned_bert/${language}/ \
	--line_by_line \
	--num_train_epochs ${epochs} \
	--per_device_train_batch_size ${batch} \
	--per_device_eval_batch_size ${batch} \
	--evaluation_strategy epoch \
	--load_best_model_at_end \
	--save_total_limit 1 \
	&> out/finetune_${language}_${preproc}_bsz${batch}

