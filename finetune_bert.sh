#!/bin/bash

echo "Loading virtual environment..."
source ${HOME}/nlp-env/bin/activate

language=english  # german, latin, swedish
model=bert-base-cased  # bert-base-german-cased, bert-base-multilingual-cased, af-ai-center/bert-base-swedish-uncased
epochs=10
batch=64

# substitute the first line below with the following line for multi-gpu training
#python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 code/bert/finetune.py \

python3 code/bert/finetune.py \
	--train_data_file ${HOME}/SemEval/finetuning_corpora/$language/lemma/corpus.txt \
	--output_dir semeval2020/models/bert/$language/finetuned \
	--model_type bert \
	--mlm \
	--do_train \
	--line_by_line \
	--save_steps 100000000 \
	--fp16 \
	--num_train_epochs $epochs \
	--per_gpu_train_batch_size $batch \
	--model_name_or_path $model \
	&> ${HOME}/projects/semeval2020/out/finetune-en
