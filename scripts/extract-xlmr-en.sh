#!/bin/bash
source ~/.bash_profile

# Load all necessary modules.
#module load 2019

# Load virtual environment.
#source ${HOME}/projects/erp/venv/bin/activate

# Set cache directory for Huggingface preprocessed datasets.
export HF_DATASETS_CACHE="/project/dmg_data/lsc/.cache/"

language=english
context=256
bsz=4

python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0  \
	$HOME/projects/semeval2020/code/xlmr/extract.py \
	--model_name_or_path /project/dmg_data/lsc/models/xlmr/finetuned/$language \
        --corpus_path $HOME/projects/semeval2020/finetuning_corpora/$language/token/1000.txt \
        --targets_path $HOME/projects/semeval2020/finetuning_corpora/$language/targets/target_forms.csv \
        --output_path /project/dmg_data/lsc/embeds/xlmr/finetuned/$language/semeval_en_test1000.tsv \
        --context_window $context \
	--batch_size $bsz \
        &> ${HOME}/projects/semeval2020/out/extract-xlmr-en_$(date +"%FT%T")


