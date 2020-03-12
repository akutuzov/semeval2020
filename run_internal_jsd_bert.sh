#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=test_data_internal

# Make folders
mkdir -p $data/results/

### TASK 2
# Iterate over languages
declare -a languages=(english german)
for language in "${languages[@]}"
do
    # Make folders
    mkdir -p $data/results/$language/
    declare -a models=(average last4 top)
    DIR=bert_embeddings/single

    for method in "${models[@]}"
    do
        ### JSD divergence on BERT ###
	python3 code/jsd.py $data/$language/targets.txt $DIR/$language/$method/corpus1.npz $DIR/$language/$method/corpus2.npz $data/results/$language/bert-jsd_${method}
    done
done