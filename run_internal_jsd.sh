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
    declare -a models=(bigone single)

    for method in "${models[@]}"
    do
        ### JSD divergence ###
	python3 code/jsd.py $data/$language/targets.txt elmo_embeddings/test/$method/$language/top/corpus1.npz elmo_embeddings/test/$method/$language/top/corpus2.npz $data/results/$language/elmo-jsd_${method}_top
	python3 code/jsd.py $data/$language/targets.txt elmo_embeddings/test/$method/$language/average/corpus1.npz elmo_embeddings/test/$method/$language/average/corpus2.npz $data/results/$language/elmo-jsd_${method}_mean
    done
done


