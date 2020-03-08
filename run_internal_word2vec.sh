#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=test_data_internal

# Make folders
mkdir -p $data/results/


# Iterate over languages
declare -a languages=(english german russian)
for language in "${languages[@]}"
do
    # Make folders
    mkdir -p $data/results/$language/

    ### Word2vec baseline: simple cosine similarity between (aligned) vectors ###

    # No alignment:
    python3 code/word2vec_baseline.py -e static_embeddings/word2vec/$language/ -t test_data_internal/$language/targets.txt > $data/results/$language/word2vec_incremental.csv

    # With alignment:
    python3 code/word2vec_baseline.py -m align -e static_embeddings/word2vec/$language/ -t test_data_internal/$language/targets.txt > $data/results/$language/word2vec_procrustes.csv

done
