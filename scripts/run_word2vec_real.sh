#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=results_real

gold=test_data_public

declare -a methods=(${1})

### TASK 2
# Iterate over languages
declare -a languages=(english german swedish latin)
for language in "${languages[@]}"
do
    echo ${language}
    # Make folders
    for method in "${methods[@]}"
    do
	python3 code/word2vec_baseline.py -m ${method} -e models/word2vec/$language/ -t $gold/$language/targets.txt > $data/results/$language/word2vec_${method}.txt
    done

done