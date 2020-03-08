#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=test_data_internal

# Iterate over languages
declare -a languages=(russian)
for language in "${languages[@]}"
do
    echo ${language}
    declare -a methods=(fd cnt_ci_cd wsi_scratch wsi_incremental)
    for method in "${methods[@]}"
    do
        echo $method
        python3 code/eval.py $data/results/$language/${method}_classes.csv $data/results/$language/${method}.csv gold/$language/task1/true_answers.tsv gold/$language/task1/true_answers.tsv
    done
done

declare -a languages=(english german)
for language in "${languages[@]}"
do
    echo ${language}
    declare -a methods=(fd cnt_ci_cd word2vec_incremental word2vec_procrustes)
    for method in "${methods[@]}"
    do
        echo $method
        python3 code/eval.py $data/results/$language/${method}_classes.csv $data/results/$language/${method}.csv gold/$language/task2/true_answers.tsv gold/$language/task2/true_answers.tsv
    done
done
