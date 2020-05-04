#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=results_real

# Iterate over languages
declare -a languages=(english german latin swedish)

declare -a layers=(top average)
declare -a models=(bigone fine single incremental)
declare -a methods=(cosine distance jsd diversity)

for language in "${languages[@]}"
do
    # echo ${language}
    for method in "${methods[@]}"
    do
    for model in "${models[@]}"
    do
	for layer in "${layers[@]}"
	do
	    echo "${language}_${method} ${model} ${layer}_$(python3 code/eval.py $data/results/$language/${method}_classes.csv $data/results/$language/${method}_${model}_${layer}.txt gold_real/task1/${language}.txt gold_real/task2/${language}.txt)"
	done
    done
    done
done

