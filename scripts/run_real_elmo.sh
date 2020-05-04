#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=results_real

our_answer=elmo_single_top_ap_cosine

gold=test_data_public

# Make folders
mkdir -p $data/${our_answer}/answer/task1
mkdir -p $data/${our_answer}/answer/task2

### TASK 2
# Iterate over languages
declare -a languages=(english german swedish latin)
for language in "${languages[@]}"
do
    # Make folders
    mkdir -p $data/results/$language/
    declare -a models=(single)

    for method in "${models[@]}"
    do
        ### Cosine similarity  ###
        # simple average:
	      python3 code/cosine_baseline.py -t $gold/$language/targets.txt -i0 elmo_embeddings/real/$method/$language/top/corpus1.npz -i1 elmo_embeddings/real/$method/$language/top/corpus2.npz > $data/${our_answer}/answer/task2/$language.txt
	    done

### TASK 1
    # Make folders
    mkdir -p $data/${our_answer}/$language/inventories
    declare -a models=(single)
    for method in "${models[@]}"
	do
	python3 code/cluster.py $gold/$language/targets.txt elmo_embeddings/real/$method/$language/top/corpus1.npz elmo_embeddings/real/$method/$language/top/corpus2.npz $data/${our_answer}/$language/inventories/elmo-cluster_${method}_top1 $data/${our_answer}/$language/inventories/elmo-cluster_${method}_top2
	python3 code/pred_from_senses.py -i0 $data/${our_answer}/$language/inventories/elmo-cluster_${method}_top1.csv -i1 $data/${our_answer}/$language/inventories/elmo-cluster_${method}_top2.csv > $data/${our_answer}/answer/task1/$language.txt

	done

done