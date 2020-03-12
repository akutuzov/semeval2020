#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=results_real

method=top

our_answer=bert_${method}_ap_cosine

gold=test_data_public

# Make folders
mkdir -p $data/${our_answer}/answer/task1
mkdir -p $data/${our_answer}/answer/task2

### TASK 2
# Iterate over languages
# declare -a languages=(english german swedish latin)
declare -a languages=(english german latin)
#for language in "${languages[@]}"
#do
    # Make folders
#    mkdir -p $data/results/$language/
        ### Cosine similarity  ###
	# simple average:
#	python3 code/cosine_baseline.py -t $gold/$language/targets.txt -i0 bert_embeddings/real/$language/$method/corpus1.npz -i1 bert_embeddings/real/$language/$method/corpus2.npz > $data/${our_answer}/answer/task2/$language.txt
	

### TASK 1
    # Make folders
#    mkdir -p $data/${our_answer}/$language/inventories
#	python3 code/cluster.py $gold/$language/targets.txt bert_embeddings/real/$language/$method/corpus1.npz bert_embeddings/real/$language/$method/corpus2.npz $data/${our_answer}/$language/inventories/bert-cluster_${method}_1 $data/${our_answer}/$language/inventories/bert-cluster_${method}_2

language=english
python3 code/pred_from_senses.py -t 0.2 -i0 $data/${our_answer}/$language/inventories/bert-cluster_${method}_1.csv -i1 $data/${our_answer}/$language/inventories/bert-cluster_${method}_2.csv > $data/${our_answer}/answer/task1/$language.txt

language=german
python3 code/pred_from_senses.py -t 0.3 -i0 $data/${our_answer}/$language/inventories/bert-cluster_${method}_1.csv -i1 $data/${our_answer}/$language/inventories/bert-cluster_${method}_2.csv > $data/${our_answer}/answer/task1/$language.txt

language=latin
python3 code/pred_from_senses.py -t 0.9 -i0 $data/${our_answer}/$language/inventories/bert-cluster_${method}_1.csv -i1 $data/${our_answer}/$language/inventories/bert-cluster_${method}_2.csv > $data/${our_answer}/answer/task1/$language.txt


#done
