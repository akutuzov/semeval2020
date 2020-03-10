#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=test_data_internal

# Make folders
mkdir -p $data/results/

### TASK 2
# Iterate over languages
declare -a languages=(english german russian)
for language in "${languages[@]}"
do
    # Make folders
    mkdir -p $data/results/$language/
    declare -a models=(incremental single)

    for method in "${models[@]}"
    do
    ### Diversity delta ###
	python3 code/elmo/get_coeffs.py -t $data/$language/targets.txt -i0 elmo_embeddings/test/$method/$language/top/corpus1.npz -i1 elmo_embeddings/test/$method/$language/top/corpus2.npz > $data/results/$language/elmo_${method}_top.csv
	python3 code/elmo/get_coeffs.py -t $data/$language/targets.txt -i0 elmo_embeddings/test/$method/$language/average/corpus1.npz -i1 elmo_embeddings/test/$method/$language/average/corpus2.npz > $data/results/$language/elmo_${method}_mean.csv

    ### Cosine similarity baselines ###
	# simple average:
	python3 code/cosine_baseline.py -t $data/$language/targets.txt -i0 elmo_embeddings/test/$method/$language/top/corpus1.npz -i1 elmo_embeddings/test/$method/$language/top/corpus2.npz > $data/results/$language/elmo-cosine_${method}_top.csv
	python3 code/cosine_baseline.py -t $data/$language/targets.txt -i0 elmo_embeddings/test/$method/$language/average/corpus1.npz -i1 elmo_embeddings/test/$method/$language/average/corpus2.npz > $data/results/$language/elmo-cosine_${method}_mean.csv
	
	# PCA:
	python3 code/cosine_baseline.py -t $data/$language/targets.txt -i0 elmo_embeddings/test/$method/$language/top/corpus1.npz -i1 elmo_embeddings/test/$method/$language/top/corpus2.npz -m pca > $data/results/$language/elmo-cosine-pca_${method}_top.csv
	python3 code/cosine_baseline.py -t $data/$language/targets.txt -i0 elmo_embeddings/test/$method/$language/average/corpus1.npz -i1 elmo_embeddings/test/$method/$language/average/corpus2.npz -m pca > $data/results/$language/elmo-cosine-pca_${method}_mean.csv
        done
done


### TASK 1
declare -a languages=(russian)
for language in "${languages[@]}"
do
    mkdir -p $data/results/$language/inventories
    declare -a models=(bigone incremental single)
    for method in "${models[@]}"
	do
	python3 code/cluster.py $data/$language/targets.txt elmo_embeddings/test/$method/$language/top/corpus1.npz elmo_embeddings/test/$method/$language/top/corpus2.npz $data/results/$language/inventories/elmo-cluster_${method}_top1 $data/results/$language/inventories/elmo-cluster_${method}_top2
	python3 code/cluster.py $data/$language/targets.txt elmo_embeddings/test/$method/$language/average/corpus1.npz elmo_embeddings/test/$method/$language/average/corpus2.npz $data/results/$language/inventories/elmo-cluster_${method}_mean1 $data/results/$language/inventories/elmo-cluster_${method}_mean2
	python3 code/pred_from_senses.py -i0 $data/results/$language/inventories/elmo-cluster_${method}_top1.csv -i1 $data/results/$language/inventories/elmo-cluster_${method}_top2.csv > $data/results/$language/elmo-cluster_${method}_top_classes.csv
	python3 code/pred_from_senses.py -i0 $data/results/$language/inventories/elmo-cluster_${method}_mean1.csv -i1 $data/results/$language/inventories/elmo-cluster_${method}_mean2.csv > $data/results/$language/elmo-cluster_${method}_mean_classes.csv

	python3 code/pred_from_senses.py --strength=True -i0 $data/results/$language/inventories/elmo-cluster_${method}_top1.csv -i1 $data/results/$language/inventories/elmo-cluster_${method}_top2.csv > $data/results/$language/elmo-cluster_${method}_top.csv
	python3 code/pred_from_senses.py --strength=True -i0 $data/results/$language/inventories/elmo-cluster_${method}_mean1.csv -i1 $data/results/$language/inventories/elmo-cluster_${method}_mean2.csv > $data/results/$language/elmo-cluster_${method}_mean.csv
	done
done
