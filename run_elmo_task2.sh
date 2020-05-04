#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data

data=results_real

gold=test_data_public

declare -a layers=(${1}) # top or average

### TASK 2
# Iterate over languages
declare -a languages=(english german swedish latin)
for language in "${languages[@]}"
do
    echo ${language}
    # Make folders
    mkdir -p $data/results/$language/
    declare -a models=(bigone fine single incremental)
    for method in "${models[@]}"
    do
        
	for layer in "${layers[@]}"
	do
	    ### Cosine similarity  ###
	    python3 code/cosine.py -t $gold/$language/targets.txt -i0 elmo_embeddings/real/$method/$language/$layer/corpus1.npz -i1 elmo_embeddings/real/$method/$language/$layer/corpus2.npz > $data/results/$language/cosine_${method}_${layer}.txt
	    
	    ### JSD divergence ###
	    python3 code/jsd.py $gold/$language/targets.txt elmo_embeddings/real/$method/$language/$layer/corpus1.npz elmo_embeddings/real/$method/$language/$layer/corpus2.npz $data/results/$language/jsd_${method}_${layer}.txt
	    
	    ### Pairwise difference ###
	    python3 code/distance.py $gold/$language/targets.txt elmo_embeddings/real/$method/$language/$layer/corpus1.npz elmo_embeddings/real/$method/$language/$layer/corpus2.npz $data/results/$language/distance_${method}_${layer}.txt
	    
	    ### Diversity difference ###
	    python3 code/get_coeffs.py --target $gold/$language/targets.txt --input0 elmo_embeddings/real/$method/$language/$layer/corpus1.npz --input1 elmo_embeddings/real/$method/$language/$layer/corpus2.npz > $data/results/$language/diversity_${method}_${layer}.txt
	    
	    ### Relatedness difference ###
	    python3 code/relatedness.py $gold/$language/targets.txt elmo_embeddings/real/$method/$language/$layer/corpus1.npz elmo_embeddings/real/$method/$language/$layer/corpus2.npz $data/results/$language/relatedness_${method}_${layer}.txt
	    
	done
	
    done

done