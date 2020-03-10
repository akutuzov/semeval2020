#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=test_data_internal

# Make folders
mkdir -p $data/results/


# Iterate over languages
#declare -a languages=(english german russian)
declare -a languages=(russian)
for language in "${languages[@]}"
do
    # Make folders
    mkdir -p $data/results/$language/

    ### Word2vec baseline: simple cosine similarity between (aligned) vectors ###

    # No alignment:
    python3 code/word2vec_baseline.py -e static_embeddings/word2vec/$language/ -t $data/$language/targets.txt > $data/results/$language/word2vec_incremental.csv

    # With alignment:
    python3 code/word2vec_baseline.py -m align -e static_embeddings/word2vec/$language/ -t $data/$language/targets.txt > $data/results/$language/word2vec_procrustes.csv

    ### Word2vec WSI baseline: difference between the number of detected senses ###

    # No alignment:
    python3 code/graph_induction.py --emb static_embeddings/word2vec/$language/corpus1.model --eval $data/$language/targets.txt --outfile $data/results/$language/model1.inventory.csv
    python3 code/graph_induction.py --emb static_embeddings/word2vec/$language/corpus2.model --eval $data/$language/targets.txt --outfile $data/results/$language/model2.inventory.csv
    
    THRESHOLD=0.09
    python3 code/pred_from_senses.py -t ${THRESHOLD} -i0 $data/results/$language/model1.inventory.csv -i1 $data/results/$language/model2.inventory.csv > $data/results/$language/wsi_scratch_classes.csv
    python3 code/pred_from_senses.py -t ${THRESHOLD} --strength=True -i0 $data/results/$language/model1.inventory.csv -i1 $data/results/$language/model2.inventory.csv > $data/results/$language/wsi_scratch.csv

    # With alignment:
     python3 code/graph_induction.py --emb static_embeddings/word2vec/$language/corpus2_incremental.model --eval $data/$language/targets.txt --outfile $data/results/$language/model2_incremental.inventory.csv

    python3 code/pred_from_senses.py -t ${THRESHOLD} -i0 $data/results/$language/model1.inventory.csv -i1 $data/results/$language/model2_incremental.inventory.csv > $data/results/$language/wsi_incremental_classes.csv
    python3 code/pred_from_senses.py -t ${THRESHOLD} --strength=True -i0 $data/results/$language/model1.inventory.csv -i1 $data/results/$language/model2_incremental.inventory.csv > $data/results/$language/wsi_incremental.csv

done
