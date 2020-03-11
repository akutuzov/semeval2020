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
    declare -a methods=(fd cnt_ci_cd wsi_scratch wsi_incremental elmo-cluster_bigone_mean elmo-cluster_bigone_top elmo-cluster_incremental_mean elmo-cluster_incremental_top elmo-cluster_single_mean elmo-cluster_single_top)

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
    declare -a methods=(fd cnt_ci_cd word2vec_incremental word2vec_procrustes wsi_scratch wsi_incremental elmo_single_top elmo_single_mean elmo_incremental_top elmo_incremental_mean elmo-cosine_single_top elmo-cosine_single_mean elmo-cosine-pca_single_top elmo-cosine-pca_single_mean bert-jsd_average bert-jsd_last4 bert-jsd_top elmo-jsd_bigone_mean elmo-jsd_bigone_top elmo-jsd_single_mean elmo-jsd_single_top)
    for method in "${methods[@]}"
    do
        echo $method
        python3 code/eval.py $data/results/$language/${method}_classes.csv $data/results/$language/${method}.csv gold/$language/task2/true_answers.tsv gold/$language/task2/true_answers.tsv
    done
done
