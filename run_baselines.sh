#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile

data=test_data_public

mkdir -p matrices/
mkdir -p results/


# Iterate over languages
declare -a languages=(english german latin swedish)
for language in "${languages[@]}"
do
    
    # Make folders
    mkdir -p matrices/$language/
    mkdir -p results/$language/

    ### Baseline 1: Normalized Frequency Difference (FD) ###
    # Get normalized (-n) frequencies for both corpora
    python3 code/freq.py -n $data/$language/corpus1/lemma $data/$language/targets.txt results/$language/freq_corpus1.csv
    python3 code/freq.py -n $data/$language/corpus2/lemma $data/$language/targets.txt results/$language/freq_corpus2.csv

    # Subtract frequencies and store absolute (-a) difference
    python3 code/diff.py -a $data/$language/targets.txt results/$language/freq_corpus1.csv results/$language/freq_corpus2.csv results/$language/fd.csv

    # Classify results into two classes according to threshold
    python3 code/class.py results/$language/fd.csv results/$language/fd_classes.csv 0.0000005


    ### Baseline 2: Count Vectors with Column Intersection and Cosine Distance (CNT+CI+CD) ###

    # Get co-occurrence matrices for both corpora
    python3 code/cnt.py $data/$language/corpus1/lemma matrices/$language/cnt_matrix1 1
    python3 code/cnt.py $data/$language/corpus2/lemma matrices/$language/cnt_matrix2 1

    # Align matrices with Column Intersection
    python3 code/ci.py matrices/$language/cnt_matrix1 matrices/$language/cnt_matrix2 matrices/$language/cnt_matrix1_aligned matrices/$language/cnt_matrix2_aligned

    # Load matrices and calculate Cosine Distance
    python3 code/cd.py $data/$language/targets.txt matrices/$language/cnt_matrix1_aligned matrices/$language/cnt_matrix2_aligned results/$language/cnt_ci_cd.csv

    # Classify results into two classes according to threshold
    python3 code/class.py results/$language/cnt_ci_cd.csv results/$language/cnt_ci_cd_classes.csv 0.4

done
