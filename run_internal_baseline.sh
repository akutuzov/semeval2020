#!/usr/bin/env bash

shopt -s expand_aliases
source ~/.bash_profile
# Get data
data=test_data_internal

# Make folders
mkdir -p $data/matrices/
mkdir -p $data/results/


# Iterate over languages
declare -a languages=(english german russian)
for language in "${languages[@]}"
do
    # Make folders
    mkdir -p $data/matrices/$language/
    mkdir -p $data/results/$language/
    ### Baseline 1: Normalized Frequency Difference (FD) ###
    # Get normalized (-n) frequencies for both corpora
    python3 code/freq.py -n $data/$language/corpus1/lemma $data/$language/targets.txt $data/results/$language/freq_corpus1.csv
    python3 code/freq.py -n $data/$language/corpus2/lemma $data/$language/targets.txt $data/results/$language/freq_corpus2.csv

    # Subtract frequencies and store absolute (-a) difference
    python3 code/diff.py -a $data/$language/targets.txt $data/results/$language/freq_corpus1.csv $data/results/$language/freq_corpus2.csv $data/results/$language/fd.csv
    # Classify results into two classes according to threshold
    python3 code/class.py $data/results/$language/fd.csv $data/results/$language/fd_classes.csv 0.0000005

    ### Baseline 2: Count Vectors with Column Intersection and Cosine Distance (CNT+CI+CD) ###
    # Get co-occurrence matrices for both corpora
    python3 code/cnt.py $data/$language/corpus1/lemma $data/matrices/$language/cnt_matrix1 1
    python3 code/cnt.py $data/$language/corpus2/lemma $data/matrices/$language/cnt_matrix2 1

    # Align matrices with Column Intersection
    python3 code/ci.py $data/matrices/$language/cnt_matrix1 $data/matrices/$language/cnt_matrix2 $data/matrices/$language/cnt_matrix1_aligned $data/matrices/$language/cnt_matrix2_aligned

    # Load matrices and calculate Cosine Distance
    python3 code/cd.py $data/$language/targets.txt $data/matrices/$language/cnt_matrix1_aligned $data/matrices/$language/cnt_matrix2_aligned $data/results/$language/cnt_ci_cd.csv
    # Classify results into two classes according to threshold
    python3 code/class.py $data/results/$language/cnt_ci_cd.csv $data/results/$language/cnt_ci_cd_classes.csv 0.4

    ### Make answer files for submission ###
    # Baseline 1
    # mkdir -p results/answer/task2/ && cp results/$language/fd.csv results/answer/task2/$language.txt
    # mkdir -p results/answer/task1/ && cp results/$language/fd_classes.csv results/answer/task1/$language.txt
    # cd results/ && zip -r answer_fd.zip answer/ && rm -r answer/ && cd ..
    # Baseline 2
    # mkdir -p results/answer/task2/ && cp results/$language/cnt_ci_cd.csv results/answer/task2/$language.txt
    # mkdir -p results/answer/task1/ && cp results/$language/cnt_ci_cd_classes.csv results/answer/task1/$language.txt
    # cd results/ && zip -r answer_cnt_ci_cd.zip answer/ && rm -r answer/ && cd ..
    
done
