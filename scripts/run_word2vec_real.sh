#!/usr/bin/env bash

languages=(english german swedish latin norwegian1 norwegian2 italian russian1 russian2 russian3)
for language in "${languages[@]}"
do
    echo ${language}
    # Make folders
	python3 ../code/word2vec_baseline.py -m align -e word2vec/$language/ -t targets/$language/targets.txt > results2022/$language/SGNS.tsv
	python3 ../code/new_scores/binary_scores.py -i ../results2022/${language}/SGNS_raw.tsv -o ../results2022/${language}/SGNS_binary.tsv
	python3 ../code/new_scores/binary_scores.py -i ../results2022/${language}/SGNS_lemma.tsv -o ../results2022/${language}/SGNS_lemma_binary.tsv
done