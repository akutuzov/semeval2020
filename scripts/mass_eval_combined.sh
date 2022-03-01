#! /bin/bash

for lang in english latin swedish german italian
do
    echo ${lang}
    for method in cos apd apd_cos_geom
	do
	    for gram in morph synt morphsynt
		do
		    mkdir -p results2022/evaluation/combined/${method}_${gram}
		    python3 ../code/eval.py results2022/combined/${lang}/${method}_${gram}_binary.tsv results2022/combined/${lang}/${method}_${gram}_graded.tsv gold_scores/${lang}/binary.txt gold_scores/${lang}/graded.txt > results2022/evaluation/combined/${method}_${gram}/${lang}.tsv
		done
	done
done