#! /bin/bash

for lang in english latin swedish german italian
do
    echo ${lang}
    for method in cos apd apd_cos_geom
	do
	    mkdir -p evaluation/${method}
	    python3 eval.py results2022/${lang}/${method}_binary.tsv results2022/${lang}/${method}.tsv gold_scores/${lang}/binary.txt gold_scores/${lang}/graded.txt > evaluation/${method}/${lang}.tsv
	done
    for gram in morph synt morphsynt
    do
	mkdir -p evaluation/${gram}
	python3 eval.py results2022/profile_predictions/${lang}_${gram}_binary.tsv results2022/profile_predictions/${lang}_${gram}_graded.tsv gold_scores/${lang}/binary.txt gold_scores/${lang}/graded.txt >> evaluation/${gram}/${lang}.tsv
    done
done