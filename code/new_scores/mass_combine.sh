#! /bin/bash

for lang in english latin swedish german italian
do
    echo ${lang}
    for method in cos apd apd_cos_geom
	do
	    for gram in morph synt morphsynt
		do
		    mkdir -p results2022/combined/${lang}
		    python3 combine_scores.py -i0 results2022/${lang}/${method}.tsv -i1 results2022/profile_predictions/${lang}_${gram}_graded.tsv -o results2022/combined/${lang}/${method}_${gram}_graded.tsv
                    python3 binary_scores.py -i results2022/combined/${lang}/${method}_${gram}_graded.tsv -o results2022/combined/${lang}/${method}_${gram}_binary.tsv
		done
    done
done