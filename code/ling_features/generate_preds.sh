#! /bin/bash

LANG=${1}

for feat in morph synt
do
    python3 compare_ling.py --input1 ../../data/features/${LANG}/corpus1_${feat}.json --input2 ../../data/features/${LANG}/corpus2_${feat}.json --output results/${LANG}_${feat}
done

for task in binary graded
do
    python3 merge.py -i1 results/${LANG}_morph_${task}.tsv -i2 results/${LANG}_synt_${task}.tsv > results/${LANG}_${task}.tsv
done
