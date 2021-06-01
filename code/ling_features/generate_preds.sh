#! /bin/bash

LANG=${1}
THR=${2}

for feat in morph synt
do
    python3 compare_ling.py --input1 ../../data/features/${LANG}/corpus1_${feat}.json --input2 ../../data/features/${LANG}/corpus2_${feat}.json --output results/${LANG}_${feat}_${THR} --threshold ${THR} --separation yes
done

for task in binary graded
do
    python3 merge.py -i1 results/${LANG}_morph_${THR}_${task}.tsv -i2 results/${LANG}_synt_${THR}_${task}.tsv > results/${LANG}_${task}_${THR}.tsv
done
