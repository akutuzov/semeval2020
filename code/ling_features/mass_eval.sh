#! /bin/bash

LANG=${1}

for feat in morph synt
do
    echo "Evaluating ${feat}"
    python3 eval.py results/${LANG}_${feat}_binary.tsv results/${LANG}_${feat}_graded.tsv gold/task1/${LANG}.txt gold/task2/${LANG}.txt
done

echo "Evaluating averaged"
python3 eval.py results/${LANG}_binary.tsv results/${LANG}_graded.tsv gold/task1/${LANG}.txt gold/task2/${LANG}.txt