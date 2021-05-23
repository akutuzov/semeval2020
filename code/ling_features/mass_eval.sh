#! /bin/bash

LANG=${1}
THR=${2}

for feat in morph synt
do
    echo "Evaluating ${feat}"
    python3 ../eval.py results/${LANG}_${feat}_${THR}_binary.tsv results/${LANG}_${feat}_${THR}_graded.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt
done

echo "Evaluating averaged"
python3 ../eval.py results/${LANG}_binary_${THR}.tsv results/${LANG}_graded_${THR}.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt
