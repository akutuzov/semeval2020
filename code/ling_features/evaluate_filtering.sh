#! /bin/bash

LANG=${1}

for filtering in none group partial delete
do
    echo ${filtering}
	   
    python3 -W ignore compare_ling.py --input1 ../../data/features/${LANG}/corpus1_reverse_synt.json --input2 ../../data/features/${LANG}/corpus2_reverse_synt.json --output results/${LANG}_reverse_synt_${filtering} --filtering ${filtering}

    python3 ../eval.py results/${LANG}_reverse_synt_${filtering}_binary.tsv results/${LANG}_reverse_synt_${filtering}_graded.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt

    echo thr 5
    
    python3 -W ignore compare_ling.py --input1 ../../data/features/${LANG}/corpus1_reverse_synt.json --input2 ../../data/features/${LANG}/corpus2_reverse_synt.json --output results/${LANG}_reverse_synt_${filtering}_5 --threshold 5 --filtering ${filtering}
    
    python3 ../eval.py results/${LANG}_reverse_synt_${filtering}_5_binary.tsv results/${LANG}_reverse_synt_${filtering}_5_graded.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt     

    echo
    
done
