#! /bin/bash

LANG=${1}

for distance in cos jsd
do
    echo
    echo ${distance}
    for agregate in max avg
    do
	echo ${agregate}
	python3 -W ignore compare_ling.py --input1 ../../data/features/${LANG}/corpus1_morph.json --input2 ../../data/features/${LANG}/corpus2_morph.json --output results/${LANG}_separate_${distance}_${agregate} --distance ${distance} --agregate ${agregate} --separation 2step

	python3 ../eval.py results/${LANG}_separate_${distance}_${agregate}_binary.tsv results/${LANG}_separate_${distance}_${agregate}_graded.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt
	
	echo thr 5
    
	python3 -W ignore compare_ling.py --input1 ../../data/features/${LANG}/corpus1_morph.json --input2 ../../data/features/${LANG}/corpus2_morph.json --output results/${LANG}_separate_${distance}_${agregate}_5 --threshold 5 --distance ${distance} --agregate ${agregate} --separation 2step

	python3 ../eval.py results/${LANG}_separate_${distance}_${agregate}_5_binary.tsv results/${LANG}_separate_${distance}_${agregate}_5_graded.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt

    echo
    done
done

       
    
	
     
