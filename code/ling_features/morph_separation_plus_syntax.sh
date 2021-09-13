#! /bin/bash

for LANG in english german latin swedish italian
do
echo
echo ${LANG}

distance=cos
aggregate=max


echo averaging

python3 -W ignore compare_ling.py --input1 ../../data/features/${LANG}/corpus1_synt.json --input2 ../../data/features/${LANG}/corpus2_synt.json --output results/${LANG}_synt_${distance}_5 --filtering 5 --distance ${distance}

python3 -W ignore compare_ling.py --input1 ../../data/features/${LANG}/corpus1_morph.json --input2 ../../data/features/${LANG}/corpus2_morph.json --output results/${LANG}_separate_${distance}_${aggregate}_5 --filtering 5 --distance ${distance} --aggregate ${aggregate} --separation 2step

for task in binary graded
do
	    python3 merge.py -i1 results/${LANG}_separate_${distance}_${aggregate}_5_${task}.tsv -i2 results/${LANG}_synt_${distance}_5_${task}.tsv > results/${LANG}_synt_separate_${distance}_${aggregate}_5_${task}.tsv
done
	
python3 ../eval.py results/${LANG}_synt_separate_${distance}_${aggregate}_5_binary.tsv results/${LANG}_synt_separate_${distance}_${aggregate}_5_graded.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt


echo combination

python3 -W ignore compare_ling.py --input1 ../../data/features/${LANG}/corpus1_morph.json --input2 ../../data/features/${LANG}/corpus2_morph.json --output results/${LANG}_separate_${distance}_${aggregate}_5_added_syntax --filtering 5 --distance ${distance} --aggregate ${aggregate} --separation 2step --added_features1 ../../data/features/${LANG}/corpus1_synt.json  --added_features2 ../../data/features/${LANG}/corpus2_synt.json 

python3 ../eval.py results/${LANG}_separate_${distance}_${aggregate}_5_added_syntax_binary.tsv results/${LANG}_separate_${distance}_${aggregate}_5_added_syntax_graded.tsv ../../test_data_truth/task1/${LANG}.txt ../../test_data_truth/task2/${LANG}.txt

done

       
    
	
     
