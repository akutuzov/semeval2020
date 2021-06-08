#! /bin/bash

distance=cos
agregate=avg

echo
echo 12
echo morphology

	python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus1_morph.json --input2 ../../data/features/russian/corpus2_morph.json --output results/russian12_separate_${distance}_${agregate}_5 --threshold 5 --distance ${distance} --agregate ${agregate} --separation 2step


	python3 ../eval.py results/russian12_separate_${distance}_${agregate}_5_binary.tsv results/russian12_separate_${distance}_${agregate}_5_graded.tsv  ../../test_data_truth/task2/russian1.txt ../../test_data_truth/task2/russian1.txt


echo averaging
	
	python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus1_synt.json --input2 ../../data/features/russian/corpus2_synt.json --output results/russian12_synt_${distance}_5 --threshold 5 --distance ${distance}

	
	for task in binary graded
	do
	    python3 merge.py -i1 results/russian12_separate_${distance}_${agregate}_5_${task}.tsv -i2 results/russian12_synt_${distance}_5_${task}.tsv > results/russian12_synt_separate_${distance}_${agregate}_5_${task}.tsv
	done

	python3 ../eval.py results/russian12_synt_separate_${distance}_${agregate}_5_binary.tsv  results/russian12_synt_separate_${distance}_${agregate}_5_graded.tsv  ../../test_data_truth/task2/russian1.txt ../../test_data_truth/task2/russian1.txt


echo combination
       
    python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus1_morph.json --input2 ../../data/features/russian/corpus2_morph.json --output results/russian12_separate_${distance}_${agregate}_5_added_syntax --threshold 5 --distance ${distance} --agregate ${agregate} --separation 2step --added_features1 ../../data/features/russian/corpus1_synt.json --added_features2 ../../data/features/russian/corpus2_synt.json

    python3 ../eval.py results/russian12_separate_${distance}_${agregate}_5_added_syntax_binary.tsv results/russian12_separate_${distance}_${agregate}_5_added_syntax_graded.tsv  ../../test_data_truth/task2/russian1.txt ../../test_data_truth/task2/russian1.txt
    

echo
echo 23
echo morphology

	python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus2_morph.json --input2 ../../data/features/russian/corpus3_morph.json --output results/russian23_separate_${distance}_${agregate}_5 --threshold 5 --distance ${distance} --agregate ${agregate} --separation 2step


	python3 ../eval.py results/russian23_separate_${distance}_${agregate}_5_binary.tsv results/russian23_separate_${distance}_${agregate}_5_graded.tsv  ../../test_data_truth/task2/russian2.txt ../../test_data_truth/task2/russian2.txt


echo averaging
	
	python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus2_synt.json --input2 ../../data/features/russian/corpus3_synt.json --output results/russian23_synt_${distance}_5 --threshold 5 --distance ${distance}

	
	for task in binary graded
	do
	    python3 merge.py -i1 results/russian23_separate_${distance}_${agregate}_5_${task}.tsv -i2 results/russian23_synt_${distance}_5_${task}.tsv > results/russian23_synt_separate_${distance}_${agregate}_5_${task}.tsv
	done

	python3 ../eval.py results/russian23_synt_separate_${distance}_${agregate}_5_binary.tsv  results/russian23_synt_separate_${distance}_${agregate}_5_graded.tsv  ../../test_data_truth/task2/russian2.txt ../../test_data_truth/task2/russian2.txt


echo combination
       
    python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus2_morph.json --input2 ../../data/features/russian/corpus3_morph.json --output results/russian23_separate_${distance}_${agregate}_5_added_syntax --threshold 5 --distance ${distance} --agregate ${agregate} --separation 2step --added_features1 ../../data/features/russian/corpus2_synt.json --added_features2 ../../data/features/russian/corpus3_synt.json

    python3 ../eval.py results/russian23_separate_${distance}_${agregate}_5_added_syntax_binary.tsv results/russian23_separate_${distance}_${agregate}_5_added_syntax_graded.tsv  ../../test_data_truth/task2/russian2.txt ../../test_data_truth/task2/russian2.txt




echo
echo 13
echo morphology

	python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus1_morph.json --input2 ../../data/features/russian/corpus3_morph.json --output results/russian13_separate_${distance}_${agregate}_5 --threshold 5 --distance ${distance} --agregate ${agregate} --separation 2step


	python3 ../eval.py results/russian13_separate_${distance}_${agregate}_5_binary.tsv results/russian13_separate_${distance}_${agregate}_5_graded.tsv  ../../test_data_truth/task2/russian3.txt ../../test_data_truth/task2/russian3.txt


echo averaging
	
	python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus1_synt.json --input2 ../../data/features/russian/corpus3_synt.json --output results/russian13_synt_${distance}_5 --threshold 5 --distance ${distance}

	
	for task in binary graded
	do
	    python3 merge.py -i1 results/russian13_separate_${distance}_${agregate}_5_${task}.tsv -i2 results/russian13_synt_${distance}_5_${task}.tsv > results/russian13_synt_separate_${distance}_${agregate}_5_${task}.tsv
	done

	python3 ../eval.py results/russian13_synt_separate_${distance}_${agregate}_5_binary.tsv  results/russian13_synt_separate_${distance}_${agregate}_5_graded.tsv  ../../test_data_truth/task2/russian3.txt ../../test_data_truth/task2/russian3.txt


echo combination
       
    python3 -W ignore compare_ling.py --input1 ../../data/features/russian/corpus1_morph.json --input2 ../../data/features/russian/corpus3_morph.json --output results/russian13_separate_${distance}_${agregate}_5_added_syntax --threshold 5 --distance ${distance} --agregate ${agregate} --separation 2step --added_features1 ../../data/features/russian/corpus1_synt.json --added_features2 ../../data/features/russian/corpus3_synt.json

    python3 ../eval.py results/russian13_separate_${distance}_${agregate}_5_added_syntax_binary.tsv results/russian13_separate_${distance}_${agregate}_5_added_syntax_graded.tsv  ../../test_data_truth/task2/russian3.txt ../../test_data_truth/task2/russian3.txt
    



    
       
    
	
     
